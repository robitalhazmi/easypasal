"""Automated deployment script for Hugging Face Spaces (Plan §8.3 / Task T7.6).

Packages the EasyPasal Gradio application, prebuilt vector and BM25 indices,
and deployment configuration for the Hugging Face Space free CPU Basic Tier.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger("easypasal.deploy_hf_space")


def prepare_space_bundle(output_dir: Path | str = "dist/hf_space") -> Path:
    """Assemble all assets required for Hugging Face Space Docker deployment.

    Args:
        output_dir: Destination staging directory.

    Returns:
        Path to prepared staging directory.
    """
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Copy Space Dockerfile and README
    shutil.copy2("docker/space/Dockerfile", out_path / "Dockerfile")
    shutil.copy2("docker/space/README.md", out_path / "README.md")
    shutil.copy2("pyproject.toml", out_path / "pyproject.toml")

    # 2. Copy source code and configs
    shutil.copytree("src", out_path / "src")
    shutil.copytree("configs", out_path / "configs")

    # 3. Copy prebuilt indices and corpus manifest
    data_dest = out_path / "data"
    data_dest.mkdir(parents=True, exist_ok=True)
    if Path("data/corpus_manifest.csv").exists():
        shutil.copy2("data/corpus_manifest.csv", data_dest / "corpus_manifest.csv")

    if Path("data/index").exists():
        shutil.copytree("data/index", data_dest / "index")
    else:
        logger.warning("No data/index found on disk! Creating placeholder directory.")
        (data_dest / "index").mkdir(parents=True, exist_ok=True)
    # 4. Generate root app.py and requirements.txt for native Gradio SDK Space compatibility
    app_wrapper = (
        '"""Hugging Face Space Gradio entrypoint."""\n'
        "import logging\n"
        "import os\n"
        "import sys\n\n"
        "# Ensure src/ is on Python search path\n"
        'sys.path.insert(0, os.path.abspath("src"))\n\n'
        "from easypasal.generation.pipeline import RAGPipeline\n"
        "from easypasal.monitoring.logger import RequestLogger\n"
        "from easypasal.ui.app import build_ui\n\n"
        "logging.basicConfig(\n"
        '    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"\n'
        ")\n"
        'logger = logging.getLogger("easypasal.space")\n\n'
        "# Initialize RAGPipeline with mock fallback\n"
        "try:\n"
        '    pipeline = RAGPipeline.from_defaults(provider="auto")\n'
        '    logger.info("RAGPipeline loaded successfully.")\n'
        "except Exception as exc:\n"
        '    logger.error("Failed to initialize RAGPipeline: %s", exc)\n'
        "    pipeline = None\n\n"
        "request_logger = RequestLogger(\n"
        '    collector_url=os.getenv("EASYPASAL_COLLECTOR_URL"),\n'
        '    collector_token=os.getenv("EASYPASAL_COLLECTOR_TOKEN", "easypasal-vps-secret-token-v1"),\n'
        ")\n\n"
        "# Export demo at module level for Hugging Face Gradio runner\n"
        "demo = build_ui(pipeline=pipeline, request_logger=request_logger)\n\n"
        'if __name__ == "__main__":\n'
        '    demo.launch(server_name="0.0.0.0", server_port=7860)\n'
    )
    (out_path / "app.py").write_text(app_wrapper, encoding="utf-8")

    requirements_txt = (
        "gradio==6.30.0\n"
        "pydantic>=2.10.0\n"
        "pydantic-settings>=2.6.0\n"
        "httpx>=0.27.0\n"
        "qdrant-client>=1.11.0\n"
        "fastembed>=0.3.0\n"
        "rank-bm25>=0.2.2\n"
        "huggingface-hub>=0.20.0\n"
        "fastapi>=0.115.0\n"
        "uvicorn>=0.32.0\n"
        "tqdm>=4.66.0\n"
        "pyyaml>=6.0\n"
        "numpy>=1.26.0\n"
        "flashrank>=0.2.10\n"
    )
    (out_path / "requirements.txt").write_text(requirements_txt, encoding="utf-8")

    logger.info("Prepared Hugging Face Space staging bundle at %s", out_path)
    return out_path


def measure_cold_start(url: str, timeout: float = 60.0) -> float | None:
    """Measure HTTP cold-start latency against a deployed Hugging Face Space URL.

    Args:
        url: Public Space endpoint URL.
        timeout: Maximum seconds to wait for initial HTTP 200 response.

    Returns:
        Elapsed seconds until wake-up, or None if timed out.
    """
    import urllib.error
    import urllib.request

    logger.info("Measuring cold start for endpoint %s (timeout: %.1fs)...", url, timeout)
    start_time = time.perf_counter()
    deadline = start_time + timeout

    while time.perf_counter() < deadline:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "EasyPasal-ColdStartProbe/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.getcode() == 200:
                    elapsed = time.perf_counter() - start_time
                    logger.info("Space awake! Cold start latency: %.2f seconds.", elapsed)
                    return round(elapsed, 2)
        except Exception:
            time.sleep(2.0)

    logger.warning("Space did not wake up within %.1fs timeout.", timeout)
    return None


def deploy_to_huggingface(
    space_id: str,
    staging_dir: Path | str,
    token: str | None = None,
) -> bool:
    """Deploy packaged Space bundle to Hugging Face Hub.

    Args:
        space_id: Hugging Face Space ID (e.g. 'username/easypasal').
        staging_dir: Path to staging bundle directory.
        token: Optional HF write access token (defaults to HF_TOKEN env var).

    Returns:
        True if upload succeeded, False otherwise.
    """
    try:
        from huggingface_hub import HfApi

        auth_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not auth_token:
            logger.error("HF_TOKEN is not set. Cannot perform automated upload.")
            return False

        api = HfApi(token=auth_token)

        # 1. Check if Space already exists before attempting repo creation
        space_exists = False
        try:
            api.repo_info(repo_id=space_id, repo_type="space")
            space_exists = True
            logger.info(
                "Hugging Face Space %s already exists. Reusing existing repository.", space_id
            )
        except Exception:
            logger.info(
                "Hugging Face Space %s not found. Attempting to create (sdk=docker)...", space_id
            )

        # 2. Create Space if not already present
        if not space_exists:
            try:
                api.create_repo(
                    repo_id=space_id,
                    repo_type="space",
                    space_sdk="docker",
                    exist_ok=True,
                )
            except Exception as create_exc:
                err_msg = str(create_exc)
                if "402" in err_msg or "Payment Required" in err_msg:
                    logger.error(
                        "Hugging Face returned 402 Payment Required: Free personal accounts can no longer create cpu-basic Docker Spaces via API without a PRO subscription.\n"
                        "Resolution Options:\n"
                        "1. If you have or subscribe to Hugging Face PRO, re-run this script.\n"
                        "2. Alternatively, create the Space on the Hugging Face web UI at https://huggingface.co/new-space (e.g. Gradio SDK with ZeroGPU), then re-run this script to upload files directly.\n"
                        "3. Or run the local demo with a free public shareable link: 'uv run python -m easypasal.ui.app --share'."
                    )
                raise create_exc

        logger.info("Uploading bundle from %s to Space %s...", staging_dir, space_id)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy EasyPasal demo with prebuilt index",
        )
        logger.info("Successfully pushed bundle to https://huggingface.co/spaces/%s", space_id)
        return True
    except Exception as exc:
        logger.error("Deployment to Hugging Face failed: %s", exc)
        return False


def main() -> None:
    """CLI entrypoint for Space bundle preparation, deployment, and cold-start verification."""
    parser = argparse.ArgumentParser(
        description="Package and deploy EasyPasal to Hugging Face Space."
    )
    parser.add_argument(
        "--space-id",
        default="robitalhazmi/easypasal",
        help="Hugging Face Space repository ID (e.g. username/easypasal)",
    )
    parser.add_argument(
        "--staging-dir",
        default="dist/hf_space",
        help="Local staging directory for the Space bundle (default: dist/hf_space)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API write token (optional, reads from HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare bundle without uploading to Hugging Face Hub",
    )
    parser.add_argument(
        "--probe-url",
        default=None,
        help="Optional public Space URL to measure cold-start wake-up latency",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.probe_url:
        latency = measure_cold_start(args.probe_url)
        if latency is not None:
            print(f"Cold-start wake-up time: {latency}s")
        else:
            print("Probe timed out.")
        return

    bundle_path = prepare_space_bundle(output_dir=args.staging_dir)

    print("\n" + "=" * 65)
    print("🚀 EasyPasal Hugging Face Space Bundle Staged")
    print("=" * 65)
    print(f"Destination:    {bundle_path.resolve()}")
    print(f"Space Target:   https://huggingface.co/spaces/{args.space_id}")
    print("Architecture:   Docker container on port 7860 (Basic CPU Tier)")
    print("Included Assets:")
    print(f"  • Dockerfile:          {bundle_path / 'Dockerfile'}")
    print(f"  • README & Metadata:   {bundle_path / 'README.md'}")
    print(f"  • Prebuilt Index:      {bundle_path / 'data/index'}")
    print(f"  • Source Code:         {bundle_path / 'src'}")
    print("=" * 65)

    if args.dry_run:
        print("\nDry-run mode active. Upload skipped.")
        print("To deploy manually via git:")
        print(f"  cd {bundle_path}")
        print("  git init && git add . && git commit -m 'Deploy EasyPasal demo'")
        print(f"  git remote add space https://huggingface.co/spaces/{args.space_id}")
        print("  git push --force space main\n")
        return

    auth_token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not auth_token:
        print("\n⚠️ No HF_TOKEN found in environment.")
        print("To upload automatically:")
        print(f"  HF_TOKEN=your_token python scripts/deploy_hf_space.py --space-id {args.space_id}")
        print("\nOr push manually:")
        print(f"  cd {bundle_path}")
        print("  git init && git add . && git commit -m 'Deploy EasyPasal demo'")
        print(f"  git remote add space https://huggingface.co/spaces/{args.space_id}")
        print("  git push --force space main\n")
        return

    success = deploy_to_huggingface(
        space_id=args.space_id,
        staging_dir=bundle_path,
        token=auth_token,
    )
    if success:
        print(f"\n✅ Space deployed successfully: https://huggingface.co/spaces/{args.space_id}\n")
    else:
        print("\n❌ Deployment failed. Check logs above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
