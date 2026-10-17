"""Log collector package for VPS deployment (Plan §9 / Task T7.2)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.collector.app import create_collector_app
    from easypasal.collector.auth import verify_collector_token
    from easypasal.collector.config import CollectorConfig
    from easypasal.collector.db import CollectorDB
    from easypasal.collector.routes import router

__all__ = [
    "CollectorConfig",
    "CollectorDB",
    "create_collector_app",
    "router",
    "verify_collector_token",
]

_APP_SYMBOLS = {"create_collector_app"}
_AUTH_SYMBOLS = {"verify_collector_token"}
_CONFIG_SYMBOLS = {"CollectorConfig"}
_DB_SYMBOLS = {"CollectorDB"}
_ROUTE_SYMBOLS = {"router"}


def __getattr__(name: str) -> object:
    if name in _APP_SYMBOLS:
        import easypasal.collector.app as _app

        return getattr(_app, name)
    if name in _AUTH_SYMBOLS:
        import easypasal.collector.auth as _auth

        return getattr(_auth, name)
    if name in _CONFIG_SYMBOLS:
        import easypasal.collector.config as _config

        return getattr(_config, name)
    if name in _DB_SYMBOLS:
        import easypasal.collector.db as _db

        return getattr(_db, name)
    if name in _ROUTE_SYMBOLS:
        import easypasal.collector.routes as _routes

        return getattr(_routes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
