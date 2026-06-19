"""PQC TrustFlow frontend package."""


def build_app():
    from .app import build_app as _build_app

    return _build_app()


def launch_app() -> None:
    from .app import launch_app as _launch_app

    _launch_app()
