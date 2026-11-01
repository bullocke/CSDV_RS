"""csdv_core.review.config: settings shared by the review server and app.

``csdv review app`` puts its options into environment variables before it
starts the server, because Solara imports the app module by name and passes it
nothing. :func:`review_config` reads them back.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ENV_EXPORT_DIR = "CSDV_REVIEW_EXPORT_DIR"
ENV_RUN_TAG = "CSDV_REVIEW_RUN_TAG"
ENV_STORE = "CSDV_REVIEW_STORE"
ENV_INTERPRETER = "CSDV_REVIEW_INTERPRETER"

__all__ = ["ReviewConfig", "default_export_dir", "review_config"]


def default_export_dir() -> Path:
    """``$CSDV_EXPORT_DIR``, else ``<results_root>/chpc_exports``."""
    from csdv_core.io.paths import project_paths

    env = os.environ.get("CSDV_EXPORT_DIR")
    return Path(env) if env else project_paths().results_root / "chpc_exports"


@dataclass(frozen=True)
class ReviewConfig:
    """Where the exports and the review GeoPackage are, and who is reviewing."""

    export_dir: Path
    run_tag: str
    store_path: Path
    interpreter: str

    @property
    def quicklook_root(self) -> Path:
        return self.export_dir / "quicklooks"

    def run_dir(self, unit: str) -> Path:
        return self.export_dir / "runs" / self.run_tag / unit

    def to_env(self) -> None:
        os.environ[ENV_EXPORT_DIR] = str(self.export_dir)
        os.environ[ENV_RUN_TAG] = self.run_tag
        os.environ[ENV_STORE] = str(self.store_path)
        os.environ[ENV_INTERPRETER] = self.interpreter


@lru_cache(maxsize=1)
def review_config() -> ReviewConfig:
    """The configuration of this process, from the environment."""
    from csdv_core.review.store import default_store_path

    run_tag = os.environ.get(ENV_RUN_TAG, "default")
    store = os.environ.get(ENV_STORE)
    return ReviewConfig(
        export_dir=Path(os.environ.get(ENV_EXPORT_DIR) or default_export_dir()),
        run_tag=run_tag,
        store_path=Path(store) if store else default_store_path(run_tag),
        interpreter=os.environ.get(ENV_INTERPRETER) or getpass.getuser(),
    )
