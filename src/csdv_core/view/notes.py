"""csdv_core.view.notes: the comment points of the map viewer.

The notes are one GeoJSON file of points in EPSG:4326 with flat properties,
so it opens in QGIS and reads as text. Every change rewrites the file through
a temporary file that is moved into place, and is also appended to a JSON
Lines log beside it (``<name>_log.jsonl``). The log is only ever appended to,
so a note that was typed and then deleted can still be found there. The
file is read again when another program (QGIS, an editor) has changed it.

A note without a comment and without a tag is not kept.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Tags a note can carry, in the order of the number keys of the page.
TAGS: tuple[str, ...] = (
    "false_positive",
    "missed",
    "wrong_type",
    "wrong_date",
    "good",
    "other",
)
#: Properties of a note that the page may set. Others are dropped.
FIELDS: tuple[str, ...] = (
    "unit",
    "place",
    "tag",
    "comment",
    "run_a",
    "run_b",
    "layer",
    "interval",
    "naip_year",
    "background",
    "threshold_a",
    "threshold_b",
    "forest_only",
    "zoom",
    "values",
)

__all__ = ["FIELDS", "TAGS", "NoteStore", "default_notes_path"]


def default_notes_path() -> Path:
    """``<data_root>/reference/map_notes/pair_maps.geojson``."""
    from csdv_core.io.paths import project_paths

    return project_paths().data_root / "reference" / "map_notes" / "pair_maps.geojson"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NoteStore:
    """Notes by ID, kept in a GeoJSON file.

    Args:
        path: The GeoJSON file. It is created with the first note.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.log_path = self.path.with_name(f"{self.path.stem}_log.jsonl")
        self._lock = threading.Lock()
        self._notes: dict[str, dict[str, Any]] = {}
        self._mtime: float | None = None
        self.load()

    def load(self) -> None:
        """Read the file, keeping features in their stored order."""
        self._notes = {}
        self._mtime = None
        if not self.path.exists():
            return
        self._mtime = self.path.stat().st_mtime
        collection = json.loads(self.path.read_text(encoding="utf-8"))
        for feature in collection.get("features", []):
            note_id = str(feature.get("properties", {}).get("id", ""))
            if note_id:
                self._notes[note_id] = feature

    def _reload_if_changed(self) -> None:
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime != self._mtime:
            logger.info("%s changed on disk, reading it again", self.path.name)
            self.load()

    def collection(self) -> dict[str, Any]:
        """Every note as a GeoJSON feature collection."""
        with self._lock:
            self._reload_if_changed()
            return {"type": "FeatureCollection", "features": list(self._notes.values())}

    def __len__(self) -> int:
        return len(self._notes)

    def put(
        self, note_id: str, lon: float, lat: float, properties: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create or replace a note.

        Args:
            note_id: ID chosen by the page, so a repeated save replaces.
            lon: Longitude in degrees.
            lat: Latitude in degrees.
            properties: Values of :data:`FIELDS`. Other keys are dropped.

        Returns:
            The stored feature, or None when the note is empty (an empty note
            is removed if it existed).
        """
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"not a lon/lat position: {lon}, {lat}")
        props = {key: properties.get(key) for key in FIELDS}
        props["comment"] = str(props["comment"] or "").strip()
        props["tag"] = str(props["tag"] or "")
        if props["tag"] and props["tag"] not in TAGS:
            raise ValueError(f"unknown tag {props['tag']!r}")
        if not props["comment"] and not props["tag"]:
            self.delete(note_id)
            return None
        with self._lock:
            self._reload_if_changed()
            now = _now()
            old = self._notes.get(note_id, {}).get("properties", {})
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 7), round(lat, 7)],
                },
                "properties": {
                    "id": note_id,
                    "created": old.get("created", now),
                    "updated": now,
                    "lon": round(lon, 7),
                    "lat": round(lat, 7),
                    **props,
                },
            }
            self._notes[note_id] = feature
            self._log({"action": "put", "time": now, "note": feature["properties"]})
            self._write()
            return feature

    def delete(self, note_id: str) -> bool:
        """Remove a note. Returns whether it existed."""
        with self._lock:
            self._reload_if_changed()
            feature = self._notes.pop(note_id, None)
            if feature is None:
                return False
            self._log(
                {"action": "delete", "time": _now(), "note": feature["properties"]}
            )
            self._write()
            return True

    def _log(self, record: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        collection = {
            "type": "FeatureCollection",
            "name": self.path.stem,
            "features": list(self._notes.values()),
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(collection, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime
