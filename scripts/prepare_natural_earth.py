"""Download pinned Natural Earth boundaries and build GeoRAG's local gazetteer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

import shapefile


VERSION = "5.1.1"
DOWNLOADS = {
    "admin0": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
    "admin1": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
}
ARCHIVE_SHA256 = {
    "admin0": "ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6",
    "admin1": "efc59726337323058f9446210adc96673179cd344e053666ee3d28cb58ba2b05",
}


def _text_values(properties: dict, *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = properties.get(key)
        if raw is None:
            continue
        for value in re.split(r"[|;]", str(raw)):
            value = value.strip()
            if value and value not in values:
                values.append(value)
    return values


def _download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    with urlopen(url, timeout=90) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    with destination.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(shapefile_path: Path, level: str) -> list[dict]:
    reader = shapefile.Reader(str(shapefile_path), encoding="utf-8")
    result = []
    for row_number, shape_record in enumerate(reader.iterShapeRecords()):
        props = shape_record.record.as_dict()
        geometry = shape_record.shape.__geo_interface__
        if level == "country":
            name = str(props.get("ADMIN") or props.get("NAME") or "").strip()
            iso = str(props.get("ADM0_A3") or props.get("ISO_A3") or f"row-{row_number}").strip()
            if not name:
                continue
            aliases = _text_values(props, "NAME", "NAME_LONG", "FORMAL_EN", "SOVEREIGNT", "BRK_NAME")
            place_id = f"country:{iso}"
            country = name
        else:
            name = str(props.get("name") or props.get("name_en") or "").strip()
            if not name:
                continue
            country = str(props.get("admin") or props.get("geonunit") or "").strip() or None
            iso = str(props.get("iso_3166_2") or props.get("hasc_1") or "").strip()
            country_iso = str(props.get("adm0_a3") or props.get("sov_a3") or "").strip()
            fallback = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
            place_id = f"admin1:{iso}" if iso else f"admin1:{country_iso}:{fallback}"
            aliases = _text_values(props, "name_en", "name_alt", "name_local", "postal", "abbrev")
        aliases = [value for value in aliases if value.casefold() != name.casefold()]
        # A few Natural Earth features share codes (typically alternate or island
        # polygons). Keep them individually addressable within this pinned version.
        if any(existing["place_id"] == place_id for existing in result):
            place_id = f"{place_id}:feature-{row_number}"
        result.append({
            "place_id": place_id,
            "level": level,
            "name": name,
            "country": country,
            "aliases": aliases,
            "geometry": geometry,
        })
    reader.close()
    return result


def prepare(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    places: list[dict] = []
    hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="georag-natural-earth-") as temporary:
        temp_dir = Path(temporary)
        for level, url in DOWNLOADS.items():
            archive = temp_dir / f"{level}.zip"
            hashes[level] = _download(url, archive)
            if hashes[level] != ARCHIVE_SHA256[level]:
                raise RuntimeError(
                    f"Natural Earth {level} archive checksum changed: "
                    f"expected {ARCHIVE_SHA256[level]}, received {hashes[level]}"
                )
            extract_dir = temp_dir / level
            extract_dir.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    target = (extract_dir / member.filename).resolve()
                    if not target.is_relative_to(extract_dir.resolve()):
                        raise RuntimeError(f"unsafe path in Natural Earth archive: {member.filename}")
                    zipped.extract(member, extract_dir)
            shp = next(extract_dir.rglob("*.shp"), None)
            if shp is None:
                raise RuntimeError(f"Natural Earth {level} archive did not contain a shapefile")
            places.extend(_records(shp, "country" if level == "admin0" else "admin1"))
    ids = [place["place_id"] for place in places]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Natural Earth produced duplicate gazetteer IDs")
    payload = {
        "source": "Natural Earth 1:10m cultural vectors",
        "version": VERSION,
        "license": "public domain",
        "archive_sha256": hashes,
        "place_count": len(places),
        "places": places,
    }
    destination = output_dir / "gazetteer.json.gz"
    temporary_file = destination.with_suffix(".json.gz.tmp")
    with temporary_file.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        compressed.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    temporary_file.replace(destination)
    print(f"Wrote {len(places):,} places to {destination}")
    print(f"Source: Natural Earth {VERSION}; SHA-256: {hashes}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/geography/natural_earth_5.1.1"))
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
