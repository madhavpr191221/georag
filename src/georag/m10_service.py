"""Local M10 query parsing, metadata constraints, exact retrieval, and previews."""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, time, timezone
import gzip
import json
import os
from pathlib import Path
import unicodedata
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError
import shapefile
from shapely.geometry import box, shape
from shapely import make_valid
from shapely.ops import unary_union
import torch
from torch.nn import functional as F

from georag.data.sen12flood import SEN12FloodDataset, sequence_id_for_record
from georag.data.sen12flood_cache import SEN12FloodTensorCache
from georag.data.splits import load_group_split_manifest
from georag.evaluation.prototype_retrieval import class_prototype, split_prototype_support
from georag.evaluation.spectral_evidence import (
    DEFAULT_MAX_PAIR_DAYS,
    INDEX_NAMES,
    change_indices,
    compute_indices,
    cosine_scores,
    index_histogram,
    make_temporal_pairs,
    summarize_indices,
)
from georag.experiments.sen12flood_contrastive import RGB_CHANNELS, _build_model
from georag.query import EOQuery, PlaceChoice
from georag.retrieval.flat import FlatIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts/milestone_9/runs"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/sen12flood/sen12flood"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "artifacts/milestone_9/cache"
DEFAULT_GAZETTEER = PROJECT_ROOT / "data/geography/natural_earth_5.1.1/gazetteer.json.gz"
DEFAULT_M8_CONFIG = PROJECT_ROOT / "configs/milestone_8.toml"
DEFAULT_M9_CONFIG = PROJECT_ROOT / "configs/milestone_9.toml"
DEFAULT_M10_CONFIG = PROJECT_ROOT / "configs/milestone_10.toml"
DEFAULT_M11_CONFIG = PROJECT_ROOT / "configs/milestone_11_spectral_evidence.toml"


class QueryParseError(RuntimeError):
    """Raised when natural-language text cannot be parsed into the contract."""


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char)).strip()


def _as_iso(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


def parse_natural_language(query: str, model: str | None = None) -> EOQuery:
    """Ask OpenAI only to map query text to the narrow validated request schema."""
    if not os.getenv("OPENAI_API_KEY"):
        raise QueryParseError("OPENAI_API_KEY is not configured in the backend environment.")
    instructions = (
        "Convert the user's Earth-observation retrieval request into the supplied schema. "
        "Supported intents: flood and no_flood for scene-label prototype retrieval; visual_similarity "
        "for an RGB learned-embedding example search; spectral_water_signal for scenes with a "
        "stronger water-like multispectral index signal; spectral_water_increase or "
        "spectral_water_decrease for same-location before/after pairs with that direction of "
        "MNDWI change; spectral_scene_similarity for a multispectral index-pattern match to a "
        "tile selected in the interface; spectral_change_similarity for a before/after index-change "
        "pattern match to a pair selected in the interface. Use intent=unsupported for other tasks. "
        "Supported filters are an inclusive date range, one "
        "country or first-order administrative region (state/province), and RGB or Sentinel-2 "
        "12-band modality preference. Set requested_modality=unspecified when not stated. "
        "Do not invent a place, date, or condition. Put requests that cannot be represented "
        "in unsupported_conditions and explain them briefly in interpretation. For an admin1 "
        "place, place only the region name in place_name and use place_country for the "
        "surrounding country when mentioned; do not combine them in place_name. Place level "
        "must be country or admin1. Districts, distance/radius, routes, spatial exclusions, "
        "SAR, claims about flooded pixels, unprovided rain or land-cover conditions, and open-ended "
        "image interpretation are unsupported. "
        "A place mention must be represented only when it is clearly a country or state/province. "
        "Do not answer the question or claim anything about retrieved imagery."
    )
    client = OpenAI()
    response = client.responses.parse(
        model=model or os.getenv("GEORAG_QUERY_MODEL", "gpt-5.6"),
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": query},
        ],
        text_format=EOQuery,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise QueryParseError("The parser returned no structured query. Please rephrase it.")
    try:
        return parsed.model_copy(update={"original_query": query, "resolved_place_id": None,
                                         "example_tile_id": None})
    except ValidationError as error:
        raise QueryParseError(f"The parsed query did not pass validation: {error}") from error


class M10Service:
    """Own local SEN12-FLOOD data, selected-run gallery vectors, and gazetteer."""

    def __init__(
        self,
        *,
        data_root: Path = DEFAULT_DATA_ROOT,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        run_root: Path = DEFAULT_RUN_ROOT,
        gazetteer_path: Path = DEFAULT_GAZETTEER,
        m8_config_path: Path = DEFAULT_M8_CONFIG,
        m9_config_path: Path = DEFAULT_M9_CONFIG,
        m10_config_path: Path = DEFAULT_M10_CONFIG,
        m11_config_path: Path = DEFAULT_M11_CONFIG,
    ) -> None:
        import tomllib

        self.run_root = run_root
        self.gazetteer_path = gazetteer_path
        m8 = tomllib.loads(m8_config_path.read_text(encoding="utf-8"))
        d, res = m8["dataset"], m8["resampling"]
        self.source = SEN12FloodDataset(
            data_root,
            d["bands"],
            d["target_resolution_m"],
            d["value_scale"],
            d["target_grid_band"],
            res["finer_than_target"],
            res["coarser_than_target"],
            res["same_resolution"],
            res["nodata_fill"],
        )
        self.dataset = SEN12FloodTensorCache(self.source, cache_root)
        self.record_by_id = {record.tile_id: record for record in self.source.records}
        self.index_by_id = {record.tile_id: index for index, record in enumerate(self.source.records)}
        self.run_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._gazetteer: list[dict[str, Any]] | None = None
        self._place_geometries: dict[str, Any] = {}
        self._preview_cache: OrderedDict[str, bytes] = OrderedDict()
        self.m9_config = tomllib.loads(m9_config_path.read_text(encoding="utf-8"))
        self.m10_config = tomllib.loads(m10_config_path.read_text(encoding="utf-8"))
        self.m11_config = tomllib.loads(m11_config_path.read_text(encoding="utf-8"))
        spectral_settings = self.m11_config["spectral"]
        if tuple(spectral_settings["index_names"]) != INDEX_NAMES:
            raise ValueError(f"M11 index names must be {INDEX_NAMES}")
        self.spectral_bins = int(spectral_settings["histogram_bins"])
        self.spectral_range = (float(spectral_settings["histogram_min"]), float(spectral_settings["histogram_max"]))
        self.change_range = (float(spectral_settings["change_histogram_min"]), float(spectral_settings["change_histogram_max"]))
        self.index_epsilon = float(spectral_settings["denominator_epsilon"])
        self.spectral_max_pair_days = int(self.m11_config["temporal"]["max_pair_days"])
        retrieval_settings = self.m11_config["retrieval"]
        if retrieval_settings["gallery_split"] != "train" or not retrieval_settings["exclude_example_sequence"]:
            raise ValueError("M11 requires the train gallery and example-sequence exclusion")
        if self.spectral_bins < 2 or self.index_epsilon < 0 or self.spectral_max_pair_days < 1:
            raise ValueError("invalid M11 spectral or temporal configuration")
        selection_artifact = Path(self.m10_config["selection"]["artifact"])
        self.selection_path = selection_artifact if selection_artifact.is_absolute() else PROJECT_ROOT / selection_artifact
        assignments = load_group_split_manifest(
            Path(self.m9_config["split"]["manifest"]), self.source.all_records, sequence_id_for_record,
        )
        train_records = [record for record in self.source.records if assignments[record.tile_id] == "train"]
        self.prototype_support_ids, self.searchable_gallery_ids = split_prototype_support(
            train_records,
            float(self.m10_config["prototype"]["support_fraction"]),
            int(self.m10_config["prototype"]["split_seed"]),
        )
        self.spectral_records = tuple(
            record for record in self.source.records
            if assignments[record.tile_id] == "train" and record.tile_id not in self.prototype_support_ids
        )
        self._spectral_index_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._spectral_scene_features: dict[str, tuple[np.ndarray, dict[str, dict[str, float]]]] = {}
        self._spectral_pair_features: dict[tuple[str, str], tuple[np.ndarray, dict[str, dict[str, float]], dict[str, dict[str, float]]]] = {}

    def available_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for run_dir in sorted(self.run_root.glob("*/*")):
            result_file = run_dir / "result.json"
            retrieval_file = run_dir / "test_retrieval.pt"
            checkpoint_file = run_dir / "checkpoint_best.pt"
            if not all(path.is_file() for path in (result_file, retrieval_file, checkpoint_file)):
                continue
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result.get("smoke"):
                continue
            runs.append({
                "run_id": run_dir.relative_to(self.run_root).as_posix(),
                "architecture": result["architecture"],
                "modality": result["modality"],
                "seed": result["seed"],
                "gallery_count": result["gallery_count"],
                "test_query_count": result["test_query_count"],
                "directory": run_dir,
            })
        return runs

    def public_runs(self) -> list[dict[str, Any]]:
        if not self.selection_path.is_file():
            raise FileNotFoundError(
                f"Validation model selection is missing at {self.selection_path}. "
                "Run `uv run python scripts/evaluate_m10_prototypes.py` first."
            )
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        saved_support_ids = set(selection.get("protocol", {}).get("support_tile_ids", []))
        if saved_support_ids != self.prototype_support_ids:
            raise ValueError("validation model selection does not match the current support split; rerun evaluate_m10_prototypes.py")
        by_id = {run["run_id"]: run for run in self.available_runs()}
        public: list[dict[str, Any]] = []
        for architecture in ("cnn", "vit"):
            preset = selection.get("presets", {}).get(architecture)
            if not preset:
                raise ValueError(f"validation selection has no {architecture} preset")
            run = by_id.get(preset["representative_run_id"])
            if run is None:
                raise ValueError(f"selected M9 run is unavailable: {preset['representative_run_id']}")
            public.append({
                **{key: value for key, value in run.items() if key != "directory"},
                "display_name": f"{'CNN' if architecture == 'cnn' else 'ViT'} - {'RGB' if run['modality'] == 'rgb' else '12-band'}",
                "validation_macro_precision_at_5_mean": preset["mean_macro_precision_at_k"],
                "validation_macro_precision_at_5_std": preset["std_macro_precision_at_k"],
                "validation_seed_count": preset["seed_count"],
                "representative_seed_score": preset["representative_macro_precision_at_k"],
                "searchable_gallery_count": len(self.searchable_gallery_ids),
                "prototype_support_count": len(self.prototype_support_ids),
            })
        return public

    def dataset_summary(self) -> dict[str, Any]:
        dates = [record.timestamp for record in self.source.records if record.timestamp is not None]
        return {
            "available_scenes": len(self.source.records),
            "catalog_scenes": len(self.source.all_records),
            "date_start": min(dates).date().isoformat() if dates else None,
            "date_end": max(dates).date().isoformat() if dates else None,
            "sensor": "Sentinel-2 MSI",
            "bands": list(self.source.bands),
        }

    def catalog(self, search: str = "", limit: int = 100) -> list[dict[str, Any]]:
        needle = _normalize_name(search)
        records = [record for record in self.source.records if not needle or needle in _normalize_name(record.tile_id)]
        return [{"tile_id": record.tile_id, "date": _as_iso(record.timestamp),
                 "sequence_id": (record.metadata or {}).get("sequence_id")}
                for record in records[:max(1, min(limit, 200))]]

    def find_places(
        self, query: str, level: str | None = None, limit: int = 10,
        country: str | None = None,
    ) -> list[PlaceChoice]:
        if not self.gazetteer_path.is_file():
            raise FileNotFoundError(
                f"Offline gazetteer not found at {self.gazetteer_path}. "
                "Run `uv run python scripts/prepare_natural_earth.py` first."
            )
        self._load_gazetteer()
        needle = _normalize_name(query)
        country_names: set[str] = set()
        if country:
            country_needle = _normalize_name(country)
            country_matches = [
                item for item in self._gazetteer or []
                if item["level"] == "country"
                and country_needle in {
                    _normalize_name(str(value))
                    for value in [item["name"], *item.get("aliases", [])]
                }
            ]
            country_names = {country_needle}
            for item in country_matches:
                country_names.add(_normalize_name(item["name"]))
                country_names.update(_normalize_name(str(alias)) for alias in item.get("aliases", []))
        if not needle:
            return []
        ranked: list[tuple[int, PlaceChoice]] = []
        for item in self._gazetteer or []:
            if level and item["level"] != level:
                continue
            if country_names and _normalize_name(str(item.get("country") or "")) not in country_names:
                continue
            names = [_normalize_name(str(value)) for value in [item["name"], *item.get("aliases", [])] if value]
            if any(name == needle for name in names):
                score = 0
            elif any(needle in name for name in names):
                score = 1
            else:
                continue
            ranked.append((score, PlaceChoice(place_id=item["place_id"], name=item["name"],
                                               level=item["level"], country=item.get("country"))))
        ranked.sort(key=lambda item: (item[0], item[1].name, item[1].country or ""))
        return [choice for _, choice in ranked[:max(1, min(limit, 20))]]

    def retrieve(self, query: EOQuery, run_id: str, k: int) -> dict[str, Any]:
        if query.intent == "unsupported" or query.unsupported_conditions:
            unsupported = ", ".join(query.unsupported_conditions) or "this request"
            raise ValueError(f"This query includes unsupported condition(s): {unsupported}.")
        if query.intent == "visual_similarity":
            raise ValueError("Select an example tile to run a visual-similarity query.")
        run = self._load_run(run_id)
        if query.requested_modality != "unspecified" and query.requested_modality != run["modality"]:
            raise ValueError(
                f"The query asks for {query.requested_modality}, but selected run {run_id} uses "
                f"{run['modality']}. Select a matching run or remove that modality constraint."
            )

        query_vector = self._query_vector(query, run)
        return self._retrieve_with_vector(query, run_id, k, query_vector, run)

    def preview(self, tile_id: str) -> bytes:
        if tile_id in self._preview_cache:
            self._preview_cache.move_to_end(tile_id)
            return self._preview_cache[tile_id]
        if tile_id not in self.index_by_id:
            raise KeyError(f"unknown or unavailable tile ID: {tile_id}")
        sample = self.dataset[self.index_by_id[tile_id]]
        image = sample.image
        rgb = image[list(RGB_CHANNELS)].numpy()
        mask = sample.valid_mask.numpy()
        output = np.zeros((*rgb.shape[1:], 3), dtype=np.uint8)
        for channel in range(3):
            values = rgb[channel][mask]
            if values.size:
                low, high = np.percentile(values, (2, 98))
                if high <= low:
                    high = low + 1.0
                scaled = np.clip((rgb[channel] - low) / (high - low), 0, 1)
                output[..., channel] = np.rint(scaled * 255).astype(np.uint8)
        from io import BytesIO
        stream = BytesIO()
        Image.fromarray(output, mode="RGB").save(stream, format="PNG", optimize=True)
        data = stream.getvalue()
        self._preview_cache[tile_id] = data
        if len(self._preview_cache) > 48:
            self._preview_cache.popitem(last=False)
        return data

    def spectral_map(self, tile_id: str, index_name: str, after_tile_id: str | None = None) -> bytes:
        """Render a spectral-index map, optionally as after-minus-before change."""
        if index_name not in INDEX_NAMES:
            raise KeyError(f"unknown spectral index: {index_name}")
        indices = self._spectral_indices(tile_id)
        values = indices[index_name]
        value_range = (-1.0, 1.0)
        title = f"{index_name.upper()} · {tile_id}"
        if after_tile_id:
            after = self._spectral_indices(after_tile_id)
            changes = change_indices(indices, after)
            values = changes[index_name]
            value_range = (-2.0, 2.0)
            title = f"Δ {index_name.upper()} · after minus before"
        from io import BytesIO
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
        image = axis.imshow(values, cmap="RdYlBu", vmin=value_range[0], vmax=value_range[1])
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        stream = BytesIO()
        figure.savefig(stream, format="png", dpi=110)
        plt.close(figure)
        return stream.getvalue()

    def retrieve_spectral(
        self, query: EOQuery, k: int, before_example_tile_id: str | None = None,
        after_example_tile_id: str | None = None,
    ) -> dict[str, Any]:
        """Rank train-gallery scenes/pairs using explicit Sentinel-2 index evidence."""
        intents = {
            "spectral_water_signal", "spectral_water_increase", "spectral_water_decrease",
            "spectral_scene_similarity", "spectral_change_similarity",
        }
        if query.intent not in intents or query.unsupported_conditions:
            unsupported = ", ".join(query.unsupported_conditions) or query.intent
            raise ValueError(f"This is not a supported spectral request: {unsupported}.")
        if query.requested_modality == "rgb":
            raise ValueError("Spectral evidence needs the Sentinel-2 optical bands; remove the RGB-only constraint.")
        if query.place_name and not query.resolved_place_id:
            raise ValueError("Choose a matching offline gazetteer result before retrieving.")
        geometry = self._resolved_geometry(query) if query.place_name else None
        start = datetime.combine(query.start_date, time.min, tzinfo=timezone.utc) if query.start_date else None
        end = datetime.combine(query.end_date, time.max, tzinfo=timezone.utc) if query.end_date else None
        if query.intent in {"spectral_water_signal", "spectral_scene_similarity"}:
            scene_example_id = before_example_tile_id if query.intent == "spectral_scene_similarity" else None
            if query.intent == "spectral_scene_similarity" and not scene_example_id:
                raise ValueError("Select an example tile for spectral scene similarity.")
            reference_descriptor = None
            if scene_example_id:
                reference_descriptor, reference_summary = self._scene_features(scene_example_id)
                if reference_summary["mndwi"]["valid_pixels"] == 0:
                    raise ValueError("The selected example has no valid MNDWI pixels for spectral comparison.")
            rows = []
            for record in self.spectral_records:
                if scene_example_id and (record.tile_id == scene_example_id or
                        sequence_id_for_record(record) == sequence_id_for_record(self.source.record_for_id(scene_example_id))):
                    continue
                if not _record_matches(record, start, end, geometry):
                    continue
                descriptor, summary = self._scene_features(record.tile_id)
                if summary["mndwi"]["valid_pixels"] == 0 or float(np.linalg.norm(descriptor)) <= 1e-12:
                    continue
                if reference_descriptor is not None:
                    score = float(cosine_scores(reference_descriptor, descriptor[None, :])[0])
                    description = "cosine similarity of NDWI, MNDWI, and NDVI histograms"
                else:
                    # A positive MNDWI is only a water-like spectral clue, not a flood decision.
                    score = summary["mndwi"]["median"]
                    description = "median MNDWI; higher means more water-like index response, not flood probability"
                rows.append((score, record, summary))
            rows.sort(key=lambda item: (-item[0], item[1].tile_id))
            results = [self._spectral_scene_result(rank, score, record, summary, description)
                       for rank, (score, record, summary) in enumerate(rows[:k], 1)]
            method = "exact_index_histogram_cosine" if reference_descriptor is not None else "median_mndwi_rank"
            measure = "cosine similarity" if reference_descriptor is not None else "median MNDWI index value"
            candidate_count = len(rows)
        else:
            pairs = make_temporal_pairs(list(self.spectral_records), self.spectral_max_pair_days)
            all_pairs = make_temporal_pairs(list(self.source.records), self.spectral_max_pair_days)
            reference_descriptor = None
            if query.intent == "spectral_change_similarity":
                if not before_example_tile_id or not after_example_tile_id:
                    raise ValueError("Select both before and after example tiles for change similarity.")
                _find_example_pair(
                    all_pairs, before_example_tile_id, after_example_tile_id, self.spectral_max_pair_days,
                )
                reference_changes = change_indices(
                    self._spectral_indices(before_example_tile_id), self._spectral_indices(after_example_tile_id)
                )
                if summarize_indices(reference_changes)["mndwi"]["valid_pixels"] == 0:
                    raise ValueError("The selected example pair has no shared valid MNDWI pixels.")
                reference_descriptor = index_histogram(reference_changes, self.spectral_bins, self.change_range)
            rows = []
            excluded_sequence = None
            if query.intent == "spectral_change_similarity" and before_example_tile_id:
                excluded_sequence = sequence_id_for_record(self.source.record_for_id(before_example_tile_id))
            for pair in pairs:
                before, after = pair["before"], pair["after"]
                if excluded_sequence and pair["sequence_id"] == excluded_sequence:
                    continue
                if not _record_matches(after, start, end, geometry):
                    continue
                descriptor, after_summary, summary = self._pair_features(before.tile_id, after.tile_id)
                if summary["mndwi"]["valid_pixels"] == 0 or float(np.linalg.norm(descriptor)) <= 1e-12:
                    continue
                if reference_descriptor is not None:
                    score = float(cosine_scores(reference_descriptor, descriptor[None, :])[0])
                    description = "cosine similarity of before/after index-change histograms"
                else:
                    score = summary["mndwi"]["median"]
                    description = "median after-minus-before MNDWI; direction is a spectral clue, not flood extent"
                rows.append((score, pair, after_summary, summary))
            if query.intent == "spectral_water_decrease":
                rows.sort(key=lambda item: (item[0], item[1]["after"].tile_id))
            else:
                rows.sort(key=lambda item: (-item[0], item[1]["after"].tile_id))
            results = [self._spectral_pair_result(rank, score, pair, after_summary, change_summary, description)
                       for rank, (score, pair, after_summary, change_summary) in enumerate(rows[:k], 1)]
            method = "exact_change_histogram_cosine" if reference_descriptor is not None else "median_delta_mndwi_rank"
            measure = "cosine similarity" if reference_descriptor is not None else "signed median MNDWI change"
            candidate_count = len(rows)
        notice = None if candidate_count >= k else f"Only {candidate_count} matching evidence item(s) were found."
        return {"query": query, "retrieval_method": method, "similarity_measure": measure,
                "candidate_count": candidate_count, "results": results, "notice": notice}

    def _spectral_indices(self, tile_id: str) -> dict[str, np.ndarray]:
        if tile_id in self._spectral_index_cache:
            self._spectral_index_cache.move_to_end(tile_id)
            return self._spectral_index_cache[tile_id]
        if tile_id not in self.index_by_id:
            raise KeyError(f"unknown or unavailable tile ID: {tile_id}")
        sample = self.dataset[self.index_by_id[tile_id]]
        indices = compute_indices(sample.image.numpy(), sample.valid_mask.numpy(), tuple(self.source.bands), self.index_epsilon)
        self._spectral_index_cache[tile_id] = indices
        if len(self._spectral_index_cache) > 8:
            self._spectral_index_cache.popitem(last=False)
        return indices

    def _scene_features(self, tile_id: str) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        if tile_id not in self._spectral_scene_features:
            indices = self._spectral_indices(tile_id)
            self._spectral_scene_features[tile_id] = (
                index_histogram(indices, self.spectral_bins, self.spectral_range), summarize_indices(indices),
            )
        return self._spectral_scene_features[tile_id]

    def _pair_features(
        self, before_id: str, after_id: str,
    ) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        key = (before_id, after_id)
        if key not in self._spectral_pair_features:
            before = self._spectral_indices(before_id)
            after = self._spectral_indices(after_id)
            changes = change_indices(before, after)
            self._spectral_pair_features[key] = (
                index_histogram(changes, self.spectral_bins, self.change_range),
                summarize_indices(after), summarize_indices(changes),
            )
        return self._spectral_pair_features[key]

    @staticmethod
    def _spectral_scene_result(rank, score, record, summary, description):
        return {"rank": rank, "score": score, "score_description": description,
                "tile_id": record.tile_id, "after_date": _as_iso(record.timestamp),
                "sequence_id": sequence_id_for_record(record), "latitude": record.latitude,
                "longitude": record.longitude, "label": record.labels[0] if record.labels else None,
                "preview_url": f"/api/tiles/{record.tile_id}/preview",
                "index_url": f"/api/tiles/{record.tile_id}/indices/mndwi", "index_summary": summary}

    @staticmethod
    def _spectral_pair_result(rank, score, pair, after_summary, change_summary, description):
        before, after = pair["before"], pair["after"]
        return {"rank": rank, "score": score, "score_description": description,
                "before_tile_id": before.tile_id, "after_tile_id": after.tile_id,
                "before_date": _as_iso(before.timestamp), "after_date": _as_iso(after.timestamp),
                "gap_days": pair["gap_days"], "sequence_id": pair["sequence_id"],
                "latitude": after.latitude, "longitude": after.longitude,
                "label": after.labels[0] if after.labels else None,
                "before_preview_url": f"/api/tiles/{before.tile_id}/preview",
                "after_preview_url": f"/api/tiles/{after.tile_id}/preview",
                "before_index_url": f"/api/tiles/{before.tile_id}/indices/mndwi",
                "after_index_url": f"/api/tiles/{after.tile_id}/indices/mndwi",
                "change_map_url": f"/api/tiles/{before.tile_id}/change/{after.tile_id}/mndwi",
                "index_summary": after_summary, "change_summary": change_summary}

    def _query_vector(self, query: EOQuery, run: dict[str, Any]) -> np.ndarray:
        if query.intent == "visual_similarity":
            raise ValueError("Select an example tile to run a visual-similarity query.")
        if query.place_name and not query.resolved_place_id:
            raise ValueError("Choose a matching offline gazetteer result before retrieving.")
        if query.place_name:
            self._resolved_geometry(query)  # Validate that this exact choice still exists.
        return class_prototype(run["prototype_embeddings"], run["prototype_labels"], query.intent)

    def retrieve_with_example(self, query: EOQuery, run_id: str, k: int, example_tile_id: str) -> dict[str, Any]:
        run = self._load_run(run_id)
        if query.intent == "unsupported" or query.unsupported_conditions:
            raise ValueError("This query contains unsupported conditions.")
        if query.requested_modality not in {"unspecified", run["modality"]}:
            raise ValueError(f"Selected run {run_id} does not match requested modality {query.requested_modality}.")
        if query.place_name and not query.resolved_place_id:
            raise ValueError("Choose a matching offline gazetteer result before retrieving.")
        if example_tile_id not in self.index_by_id:
            raise KeyError(f"unknown or unavailable example tile ID: {example_tile_id}")
        query_vector = self._encode_tile(example_tile_id, run)
        return self._retrieve_with_vector(query, run_id, k, query_vector, run, example_tile_id)

    def _retrieve_with_vector(self, query: EOQuery, run_id: str, k: int,
                              query_vector: np.ndarray, run: dict[str, Any],
                              excluded_tile_id: str | None = None) -> dict[str, Any]:
        if query.intent == "unsupported" or query.unsupported_conditions:
            raise ValueError("This query contains unsupported conditions.")
        if query.place_name and not query.resolved_place_id:
            raise ValueError("Choose a matching offline gazetteer result before retrieving.")
        if query.requested_modality not in {"unspecified", run["modality"]}:
            raise ValueError(f"Selected run {run_id} does not match requested modality {query.requested_modality}.")
        start = datetime.combine(query.start_date, time.min, tzinfo=timezone.utc) if query.start_date else None
        end = datetime.combine(query.end_date, time.max, tzinfo=timezone.utc) if query.end_date else None
        geometry = self._resolved_geometry(query) if query.place_name else None
        records, vectors = [], []
        eligible = _eligible_indices(run["gallery_records"], start, end, geometry, excluded_tile_id)
        for i in eligible:
            records.append(run["gallery_records"][i])
            vectors.append(run["gallery_embeddings"][i])
        example_mode = excluded_tile_id is not None
        retrieval_method = "example_tile_exact_flat_cosine" if example_mode else "exact_flat_cosine"
        query_method = (
            f"selected M9 {run['architecture']} encoder applied to example tile {excluded_tile_id}"
            if example_mode else "normalized prototype from separate labeled training-support sequences"
        )
        if not vectors:
            return {"query": query, "run_id": run_id, "retrieval_method": retrieval_method,
                    "query_vector_method": query_method, "similarity_measure": "cosine similarity",
                    "candidate_count": 0, "results": [], "notice": "No gallery scenes match these constraints."}
        flat = FlatIndex(np.stack(vectors), records)
        hits = flat.search(query_vector, min(k, flat.size), "cosine")
        results = [{"rank": h.rank, "tile_id": h.tile_id, "score": h.score,
                    "preview_url": f"/api/tiles/{h.tile_id}/preview", "timestamp": h.record["timestamp"],
                    "latitude": h.record["latitude"], "longitude": h.record["longitude"],
                    "sequence_id": h.record["sequence_id"], "sensor": h.record["sensor"],
                    "bands": h.record["bands"], "labels": h.record["labels"], "bbox": h.record["bbox"]}
                   for h in hits]
        return {"query": query, "run_id": run_id, "retrieval_method": retrieval_method,
                "query_vector_method": query_method, "similarity_measure": "cosine similarity",
                "candidate_count": flat.size, "results": results,
                "notice": f"Only {flat.size} scene(s) matched." if flat.size < k else None}

    def _encode_tile(self, tile_id: str, run: dict[str, Any]) -> np.ndarray:
        checkpoint = torch.load(run["directory"] / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        stats = json.loads((run["directory"] / "band_statistics.json").read_text(encoding="utf-8"))
        config = checkpoint["config"]
        channels = RGB_CHANNELS if run["modality"] == "rgb" else tuple(range(12))
        mean = stats["mean"]
        std = stats["std"]
        model = _build_model(run["architecture"], mean, std, config)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        sample = self.dataset[self.index_by_id[tile_id]]
        tensor = sample.image[list(channels)].unsqueeze(0)
        mask = sample.valid_mask.unsqueeze(0)
        with torch.inference_mode():
            vector = model(tensor, mask)[0].cpu().numpy()
        return np.asarray(vector, dtype=np.float32)

    def _load_run(self, run_id: str) -> dict[str, Any]:
        if run_id in self.run_cache:
            self.run_cache.move_to_end(run_id)
            return self.run_cache[run_id]
        matches = [run for run in self.available_runs() if run["run_id"] == run_id]
        if not matches:
            raise KeyError(f"unknown or incomplete M9 run: {run_id}")
        run = matches[0]
        state = torch.load(run["directory"] / "test_retrieval.pt", map_location="cpu", weights_only=False)
        embeddings = F.normalize(torch.as_tensor(state["gallery_embeddings"], dtype=torch.float32), p=2, dim=1).numpy()
        gallery_ids = [str(value) for value in state["gallery_tile_ids"]]
        if len(gallery_ids) != len(embeddings) or len(set(gallery_ids)) != len(gallery_ids):
            raise ValueError(f"invalid ID/vector alignment in M9 run {run_id}")
        expected_ids = self.prototype_support_ids | self.searchable_gallery_ids
        if set(gallery_ids) != expected_ids:
            raise ValueError(f"M9 gallery IDs do not match the configured support/search split for {run_id}")
        records = []
        labels = []
        for tile_id in gallery_ids:
            tile = self.record_by_id.get(tile_id)
            if tile is None:
                raise ValueError(f"gallery ID {tile_id} is absent from local SEN12-FLOOD metadata")
            labels.append(tile.labels[0] if tile.labels else "")
            records.append(_record_dict(tile))
        support_rows = np.asarray([tile_id in self.prototype_support_ids for tile_id in gallery_ids], dtype=bool)
        support_embeddings = embeddings[support_rows]
        support_ids = [tile_id for tile_id, selected in zip(gallery_ids, support_rows, strict=True) if selected]
        support_labels = [label for label, selected in zip(labels, support_rows, strict=True) if selected]
        gallery_rows = ~support_rows
        run.update({
            "gallery_embeddings": embeddings[gallery_rows],
            "gallery_tile_ids": [tile_id for tile_id, selected in zip(gallery_ids, gallery_rows, strict=True) if selected],
            "gallery_records": [record for record, selected in zip(records, gallery_rows, strict=True) if selected],
            "gallery_labels": [label for label, selected in zip(labels, gallery_rows, strict=True) if selected],
            "prototype_embeddings": support_embeddings,
            "prototype_tile_ids": support_ids,
            "prototype_labels": support_labels,
        })
        self.run_cache[run_id] = run
        if len(self.run_cache) > 2:
            self.run_cache.popitem(last=False)
        return run

    def _load_gazetteer(self) -> None:
        if self._gazetteer is None:
            with gzip.open(self.gazetteer_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            self._gazetteer = payload["places"]

    def _resolved_geometry(self, query: EOQuery):
        if not query.resolved_place_id:
            raise ValueError("Choose a matching offline gazetteer result before retrieving.")
        self._load_gazetteer()
        item = next((row for row in self._gazetteer or [] if row["place_id"] == query.resolved_place_id), None)
        if (item is None or item["level"] != query.place_level
                or _normalize_name(item["name"]) != _normalize_name(query.place_name or "")
                or (query.place_country and _normalize_name(item.get("country") or "")
                    != _normalize_name(query.place_country))):
            raise ValueError("The selected geographic match is not present in the current gazetteer.")
        if query.resolved_place_id not in self._place_geometries:
            geometry = shape(item["geometry"])
            self._place_geometries[query.resolved_place_id] = geometry if geometry.is_valid else make_valid(geometry)
        return self._place_geometries[query.resolved_place_id]


def _record_dict(record) -> dict[str, Any]:
    metadata = record.metadata or {}
    bbox = metadata.get("source_bbox")
    timestamp = _as_iso(record.timestamp)
    timestamp_dt = record.timestamp
    if timestamp_dt is not None and timestamp_dt.tzinfo is None:
        timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)
    return {
        "tile_id": record.tile_id, "timestamp": timestamp, "timestamp_dt": timestamp_dt,
        "latitude": record.latitude, "longitude": record.longitude,
        "sequence_id": metadata.get("sequence_id"), "sensor": record.sensor,
        "bands": list(record.bands), "labels": list(record.labels),
        "bbox": list(bbox) if bbox else None,
    }


def _bbox_intersects(bounds: list[float] | tuple[float, ...], geometry) -> bool:
    west, south, east, north = (float(value) for value in bounds)
    if west <= east:
        return geometry.intersects(box(west, south, east, north))
    return geometry.intersects(unary_union([box(west, south, 180, north), box(-180, south, east, north)]))


def intent_prototype(embeddings: np.ndarray, labels: list[str], intent: str) -> np.ndarray:
    """Backward-compatible wrapper for normalized labeled-support prototypes."""
    return class_prototype(embeddings, labels, intent)


def _record_matches(record, start: datetime | None, end: datetime | None, geometry) -> bool:
    timestamp = record.timestamp
    if timestamp is None:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if start and timestamp < start:
        return False
    if end and timestamp > end:
        return False
    metadata = record.metadata or {}
    bbox = metadata.get("source_bbox")
    if geometry is not None and (not bbox or not _bbox_intersects(bbox, geometry)):
        return False
    return True


def _find_example_pair(
    pairs: list[dict[str, Any]], before_id: str, after_id: str,
    max_gap_days: int = DEFAULT_MAX_PAIR_DAYS,
) -> dict[str, Any]:
    for pair in pairs:
        if pair["before"].tile_id == before_id and pair["after"].tile_id == after_id:
            return pair
    raise ValueError(
        "The selected example dates must be consecutive available observations in the same sequence "
        f"and no more than {max_gap_days} days apart."
    )


def _eligible_indices(
    records: list[dict[str, Any]], start: datetime | None, end: datetime | None,
    geometry=None, excluded_tile_id: str | None = None,
) -> list[int]:
    """Return gallery rows satisfying metadata constraints before exact scoring."""
    indices = []
    for i, record in enumerate(records):
        if excluded_tile_id and record["tile_id"] == excluded_tile_id:
            continue
        acquired = record["timestamp_dt"]
        if start and (acquired is None or acquired < start):
            continue
        if end and (acquired is None or acquired > end):
            continue
        bounds = record.get("bbox")
        if geometry is not None and (not bounds or not _bbox_intersects(bounds, geometry)):
            continue
        indices.append(i)
    return indices
