"""OpenAI vision-assisted relevance judgments for the existing M7 retrieval audit.

These are model judgments, not human annotations. Each request sees only one
query and one retrieved RGB tile; dataset labels are used only afterward for
descriptive comparison.
"""

from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence
from urllib.parse import quote

from pydantic import BaseModel


DEFAULT_MODEL = "gpt-5.6"
PROMPT_VERSION = "m7-vlm-relevance-v1"
DECISIONS = ("relevant", "uncertain", "not_relevant")
CONFIDENCE_LEVELS = ("low", "medium", "high")


class VLMJudgment(BaseModel):
    """Small structured output; the evidence field is a concise visual note."""

    judgment: Literal["relevant", "uncertain", "not_relevant"]
    visual_evidence: str
    confidence: Literal["low", "medium", "high"]


def run_vlm_audit(
    audit_csv: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
) -> dict:
    """Judge the audit rows and persist resumable results plus presentation files."""

    if not model.strip():
        raise ValueError("model must be a non-empty API model name")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment before running the VLM audit")

    source = Path(audit_csv).resolve()
    data_root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_rows = _read_audit_rows(source)
    rows = all_rows
    if limit is not None:
        rows = rows[:limit]

    state_path = destination / "judgments.jsonl"
    existing = _read_judgments(state_path)
    for record in existing.values():
        if record.get("requested_model", record.get("model")) != model or record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                "existing VLM results use a different model or prompt version; "
                "choose a new --output-dir to keep runs separate"
            )
    client = _make_client()

    for index, row in enumerate(rows, start=1):
        key = _row_key(row)
        if key in existing:
            continue
        image_path = (data_root / row["image_path"]).resolve()
        if not image_path.is_relative_to(data_root):
            raise ValueError(f"image path escapes the configured dataset root: {row['image_path']}")
        if not image_path.is_file():
            raise FileNotFoundError(f"retrieved image does not exist: {image_path}")

        judgment, response_id, api_model, usage = _judge_one(
            client, image_path, row["query_text"], model
        )
        saved = {
            **row,
            **judgment.model_dump(),
            "judgment_source": "openai_vlm",
            "model": api_model or model,
            "requested_model": model,
            "prompt_version": PROMPT_VERSION,
            "response_id": response_id,
            "usage": usage,
            "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(state_path, saved)
        existing[key] = saved
        print(f"Judged {len(existing)}/{len(rows)}: {row['query_id']} rank {row['rank']}", flush=True)

    result_rows = [existing[_row_key(row)] for row in all_rows if _row_key(row) in existing]
    summary = summarize_judgments(result_rows, expected_count=len(all_rows), model=model)
    _write_outputs(destination, result_rows, summary, data_root)
    return summary


def _make_client():
    from openai import OpenAI

    return OpenAI()


def _read_audit_rows(path: Path) -> list[dict[str, str]]:
    required = {
        "query_id", "query_text", "target_label", "rank", "tile_id", "score",
        "labels", "field_id", "image_path",
    }
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"audit CSV is missing fields: {sorted(required - set(reader.fieldnames or ())) }")
        rows = [
            {field: row.get(field, "") or "" for field in required}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"audit CSV has no retrieval rows: {path}")
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("audit CSV contains duplicate query/rank/tile rows")
    return rows


def _row_key(row: dict) -> str:
    return f"{row['query_id']}\t{row['rank']}\t{row['tile_id']}"


def _read_judgments(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
                if record.get("judgment") not in DECISIONS:
                    raise ValueError("invalid judgment value")
                key = _row_key(record)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                raise ValueError(f"invalid VLM state at {path}:{line_number}: {error}") from error
            if key in records:
                raise ValueError(f"duplicate VLM judgment in state file: {key}")
            records[key] = record
    return records


def _judge_one(client, image_path: Path, query_text: str, model: str):
    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    instructions = (
        "Assess whether this aerial RGB tile visibly supports the user's described "
        "pattern. Decide from visible image evidence only. The crop may be ambiguous; "
        "use uncertain when the pattern cannot be judged reliably. Do not infer from "
        "dataset labels, which are not provided. Do not claim details that are not "
        "visible. Return a concise note describing visual evidence, not hidden reasoning."
    )
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "developer", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Retrieval query: {query_text}"},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{encoded}",
                        "detail": "high",
                    },
                ],
            },
        ],
        text_format=VLMJudgment,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"OpenAI returned no parsed VLM judgment for {image_path.name}")
    if not parsed.visual_evidence.strip() or len(parsed.visual_evidence) > 400:
        raise RuntimeError(f"OpenAI returned invalid visual evidence for {image_path.name}")
    usage = response.usage.model_dump() if response.usage is not None else None
    return parsed, response.id, response.model, usage


def summarize_judgments(
    rows: Sequence[dict], *, expected_count: int, model: str
) -> dict:
    """Summarize judgments; uncertain cases form explicit precision bounds."""

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("judgment") not in DECISIONS:
            raise ValueError(f"invalid VLM decision for {_row_key(row)}")
        grouped.setdefault(row["query_id"], []).append(row)
    queries: list[dict] = []
    for query_id in sorted(grouped):
        query_rows = sorted(grouped[query_id], key=lambda row: int(row["rank"]))
        counts = {decision: sum(row["judgment"] == decision for row in query_rows) for decision in DECISIONS}
        n = len(query_rows)
        target_matches = [
            row for row in query_rows
            if row["judgment"] != "uncertain"
        ]
        annotation_agreement = (
            sum(
                (row["target_label"] in {label for label in row.get("labels", "").split(",") if label})
                == (row["judgment"] == "relevant")
                for row in target_matches
            ) / len(target_matches)
            if target_matches else None
        )
        queries.append(
            {
                "query_id": query_id,
                "query_text": query_rows[0]["query_text"],
                "target_label": query_rows[0]["target_label"],
                "judged_count": n,
                "relevant": counts["relevant"],
                "uncertain": counts["uncertain"],
                "not_relevant": counts["not_relevant"],
                "precision_at_5_lower": counts["relevant"] / n if n else None,
                "precision_at_5_upper": (counts["relevant"] + counts["uncertain"]) / n if n else None,
                "annotation_agreement_on_definite": annotation_agreement,
            }
        )
    per_query_lowers = [item["precision_at_5_lower"] for item in queries if item["precision_at_5_lower"] is not None]
    per_query_uppers = [item["precision_at_5_upper"] for item in queries if item["precision_at_5_upper"] is not None]
    definite = [row for row in rows if row["judgment"] != "uncertain"]
    macro_annotation_agreement = (
        sum(
            (row["target_label"] in {label for label in row.get("labels", "").split(",") if label})
            == (row["judgment"] == "relevant")
            for row in definite
        ) / len(definite)
        if definite else None
    )
    return {
        "status": "complete" if len(rows) == expected_count else "partial",
        "judgment_source": "OpenAI VLM; not human ground truth",
        "model": model,
        "models_used": sorted({row.get("model", model) for row in rows}),
        "prompt_version": PROMPT_VERSION,
        "judged_count": len(rows),
        "expected_count": expected_count,
        "query_count": len(queries),
        "decision_counts": {
            decision: sum(row["judgment"] == decision for row in rows)
            for decision in DECISIONS
        },
        "macro_precision_at_5_lower": sum(per_query_lowers) / len(per_query_lowers) if per_query_lowers else None,
        "macro_precision_at_5_upper": sum(per_query_uppers) / len(per_query_uppers) if per_query_uppers else None,
        "annotation_agreement_on_definite": macro_annotation_agreement,
        "queries": queries,
    }


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _write_outputs(output_dir: Path, rows: Sequence[dict], summary: dict, dataset_root: Path) -> None:
    _write_json_atomic(output_dir / "summary.json", summary)
    fields = [
        "query_id", "query_text", "target_label", "rank", "tile_id", "score",
        "labels", "field_id", "image_path", "judgment", "visual_evidence",
        "confidence", "judgment_source", "model", "requested_model", "prompt_version",
        "response_id", "usage", "judged_at_utc",
    ]
    with (output_dir / "judgments.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["usage"] = json.dumps(row.get("usage"), sort_keys=True)
            writer.writerow(serializable)
    _write_gallery(output_dir / "index.html", rows, output_dir, dataset_root)
    _write_report(output_dir / "report.md", summary)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as destination:
            json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_gallery(path: Path, rows: Sequence[dict], output_dir: Path, dataset_root: Path) -> None:
    cards = []
    for row in rows:
        image = (dataset_root / row["image_path"]).resolve()
        relative_image = quote(Path(os.path.relpath(image, output_dir)).as_posix(), safe="/:._-")
        annotation_labels = escape(row.get("labels", "")) or "none"
        cards.append(
            "<article class='card'>"
            f"<img src='{relative_image}' alt='Retrieved RGB tile for {escape(row['query_id'])}'>"
            f"<div class='content'><p class='query'>{escape(row['query_text'])}</p>"
            f"<p><b>{escape(row['judgment'].replace('_', ' '))}</b> · "
            f"confidence: {escape(row['confidence'])} · rank {escape(str(row['rank']))} · "
            f"cosine {escape(str(row['score']))}</p>"
            f"<p>{escape(row['visual_evidence'])}</p>"
            f"<p class='muted'>Annotation labels, shown after judgment: {annotation_labels}</p>"
            f"<p class='muted'>{escape(row['tile_id'])}</p></div></article>"
        )
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GeoRAG M7 VLM audit</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17211d}
.notice{background:#f2f6f4;padding:1rem;border-left:4px solid #19715f}.card{display:grid;grid-template-columns:minmax(180px,320px) 1fr;gap:1rem;border-bottom:1px solid #ddd;padding:1rem 0}
img{width:100%;height:auto;object-fit:contain;background:#eee}.content p{margin:.5rem 0}.query{font-weight:650}.muted{color:#626b66;font-size:.9rem}
@media(max-width:650px){.card{grid-template-columns:1fr}}</style></head><body>
<h1>GeoRAG M7: VLM-assisted relevance judgments</h1>
<p class="notice">These are model-generated judgments, not human labels. Annotation labels are displayed only after the VLM decision for comparison.</p>
<main>""" + "\n".join(cards) + "</main></body></html>"
    path.write_text(html, encoding="utf-8")


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# M7 VLM-assisted relevance audit",
        "",
        f"**Status:** {summary['status']} ({summary['judged_count']}/{summary['expected_count']} pairs).  ",
        f"**Requested model:** `{summary['model']}`; **API model:** "
        f"`{', '.join(summary['models_used']) or summary['model']}`. "
        "**Judgment source:** OpenAI VLM, not human ground truth.",
        "",
        "The VLM saw each natural-language query and RGB tile, but not the Agriculture-Vision labels. "
        "It returned a relevance decision, confidence category, and brief visible-evidence note. "
        "The confidence category is self-reported and is not calibrated. "
        "Precision bounds treat `uncertain` cases as irrelevant for the lower bound and relevant for the upper bound.",
        "",
        f"- Macro VLM Precision@5 lower bound: {_fmt(summary['macro_precision_at_5_lower'])}",
        f"- Macro VLM Precision@5 upper bound: {_fmt(summary['macro_precision_at_5_upper'])}",
        f"- Decisions: {summary['decision_counts']['relevant']} relevant, "
        f"{summary['decision_counts']['uncertain']} uncertain, "
        f"{summary['decision_counts']['not_relevant']} not relevant.",
        f"- Agreement with dataset labels on definite decisions: {_fmt(summary['annotation_agreement_on_definite'])}",
        "- See [the image gallery](index.html), [row-level CSV](judgments.csv), and [resumable JSONL](judgments.jsonl).",
        "",
        "| Query | Relevant | Uncertain | Not relevant | Precision@5 interval | Annotation agreement* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for query in summary["queries"]:
        interval = f"{_fmt(query['precision_at_5_lower'])} to {_fmt(query['precision_at_5_upper'])}"
        lines.append(
            f"| {query['query_id']} | {query['relevant']} | {query['uncertain']} | "
            f"{query['not_relevant']} | {interval} | {_fmt(query['annotation_agreement_on_definite'])} |"
        )
    lines.extend(
        [
            "",
            "\\* Agreement compares model judgments against available class labels after VLM evaluation. "
            "It is descriptive label agreement, not a measure of human relevance or model correctness.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"
