from __future__ import annotations

import csv
import json

from PIL import Image
import pytest

from georag.experiments import vlm_audit


def test_summary_counts_uncertain_as_precision_interval_not_binary_truth() -> None:
    rows = [
        _judgment("q1", "a", 1, "relevant", labels="target"),
        _judgment("q1", "b", 2, "uncertain", labels="other"),
        _judgment("q1", "c", 3, "not_relevant", labels="other"),
        _judgment("q2", "d", 1, "not_relevant", labels="other"),
    ]

    summary = vlm_audit.summarize_judgments(rows, expected_count=4, model="test-model")

    assert summary["status"] == "complete"
    assert summary["judgment_source"] == "OpenAI VLM; not human ground truth"
    assert summary["macro_precision_at_5_lower"] == pytest.approx((1 / 3 + 0) / 2)
    assert summary["macro_precision_at_5_upper"] == pytest.approx((2 / 3 + 0) / 2)
    assert summary["queries"][0]["annotation_agreement_on_definite"] == pytest.approx(1.0)


def test_run_resumes_saved_judgments_and_preserves_human_csv(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "train" / "tile.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(30, 120, 40)).save(image_path)
    audit_csv = tmp_path / "human_audit.csv"
    rows = [
        _csv_row("q1", "tile-a", 1, "tile.jpg"),
        _csv_row("q1", "tile-b", 2, "tile.jpg"),
    ]
    with audit_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original_csv = audit_csv.read_bytes()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(vlm_audit, "_make_client", lambda: object())
    calls: list[str] = []

    def fake_judge(_client, _image, query, model):
        calls.append(query)
        judgment = vlm_audit.VLMJudgment(
            judgment="relevant", visual_evidence="Regular crop rows are visible.", confidence="medium"
        )
        return judgment, f"resp-{len(calls)}", f"{model}-snapshot", {"input_tokens": 8}

    monkeypatch.setattr(vlm_audit, "_judge_one", fake_judge)
    output = tmp_path / "vlm-audit"
    first = vlm_audit.run_vlm_audit(audit_csv, dataset_root, output, model="gpt-test", limit=1)
    second = vlm_audit.run_vlm_audit(audit_csv, dataset_root, output, model="gpt-test")
    third = vlm_audit.run_vlm_audit(audit_csv, dataset_root, output, model="gpt-test")

    assert len(calls) == 2
    assert first["status"] == "partial" and first["judged_count"] == 1
    assert second["judged_count"] == third["judged_count"] == 2
    assert audit_csv.read_bytes() == original_csv
    assert len((output / "judgments.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert "not human labels" in (output / "index.html").read_text(encoding="utf-8")
    assert "Precision@5" in (output / "report.md").read_text(encoding="utf-8")
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete"


def test_single_judgment_sends_query_and_image_only(tmp_path) -> None:
    image_path = tmp_path / "tile.jpg"
    Image.new("RGB", (8, 8), color=(50, 100, 120)).save(image_path)

    class Response:
        id = "response-1"
        model = "gpt-test-snapshot"
        usage = None
        output_parsed = vlm_audit.VLMJudgment(
            judgment="uncertain", visual_evidence="The crop is too small to distinguish the pattern.", confidence="low"
        )

    class Responses:
        def parse(self, **kwargs):
            self.request = kwargs
            return Response()

    class Client:
        responses = Responses()

    client = Client()
    vlm_audit._judge_one(client, image_path, "Show a drydown pattern", "gpt-test")
    request = client.responses.request
    serialized = json.dumps(request["input"])
    assert "Show a drydown pattern" in serialized
    assert "image/jpeg;base64," in serialized
    assert set(request["input"][1]) == {"role", "content"}
    assert {part["type"] for part in request["input"][1]["content"]} == {"input_text", "input_image"}
    assert request["text_format"] is vlm_audit.VLMJudgment
    assert request["model"] == "gpt-test"


def test_run_requires_api_key_before_work(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        vlm_audit.run_vlm_audit(tmp_path / "audit.csv", tmp_path, tmp_path / "out")


def test_existing_state_with_other_model_is_not_mixed(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    audit_csv = tmp_path / "human_audit.csv"
    row = _csv_row("q1", "tile-a", 1, "missing.jpg")
    with audit_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "vlm-audit"
    output.mkdir()
    (output / "judgments.jsonl").write_text(
        json.dumps({**row, "judgment": "relevant", "model": "other", "prompt_version": vlm_audit.PROMPT_VERSION}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="different model"):
        vlm_audit.run_vlm_audit(audit_csv, dataset_root, output, model="requested")


def _csv_row(query_id: str, tile_id: str, rank: int, image_path: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "query_text": "Aerial farmland with crop rows.",
        "target_label": "target",
        "rank": str(rank),
        "tile_id": tile_id,
        "score": "0.8",
        "labels": "target",
        "field_id": "field-1",
        "image_path": f"train/{image_path}",
        "human_relevance": "",
        "notes": "",
    }


def _judgment(query_id: str, tile_id: str, rank: int, decision: str, labels: str) -> dict:
    return {
        **_csv_row(query_id, tile_id, rank, "tile.jpg"),
        "judgment": decision,
        "labels": labels,
    }
