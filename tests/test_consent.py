"""Tests for echotype_lab.consent."""

from __future__ import annotations

import json


class TestConsentModule:
    def test_non_interactive_returns_true(self, tmp_path):
        from echotype_lab.consent import request_consent

        log = tmp_path / "consent.jsonl"
        result = request_consent(log_path=log, non_interactive=True)
        assert result is True

    def test_consent_log_written(self, tmp_path):
        from echotype_lab.consent import request_consent

        log = tmp_path / "consent.jsonl"
        request_consent(log_path=log, non_interactive=True)
        assert log.exists()

        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["accepted"] is True
        assert "timestamp" in record

    def test_declined_returns_false(self, tmp_path, monkeypatch):
        from echotype_lab.consent import request_consent

        monkeypatch.setattr("builtins.input", lambda _: "no")
        log = tmp_path / "consent.jsonl"
        result = request_consent(log_path=log)
        assert result is False

    def test_declined_log_records_false(self, tmp_path, monkeypatch):
        from echotype_lab.consent import request_consent

        monkeypatch.setattr("builtins.input", lambda _: "no")
        log = tmp_path / "consent.jsonl"
        request_consent(log_path=log)

        record = json.loads(log.read_text().strip())
        assert record["accepted"] is False

    def test_log_parent_dirs_created(self, tmp_path):
        from echotype_lab.consent import request_consent

        log = tmp_path / "sub" / "dir" / "consent.jsonl"
        request_consent(log_path=log, non_interactive=True)
        assert log.exists()
