"""The weekly LLM token budget does not exist (config key, Config card, /config/token-budget,
weekly_budget_status). Checks that nothing of it remains."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from neospike import config, topics
from neospike.dashboard import plugin_api as api

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"


def test_budget_is_gone_from_code_config_and_routes() -> None:
    assert not hasattr(topics, "weekly_budget_status") and not hasattr(config, "DEFAULT_LLM_WEEKLY_TOKEN_BUDGET")
    assert "llm_weekly_token_budget" not in {item.name for item in config.fields(config.NeospikeConfig)}
    app = FastAPI()
    app.include_router(api.router)
    assert TestClient(app).get("/config/token-budget").status_code == 404


def test_config_tab_has_no_budget_card() -> None:
    script = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8-sig")
    for text in ("config.budget.", "token-budget", "TokenQuotaView", "DEFAULT_WEEKLY_TOKEN_BUDGET"):
        assert text not in script, text
