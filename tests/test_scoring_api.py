"""Rating engine, batch scoring and the FastAPI endpoints, on a tariff built from a simulated book."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fremtpl2_pricing.api import app as app_module
from fremtpl2_pricing.config import Config
from fremtpl2_pricing.models.frequency import run_frequency
from fremtpl2_pricing.models.frequency import save_artifact as save_freq
from fremtpl2_pricing.models.severity import run_severity
from fremtpl2_pricing.models.severity import save_artifact as save_sev
from fremtpl2_pricing.reporting.phase1 import _write_tables
from fremtpl2_pricing.scoring.batch import score_file
from fremtpl2_pricing.scoring.engine import TARIFF_FILE, RatingEngine, build_tariff, save_tariff
from tests.test_benchmarks_compare import _book

POLICY = {
    "VehPower": 6,
    "VehAge": 4,
    "DrivAge": 35,
    "BonusMalus": 50,
    "VehBrand": "B12",
    "VehGas": "Regular",
    "Density": 1200.0,
    "Region": "R24",
    "exposure": 1.0,
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tariff")
    cfg, con = _book(tmp)
    cfg = Config(raw={**cfg.raw, "scoring": {**cfg.raw["scoring"], "variant": "full"}}, root=cfg.root)
    con.execute("CREATE OR REPLACE TABLE claims_clean AS SELECT *, 0.0 AS large_loss_excess FROM claims_clean")
    art = cfg.path("artifacts_dir")
    fr = run_frequency(con, cfg)
    save_freq(fr, cfg, art / "frequency_glm.json")
    _write_tables({"freq_factor_tests": fr.factor_tests}, cfg.path("tables_dir"))
    save_sev(run_severity(con, cfg), cfg, art / "severity_glm.json")
    save_tariff(build_tariff(con, cfg), art / TARIFF_FILE)
    return art


@pytest.fixture(scope="module")
def engine(artifacts: Path) -> RatingEngine:
    return RatingEngine.load(artifacts)


@pytest.fixture
def client(artifacts: Path) -> Iterator[TestClient]:
    with TestClient(app_module.create_app(artifacts)) as c:
        yield c


def test_tariff_is_versioned_and_reproducible(engine: RatingEngine, artifacts: Path) -> None:
    assert engine.model_version.startswith("fremtpl2-glm-tariff-")
    t = engine.tariff
    assert t["short_period_factors"]["(0.999, 1]"] == pytest.approx(1.0)
    assert set(t["large_loss_loading"]) == {"full_year", "part_year"}


def test_quote_formula(engine: RatingEngine) -> None:
    df = pd.DataFrame([POLICY, {**POLICY, "exposure": 0.5}])
    q = engine.quote(df)
    t = engine.tariff
    full = q.iloc[0]
    assert full["premium_for_term"] == pytest.approx(full["technical_premium_annual"])
    assert full["technical_premium_annual"] == pytest.approx(
        full["frequency"]
        * full["severity"]
        * (1 + t["large_loss_loading"]["full_year"])
        / (1 - t["expense_ratio"] - t["profit_margin"])
    )
    half = q.iloc[1]
    sp = t["short_period_factors"]["(0.25, 0.5]"]
    ll = (1 + t["large_loss_loading"]["part_year"]) / (1 + t["large_loss_loading"]["full_year"])
    assert half["premium_for_term"] == pytest.approx(half["technical_premium_annual"] * 0.5 * sp * ll)
    assert (q["model_version"] == engine.model_version).all()


def test_cap_limits_change_against_current_premium(engine: RatingEngine) -> None:
    uncapped = engine.quote(pd.DataFrame([POLICY]))["premium_for_term"].iloc[0]
    cur = uncapped / 2
    q = engine.quote(pd.DataFrame([{**POLICY, "current_premium": cur}])).iloc[0]
    assert q["premium_for_term"] == pytest.approx(cur * (1 + engine.tariff["cap"]))
    assert bool(q["capped"])


def test_validation_rejects_with_reasons(engine: RatingEngine) -> None:
    bad = pd.DataFrame(
        [
            {**POLICY, "DrivAge": 12},
            {**POLICY, "Region": "R99"},
            {**POLICY, "exposure": 1.5},
            {**POLICY, "BonusMalus": None},
            POLICY,
        ]
    )
    valid, rejected = engine.validate(bad)
    assert len(valid) == 1 and len(rejected) == 4
    reasons = list(rejected["reject_reason"])
    assert "DrivAge outside" in reasons[0] and "Region" in reasons[1] and "exposure" in reasons[2]
    assert "BonusMalus" in reasons[3]


def test_top_factors_are_sorted_by_effect(engine: RatingEngine) -> None:
    tops = engine.top_factors(pd.DataFrame([{**POLICY, "BonusMalus": 120}]))[0]
    assert 0 < len(tops) <= engine.tariff["top_factors"]
    sizes = [abs(np.log(t["relativity"])) for t in tops]
    assert sizes == sorted(sizes, reverse=True)


def test_batch_file_round_trip(engine: RatingEngine, tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    pd.DataFrame([POLICY, {**POLICY, "VehBrand": "B99"}]).to_csv(src, index=False)
    n_ok, n_bad = score_file(engine, src, tmp_path / "out.csv")
    assert (n_ok, n_bad) == (1, 1)
    out = pd.read_csv(tmp_path / "out.csv")
    assert {"premium_for_term", "model_version", "top_factors", "scored_at"} <= set(out.columns)
    assert "reject_reason" in pd.read_csv(tmp_path / "out_rejected.csv").columns


def test_api_endpoints(client: TestClient, engine: RatingEngine) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_version": engine.model_version}
    info = client.get("/model-info").json()
    assert "Region" in info["categories"] and info["model_version"] == engine.model_version
    r = client.post("/quote", json=POLICY)
    assert r.status_code == 200
    body = r.json()
    expected = engine.quote(pd.DataFrame([POLICY]))["premium_for_term"].iloc[0]
    assert body["premium_for_term"] == pytest.approx(expected)
    assert body["top_factors"] and body["model_version"] == engine.model_version
    batch = client.post("/quote/batch", json={"policies": [POLICY, {**POLICY, "exposure": 0.25}]})
    assert batch.status_code == 200 and len(batch.json()["quotes"]) == 2


def test_api_rejects_bad_input(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.post("/quote", json={**POLICY, "VehBrand": "B99"}).status_code == 422  # unknown level
    assert client.post("/quote", json={**POLICY, "DrivAge": 17}).status_code == 422  # below tariff range
    assert client.post("/quote", json={**POLICY, "colour": "red"}).status_code == 422  # unexpected field
    assert client.post("/quote", json={k: v for k, v in POLICY.items() if k != "Region"}).status_code == 422
    monkeypatch.setattr(app_module, "MAX_BATCH", 1)
    assert client.post("/quote/batch", json={"policies": [POLICY, POLICY]}).status_code == 413


def test_filable_variant_drops_region_and_merges_malus_bands(artifacts: Path, tmp_path: Path) -> None:
    from fremtpl2_pricing.config import load_config
    from tests.test_benchmarks_compare import _book

    cfg, con = _book(tmp_path)
    con.execute("CREATE OR REPLACE TABLE claims_clean AS SELECT *, 0.0 AS large_loss_excess FROM claims_clean")
    import shutil

    shutil.copytree(artifacts, cfg.path("artifacts_dir"), dirs_exist_ok=True)
    raw = {**cfg.raw, "scoring": {**cfg.raw["scoring"], "variant": "filable"}}
    t = build_tariff(con, Config(raw=raw, root=cfg.root))
    names = [f["name"] for f in t["frequency"]["design"]["factors"]]
    assert t["variant"] == "filable" and "Region" not in names
    bm = next(f for f in t["frequency"]["design"]["factors"] if f["name"] == "BonusMalus")
    assert bm["edges"] == [float(x) for x in load_config().section("pricing")["filable_glm"]["bonusmalus_bands"]]
    eng = RatingEngine(t)
    q = eng.quote(eng.validate(pd.DataFrame([{**POLICY, "Region": "R99"}]))[0])  # Region no longer rated
    assert len(q) == 1 and q["premium_for_term"].iloc[0] > 0
    assert eng.model_version != RatingEngine.load(artifacts).model_version
