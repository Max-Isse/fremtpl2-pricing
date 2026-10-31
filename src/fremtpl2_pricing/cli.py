"""Command-line entry point: `fremtpl2 <step>` (or `python -m fremtpl2_pricing.cli <step>`)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from fremtpl2_pricing.config import Config, load_config

log = logging.getLogger("fremtpl2")


def cmd_fetch(cfg: Config, force: bool = False) -> None:
    from fremtpl2_pricing.data.fetch import fetch_raw

    paths = fetch_raw(cfg, force=force)
    log.info("raw data cached: %s", {k: str(v) for k, v in paths.items()})


def cmd_build(cfg: Config) -> None:
    from fremtpl2_pricing.data.build import build_database

    build_database(cfg).close()
    log.info("database built: %s", cfg.path("duckdb"))


def cmd_report(cfg: Config) -> None:
    from fremtpl2_pricing.analysis.exposure import run_exposure_analysis
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.data.quality import build_quality_report
    from fremtpl2_pricing.reporting.phase1 import write_exposure_report, write_quality_report

    con = connect(cfg)
    try:
        log.info("wrote %s", write_quality_report(build_quality_report(con, cfg), cfg))
        log.info("wrote %s", write_exposure_report(run_exposure_analysis(con, cfg), cfg))
    finally:
        con.close()


def cmd_frequency(cfg: Config) -> None:
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.models.frequency import run_frequency, save_artifact
    from fremtpl2_pricing.reporting.frequency import write_frequency_report

    con = connect(cfg, read_only=True)
    try:
        res = run_frequency(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", save_artifact(res, cfg, cfg.path("artifacts_dir") / "frequency_glm.json"))
    log.info("wrote %s", write_frequency_report(res, cfg))


def cmd_severity(cfg: Config) -> None:
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.models.severity import run_severity, save_artifact
    from fremtpl2_pricing.reporting.severity import write_severity_report

    con = connect(cfg, read_only=True)
    try:
        res = run_severity(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", save_artifact(res, cfg, cfg.path("artifacts_dir") / "severity_glm.json"))
    log.info("wrote %s", write_severity_report(res, cfg))


def cmd_pure_premium(cfg: Config) -> None:
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.models.pure_premium import run_pure_premium, save_artifact
    from fremtpl2_pricing.reporting.pure_premium import write_pure_premium_report

    con = connect(cfg, read_only=True)
    try:
        res = run_pure_premium(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", save_artifact(res, cfg, cfg.path("artifacts_dir") / "pure_premium.json"))
    log.info("wrote %s", write_pure_premium_report(res, cfg))


def cmd_benchmarks(cfg: Config) -> None:
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.models.benchmarks import run_benchmarks, save_benchmarks

    con = connect(cfg, read_only=True)
    try:
        res = run_benchmarks(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", save_benchmarks(res, cfg))
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    for name, t in res.tuning.items():
        t.to_csv(tables / f"tuning_{name}.csv", index=False, float_format="%.10g")


def cmd_compare(cfg: Config) -> None:
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.evaluation.compare import run_comparison
    from fremtpl2_pricing.reporting.model_comparison import write_comparison_report

    con = connect(cfg, read_only=True)
    try:
        res = run_comparison(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", write_comparison_report(res, cfg))


def cmd_explain(cfg: Config) -> None:
    from fremtpl2_pricing.analysis.explain import run_explain
    from fremtpl2_pricing.analysis.fairness import run_fairness
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.reporting.explain_fairness import write_explain_fairness_report

    con = connect(cfg, read_only=True)
    try:
        ex, fa = run_explain(con, cfg), run_fairness(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", write_explain_fairness_report(ex, fa, cfg))


def cmd_simulate(cfg: Config) -> None:
    from fremtpl2_pricing.data.db import connect
    from fremtpl2_pricing.pricing.simulation import run_simulation
    from fremtpl2_pricing.reporting.strategy_simulation import write_simulation_report

    con = connect(cfg, read_only=True)
    try:
        res = run_simulation(con, cfg)
    finally:
        con.close()
    log.info("wrote %s", write_simulation_report(res, cfg))


STEPS = [
    "fetch",
    "build",
    "report",
    "frequency",
    "severity",
    "pure_premium",
    "benchmarks",
    "compare",
    "explain",
    "simulate",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fremtpl2", description=__doc__)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--config", default=None, help="path to YAML config")
    parser.add_argument("--force", action="store_true", help="re-download raw data")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    steps: dict[str, Callable[[], None]] = {
        "fetch": lambda: cmd_fetch(cfg, args.force),
        "build": lambda: cmd_build(cfg),
        "report": lambda: cmd_report(cfg),
        "frequency": lambda: cmd_frequency(cfg),
        "severity": lambda: cmd_severity(cfg),
        "pure_premium": lambda: cmd_pure_premium(cfg),
        "benchmarks": lambda: cmd_benchmarks(cfg),
        "compare": lambda: cmd_compare(cfg),
        "explain": lambda: cmd_explain(cfg),
        "simulate": lambda: cmd_simulate(cfg),
    }
    for name in STEPS if args.step == "all" else [args.step]:
        steps[name]()


if __name__ == "__main__":
    main()
