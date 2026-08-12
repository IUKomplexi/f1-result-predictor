"""The single ``f1`` command-line interface.

Consolidates the six old console scripts (``f1-predict``, ``f1-train``,
``f1-backtest``, ``f1-calibrate``, ``f1-search``, ``f1-web``) into one
``f1`` entry point with argparse subcommands. Every subcommand delegates to the
shared keyword-only ``run_*`` wrappers so the CLI and the web job runner stay
on one code path (parity invariant). Per-module ``main()`` delegators were
removed when this file was introduced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_COMMON = {
    "enable_features": {
        "flags": ("--enable-features",),
        "kwargs": {"default": "", "help": "comma-separated features to enable on top of config"},
    },
    "disable_features": {
        "flags": ("--disable-features",),
        "kwargs": {"default": "", "help": "comma-separated features to disable on top of config"},
    },
}


def _split_features(comma: str) -> list[str]:
    return [f for f in comma.split(",") if f]


def _add_common(p: argparse.ArgumentParser) -> None:
    for spec in _COMMON.values():
        p.add_argument(*spec["flags"], **spec["kwargs"])


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------

def _predict(args: argparse.Namespace) -> int:
    from f1core.config import load_config
    from f1core.predict import format_console, format_report, get_prediction

    cfg = load_config()
    try:
        pred = get_prediction(
            season=args.season,
            round_=args.round,
            grid_csv=args.grid,
            refresh=args.refresh,
            cfg=cfg,
            model_path=args.model,
            enable_features=_split_features(args.enable_features),
            disable_features=_split_features(args.disable_features),
        )
    except ValueError as exc:
        # get_prediction raises ValueError for bad arguments (e.g. --season
        # without --round); keep the CLI error clean, without a traceback.
        raise SystemExit(str(exc)) from None
    result, meta = pred["result"], pred["meta"]
    target_season, target_round = pred["season"], pred["round"]

    report = format_report(
        result, target_season, target_round, meta,
        verified=pred["verified"], checkpoint=pred["checkpoint"],
        calibrated=pred["calibrated"],
    )
    out = Path(args.out or cfg["report"]["prediction"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(format_console(result, meta, target_season, target_round))
    print(f"\nWrote {out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from model.train import run as run_train

    run_train(
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, dataset=args.dataset, out=args.out,
        enable_features=_split_features(args.enable_features),
        disable_features=_split_features(args.disable_features),
        log=lambda msg: print(msg, flush=True),
    )
    return 0


def _backtest(args: argparse.Namespace) -> int:
    from model.evaluate import run as run_backtest

    result = run_backtest(
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, dataset=args.dataset,
        out=args.out, out_json=args.out_json, quantize=not args.no_quantize,
        enable_features=_split_features(args.enable_features),
        disable_features=_split_features(args.disable_features),
        log=lambda msg: print(msg, flush=True),
    )
    overall = pd.DataFrame(result["overall"]).T
    print(
        f"Features ({result['n_features']}, fp {result['fingerprint']}): "
        f"{', '.join(result['features'])}"
    )
    print(overall.to_string())
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from model.calibrate import run as run_calibrate

    result = run_calibrate(
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, dataset=args.dataset,
        out=args.out, out_json=args.out_json,
        fit_through_season=args.fit_through, eval_from_season=args.eval_from,
        enable_features=_split_features(args.enable_features),
        disable_features=_split_features(args.disable_features),
        log=lambda msg: print(msg, flush=True),
    )
    print(result["summary"])
    print(f"\nWrote {result['calibrators']}")
    return 0


def _search(args: argparse.Namespace) -> int:
    from model.search import run as run_search

    result = run_search(
        n=args.n, seed=args.seed, max_test_season=args.max_test_season,
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, dataset=args.dataset,
        enable_features=_split_features(args.enable_features),
        disable_features=_split_features(args.disable_features),
        log=lambda msg: print(msg, flush=True),
    )
    print(f"Walk-forward search (test seasons <= {args.max_test_season}):")
    print(pd.DataFrame(result["results"]).to_string(index=False))
    print("\nBest configuration (write to config [model.params] or DEFAULT_PARAMS):")
    print(result["best"])
    return 0


def _web(args: argparse.Namespace) -> int:
    import uvicorn

    from f1web.app import create_app

    level = "debug" if args.debug else "info"
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=level)
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="f1", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict a race's points")
    p.add_argument("--season", type=int, help="target season (default: next race)")
    p.add_argument("--round", type=int, help="target round (required with --season)")
    p.add_argument("--grid", help="CSV with 'driver_id,grid' for an upcoming race")
    p.add_argument(
        "--model", help="model checkpoint path (default: config [model] checkpoint)"
    )
    p.add_argument("--out", help="report path (default: reports/prediction.md)")
    p.add_argument("--refresh", action="store_true", help="ignore the raw-data cache")
    _add_common(p)
    p.set_defaults(func=_predict)

    p = sub.add_parser("train", help="train the final model")
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--dataset", default="data/features.parquet")
    p.add_argument("--out", default="data/model/hurdle.joblib")
    _add_common(p)
    p.set_defaults(func=_train)

    p = sub.add_parser("backtest", help="walk-forward backtest vs baselines")
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--dataset", default="data/features.parquet")
    p.add_argument("--out", default="reports/backtest.md")
    p.add_argument("--out-json", default="reports/backtest.json",
                   help="JSON snapshot for the web dashboard")
    p.add_argument("--no-quantize", action="store_true",
                   help="keep continuous expected points (deployed output is quantized)")
    _add_common(p)
    p.set_defaults(func=_backtest)

    p = sub.add_parser("calibrate", help="fit + deploy isotonic probability calibrators")
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--dataset", default="data/features.parquet")
    p.add_argument("--out", default="data/model/calibrators.joblib")
    p.add_argument("--out-json", default="reports/calibration.json",
                   help="JSON snapshot for the web dashboard")
    p.add_argument("--fit-through", dest="fit_through", type=int, default=None,
                   help="latest season to fit calibrators on (default: all but eval window)")
    p.add_argument("--eval-from", dest="eval_from", type=int, default=None,
                   help="first season to evaluate hold-out Brier on (default: none)")
    _add_common(p)
    p.set_defaults(func=_calibrate)

    p = sub.add_parser("search", help="walk-forward hyperparameter search")
    p.add_argument("--n", type=int, default=16, help="configs to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-test-season", type=int, default=2019,
                   help="latest test season in the search window")
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--dataset", default="data/features.parquet")
    _add_common(p)
    p.set_defaults(func=_search)

    p = sub.add_parser("web", help="run the FastAPI app + dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=_web)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
