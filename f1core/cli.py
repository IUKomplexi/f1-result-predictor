"""The single ``f1`` command-line interface.

Consolidates the six old console scripts (``f1-predict``, ``f1-train``,
``f1-backtest``, ``f1-calibrate``, ``f1-web``) — plus the
former ``scripts/fetch_all.py`` shim, now the ``f1 fetch`` subcommand — into
one ``f1`` entry point with argparse subcommands. Every subcommand delegates
to the shared keyword-only ``run_*`` wrappers so the CLI and the web job
runner stay on one code path (parity invariant). Per-module ``main()``
delegators were removed when this file was introduced.
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


def _tune(args: argparse.Namespace) -> int:
    from model.tune import run as run_tune

    result = run_tune(
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, dataset=args.dataset,
        out=args.out, out_json=args.out_json,
        candidates=args.candidates, metric=args.metric,
        quantize=not args.no_quantize,
        enable_features=_split_features(args.enable_features),
        disable_features=_split_features(args.disable_features),
        log=lambda msg: print(msg, flush=True),
    )
    print(f"\nBest ({result['metric']}): {result['best']['params']}")
    print(f"  {result['best']['metrics']}")
    print(f"Baseline: {result['baseline']['params']}")
    print(f"  {result['baseline']['metrics']}")
    for rank, row in enumerate(result["top"], start=1):
        print(f"Top-{rank}: {row['params']} -> {row['metrics']}")
    print(f"\nWrote {result['report']}")
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


def _fetch(args: argparse.Namespace) -> int:
    from f1data.fetch import run as run_fetch

    result = run_fetch(
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, sleep=args.sleep,
        log=lambda msg: print(msg, flush=True),
    )
    print(f"Fetched {result['start']}-{result['end']} in {result['elapsed_s']}s")
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

    p = sub.add_parser("fetch", help="fetch + cache raw race data (data/raw)")
    p.add_argument("--start", type=int, default=None, help="default: config [data] start_season")
    p.add_argument("--end", type=int, default=None, help="default: config [data] end_season")
    p.add_argument("--refresh", action="store_true", help="refetch even if cached")
    p.add_argument("--cache-dir", default=None, help="default: config [data] cache_dir")
    p.add_argument("--sleep", type=float, default=None,
                   help="default: config [api] sleep_seconds")
    p.set_defaults(func=_fetch)

    p = sub.add_parser("train", help="train the final model")
    p.add_argument("--start", type=int, default=None, help="default: config [data] start_season")
    p.add_argument("--end", type=int, default=None, help="default: config [data] end_season")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default=None, help="default: config [data] cache_dir")
    p.add_argument("--dataset", default=None, help="default: config [data] dataset")
    p.add_argument("--out", default=None, help="default: config [model] checkpoint")
    _add_common(p)
    p.set_defaults(func=_train)

    p = sub.add_parser("backtest", help="walk-forward backtest vs baselines")
    p.add_argument("--start", type=int, default=None, help="default: config [data] start_season")
    p.add_argument("--end", type=int, default=None, help="default: config [data] end_season")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default=None, help="default: config [data] cache_dir")
    p.add_argument("--dataset", default=None, help="default: config [data] dataset")
    p.add_argument("--out", default=None, help="default: config [report] backtest")
    p.add_argument("--out-json", default=None,
                   help="JSON snapshot for the web dashboard (default: --out with .json)")
    p.add_argument("--no-quantize", action="store_true",
                   help="keep continuous expected points (deployed output is quantized)")
    _add_common(p)
    p.set_defaults(func=_backtest)

    p = sub.add_parser("calibrate", help="fit + deploy isotonic probability calibrators")
    p.add_argument("--start", type=int, default=None, help="default: config [data] start_season")
    p.add_argument("--end", type=int, default=None, help="default: config [data] end_season")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default=None, help="default: config [data] cache_dir")
    p.add_argument("--dataset", default=None, help="default: config [data] dataset")
    p.add_argument("--out", default=None, help="default: config [model] calibrators")
    p.add_argument("--out-json", default=None,
                   help="JSON snapshot for the web dashboard (default: reports/calibration.json)")
    p.add_argument("--fit-through", dest="fit_through", type=int, default=None,
                   help="latest season to fit calibrators on (default: all but eval window)")
    p.add_argument("--eval-from", dest="eval_from", type=int, default=None,
                   help="first season to evaluate hold-out Brier on (default: none)")
    _add_common(p)
    p.set_defaults(func=_calibrate)

    p = sub.add_parser("tune", help="walk-forward hyperparameter search (f1 tune)")
    p.add_argument("--start", type=int, default=None, help="default: config [data] start_season")
    p.add_argument("--end", type=int, default=None, help="default: config [data] end_season")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default=None, help="default: config [data] cache_dir")
    p.add_argument("--dataset", default=None, help="default: config [data] dataset")
    p.add_argument("--out", default=None, help="default: reports/tuning.md")
    p.add_argument("--out-json", default=None,
                   help="JSON snapshot (default: --out with .json)")
    p.add_argument("--candidates", type=int, default=24,
                   help="number of param sets to evaluate (default: 24, max 576)")
    p.add_argument("--metric", choices=["mae", "winner_hit", "top3_overlap", "spearman"],
                   default="mae", help="objective metric (default: mae)")
    p.add_argument("--no-quantize", action="store_true",
                   help="keep continuous expected points (deployed output is quantized)")
    _add_common(p)
    p.set_defaults(func=_tune)

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
