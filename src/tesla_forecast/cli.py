"""Command line interface:  ``tesla-forecast {data,backtest,train,forecast,figures}``."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import load_config
from .forecasting.pipeline import ForecastPipeline
from .utils import get_logger

log = get_logger(__name__)


def _summary(pipe: ForecastPipeline) -> str:
    m = pipe.metrics
    h = pipe.horizon
    hs = sorted({1, 5, h})
    sub = m[m.horizon.isin(hs)][["model", "horizon", "RMSE", "MAE", "MASE", "DirAcc_%", "Skill_vs_naive_%", "DM_p"]]
    return sub.sort_values(["horizon", "RMSE"]).to_string(index=False, float_format=lambda x: f"{x:,.3f}")


def _build(args) -> ForecastPipeline:
    overrides: dict = {}
    if args.horizon:
        overrides["horizon"] = args.horizon
    if args.offline:
        overrides.setdefault("data", {})["offline"] = True
    if args.models:
        overrides.setdefault("models", {})["enabled"] = args.models.split(",")
    if args.quick:
        overrides.setdefault("models", {}).setdefault("deep", {}).update(n_seeds=1, max_epochs=25)
        overrides["backtest"] = {"n_test_origins": 378, "refit_every": 126}
    cfg = load_config(args.config, overrides)
    return ForecastPipeline(cfg).prepare(force_refresh=args.refresh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tesla-forecast", description=__doc__)
    ap.add_argument("command", choices=["backtest", "train", "forecast", "data", "figures"])
    ap.add_argument("--config", help="extra YAML merged over configs/default.yaml")
    ap.add_argument("--horizon", type=int, help="forecast length in trading days")
    ap.add_argument("--models", help="comma-separated subset, e.g. naive,ridge,lightgbm")
    ap.add_argument("--quick", action="store_true", help="smaller/faster backtest (smoke runs)")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    ap.add_argument("--refresh", action="store_true", help="ignore the data cache")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 160)

    if args.command == "data":
        pipe = _build(args)
        print(pipe.prices.df.tail(10))
        print(f"\nsource={pipe.prices.source}  rows={len(pipe.prices.df)}  notes={pipe.prices.notes}")
        return 0

    if args.command == "figures":
        from .viz.report import save_report_figures

        cfg = load_config(args.config, {"data": {"offline": True}} if args.offline else None)
        pipe = ForecastPipeline.load(cfg=cfg).prepare(force_refresh=args.refresh)
        res = pipe.forecast()
        out = cfg.resolve(cfg.paths.reports_dir) / "figures"
        for path in save_report_figures(pipe, res, pipe.simulate(res), out):
            print("saved", path.relative_to(cfg.root))
        return 0

    if args.command == "forecast":
        cfg = load_config(args.config, {"data": {"offline": True}} if args.offline else None)
        pipe = ForecastPipeline.load(cfg=cfg)
        pipe.prepare(force_refresh=args.refresh)
        res = pipe.forecast()
        print(f"\nTSLA forecast as of {res.as_of.date()} (last close ${res.last_close:,.2f}, data: {res.data_source})")
        print(res.to_frame().round(2).to_string())
        return 0

    pipe = _build(args)
    pipe.backtest()
    print("\nOut-of-sample walk-forward results\n" + _summary(pipe))
    if args.command == "train":
        pipe.fit()
        pipe.save()
        res = pipe.forecast()
        print(f"\nForecast as of {res.as_of.date()} (last close ${res.last_close:,.2f}):")
        print(res.to_frame().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
