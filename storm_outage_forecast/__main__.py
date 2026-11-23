from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bundle, download, pipeline, synthetic
from .config import ConfigError, load_config


def report(metrics: dict) -> str:
    if not metrics:
        return "no metrics yet, run the pipeline first"
    lines = [f"data: {metrics.get('data_kind')}  years: {metrics.get('data', {}).get('years')}"]
    risk = metrics.get("risk", {}).get("test", {})
    for name in ("climatology", "logistic", "lightgbm"):
        m = risk.get(name, {})
        if m:
            lines.append(
                f"{name:12s} auc {m.get('roc_auc', 0):.3f}  ap {m.get('average_precision', 0):.3f}"
                f"  brier skill {m.get('brier_skill', 0):.3f}  ece {m.get('ece', 0):.4f}"
            )
    size = metrics.get("risk", {}).get("size", {})
    if size:
        lines.append(
            f"size interval coverage {size['interval_coverage']:.3f} "
            f"(nominal {size['nominal_coverage']:.2f})"
        )
    rest = metrics.get("restoration", {})
    if rest:
        lines.append(
            f"restoration c-index {rest['c_index_model']:.3f} "
            f"(cause medians {rest['c_index_cause_median']:.3f})"
        )
    stage = metrics.get("staging", {})
    if stage:
        share = stage["share_of_perfect_savings"]
        lines.append(
            f"crew staging: forecast {share['forecast']:.2f}, climatology "
            f"{share['climatology']:.2f} of perfect-foresight savings"
        )
    rel = metrics.get("reliability", {})
    if rel.get("spearman") is not None:
        lines.append(
            f"EIA-861 SAIDI check: spearman {rel['spearman']:.3f} over {rel['state_years']} "
            f"state-years"
        )
    lines.append(f"timing: {json.dumps(metrics.get('timing_s', {}))}")
    lines.append(f"peak memory MB: {json.dumps(metrics.get('peak_memory_mb', {}))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m storm_outage_forecast")
    parser.add_argument("--config", default=None, help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="download the public data sets")
    fetch.add_argument("--only", default="", help=f"comma list from {','.join(download.SOURCES)}")
    sub.add_parser("probe", help="check that every source url answers")
    synth = sub.add_parser("synth", help="write a synthetic replica in the real file formats")
    synth.add_argument("--profile", choices=sorted(synthetic.PROFILES), default="tiny")
    resume = {
        "dest": "start",
        "choices": pipeline.STAGES,
        "default": None,
        "help": "resume from this stage and keep what the warehouse already holds",
    }
    builder = sub.add_parser("build", help="raw files to the DuckDB warehouse and model table")
    builder.add_argument("--from", **resume)
    sub.add_parser("train", help="risk model, restoration model and crew staging")
    sub.add_parser("bundle", help="export the dashboard data")
    whole = sub.add_parser("run", help="build, train and bundle")
    whole.add_argument("--from", **resume)
    sub.add_parser("report", help="print the headline metrics")
    space = sub.add_parser("space", help="assemble the Hugging Face Space folder")
    space.add_argument("--dest", required=True, help="folder for the Space files")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 2
    if args.command == "download":
        only = [s.strip() for s in args.only.split(",") if s.strip()] or None
        try:
            download.run(cfg, only)
        except download.DownloadError as err:
            print(f"download error: {err}", file=sys.stderr)
            return 1
    elif args.command == "probe":
        rows = download.probe(cfg)
        return 0 if all(r["status"] == "ok" for r in rows) else 1
    elif args.command == "synth":
        synthetic.generate(cfg, args.profile)
    elif args.command == "build":
        pipeline.build(cfg, start=args.start)
    elif args.command == "train":
        pipeline.train(cfg)
    elif args.command == "bundle":
        pipeline.make_bundle(cfg)
    elif args.command == "run":
        pipeline.run(cfg, start=args.start)
        print(report(pipeline.read_metrics(cfg)))
    elif args.command == "report":
        print(report(pipeline.read_metrics(cfg)))
    elif args.command == "space":
        bundle.build_space(cfg, Path(args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
