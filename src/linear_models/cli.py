"""Command line: ``linear-models replicate | loglog | figures | all``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from linear_models import analysis, data


def _coef_table(model: dict) -> str:
    lines = [f"  {'term':<16}{'coef':>14}{'std err':>12}{'t':>10}{'p':>11}"]
    for name, c in model["coefficients"].items():
        lines.append(
            f"  {name:<16}{c['coef']:>14.6g}{c['se']:>12.4g}{c['t']:>10.3f}{c['p']:>11.3g}"
        )
    lines.append(
        f"  n = {model['n']}, R² = {model['r2']:.4f}, adj. R² = {model['adj_r2']:.4f}, "
        f"F = {model['f_stat']:.4g} (p = {model['f_pvalue']:.3g})"
    )
    return "\n".join(lines)


def _residual_lines(res: dict) -> str:
    return (
        f"  residuals: median {res['median']:.4g}, skewness {res['skewness']:.2f}, "
        f"excess kurtosis {res['excess_kurtosis']:.1f}, Jarque-Bera {res['jarque_bera']:.4g} "
        f"(p = {res['jarque_bera_p']:.3g}), Breusch-Pagan {res['breusch_pagan']:.3g} "
        f"(p = {res['breusch_pagan_p']:.3g})\n"
        "  largest residuals: "
        + ", ".join(f"{t['country']} ({t['share_of_rss']:.1%} of RSS)" for t in res["top"])
    )


def print_replicate(r: dict) -> None:
    print("Coursework model, Y = land area, X1 = population, X2 = density")
    for name, s in r["simple"].items():
        print(f"  rho(Y, {name}) = {s['rho']:+.4f}   R² = {s['r2']:.4f}")
    print(_coef_table(r["multiple"]))
    ne = r["normal_equations"]
    print(
        f"  normal equations vs QR: max relative difference {ne['max_rel_diff_vs_qr']:.2g}; "
        f"cond(X) = {ne['cond_X']:.3g}, cond(XᵀX) = {ne['cond_XtX']:.3g}"
    )
    print(_residual_lines(r["residuals"]))
    print(f"  top 5 residuals hold {r['residuals']['top5_share_of_rss']:.1%} of the RSS")


def print_loglog(r: dict) -> None:
    print("Log-log model, rows with every value > 0 (dropped: " + ", ".join(r["dropped"]) + ")")
    print(_coef_table(r["model"]))
    for name, t in r["identity_tests"].items():
        print(f"  H0 {name}: t = {t['t']:+.3f}, p = {t['p']:.3g}")
    print(_residual_lines(r["residuals"]))
    rb = r["robust"]
    print(
        f"Robustness: density >= {rb['min_density']:g} and area >= {rb['min_area']:g} km² "
        "(small rounding error)"
    )
    print(_coef_table(rb["model"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linear-models", description=__doc__)
    parser.add_argument("command", choices=["replicate", "loglog", "figures", "all"])
    parser.add_argument("--data", type=Path, default=None, help="CSV path (default: data/)")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/img"), help="figure folder")
    parser.add_argument("--json", type=Path, default=None, help="write all results as JSON")
    args = parser.parse_args(argv)

    df = data.load(args.data)
    results: dict = {}
    if args.command in ("replicate", "all"):
        results["replicate"] = analysis.replicate(df)
        print_replicate(results["replicate"])
    if args.command in ("loglog", "all"):
        if results:
            print()
        results["loglog"] = analysis.loglog(df)
        print_loglog(results["loglog"])
    if args.command in ("figures", "all"):
        from linear_models.plots import all_figures

        for path in all_figures(df, args.out_dir):
            print(f"wrote {path}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
