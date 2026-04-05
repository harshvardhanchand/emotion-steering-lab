"""Alpha sweep runner and summary renderers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from art.errors import ArtError
from art.steering.run import run_steering


def alpha_values(alpha_min: float, alpha_max: float, alpha_step: float) -> list[float]:
    if alpha_step <= 0:
        raise ArtError("--alpha-step must be > 0")
    vals: list[float] = []
    x = alpha_min
    while x <= alpha_max + 1e-9:
        vals.append(round(x, 6))
        x += alpha_step
    return vals


def run_alpha_sweep(
    *,
    probe_artifact: dict[str, Any],
    probe_name: str,
    cases: list[dict[str, Any]],
    base_run_id: str,
    sweep_run_id: str,
    scope: str,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    backend_name: str,
    model_id: str,
    tokenizer_id: str,
    device: str,
    dtype: str,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], str, str]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, float]] = []
    by_alpha_category: dict[tuple[float, str], dict[str, float]] = {}

    values = alpha_values(alpha_min, alpha_max, alpha_step)
    for idx, alpha in enumerate(values):
        run_id = f"{sweep_run_id}_a{idx:03d}"
        out = run_steering(
            probe_artifact=probe_artifact,
            probe_name=probe_name,
            alpha=alpha,
            cases=cases,
            base_run_id=base_run_id,
            steer_run_id=run_id,
            scope=scope,
            backend_name=backend_name,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        rows.extend(out)

        sample_rows = [r for r in out if r.get("record_type") == "sample"]
        if not sample_rows:
            continue
        n = float(len(sample_rows))
        base_fail = sum(1 for r in sample_rows if bool(r["baseline_failure"])) / n
        steer_fail = sum(1 for r in sample_rows if bool(r["steered_failure"])) / n
        base_crit = sum(1 for r in sample_rows if int(r["baseline_final_severity"]) == 3) / n
        steer_crit = sum(1 for r in sample_rows if int(r["steered_final_severity"]) == 3) / n
        summary.append(
            {
                "alpha": float(alpha),
                "sample_count": float(n),
                "baseline_failure_rate": float(base_fail),
                "steered_failure_rate": float(steer_fail),
                "delta_failure_rate": float(steer_fail - base_fail),
                "baseline_critical_rate": float(base_crit),
                "steered_critical_rate": float(steer_crit),
                "delta_critical_rate": float(steer_crit - base_crit),
            }
        )

        aggregates = [r for r in out if r.get("record_type") == "aggregate"]
        for agg in aggregates:
            by_alpha_category[(float(alpha), str(agg["category"]))] = {
                "delta_failure_rate": float(agg["delta_failure_rate"]),
                "delta_critical_rate": float(agg["delta_critical_rate"]),
            }

    if not summary:
        raise ArtError("Alpha sweep produced no sample rows")

    summary.sort(key=lambda x: x["alpha"])
    best = min(summary, key=lambda x: x["steered_failure_rate"])

    # Markdown summary with compact bar plots.
    md_lines = [
        "# Steering Alpha Sweep Summary",
        "",
        f"- Probe: `{probe_name}`",
        f"- Scope: `{scope}`",
        f"- Best alpha by steered failure rate: `{best['alpha']}`",
        "",
        "| alpha | n | base_fail | steered_fail | delta_fail | base_crit | steered_crit | delta_crit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        md_lines.append(
            "| "
            f"{row['alpha']:.3f} | "
            f"{int(row['sample_count'])} | "
            f"{row['baseline_failure_rate']:.3f} | "
            f"{row['steered_failure_rate']:.3f} | "
            f"{row['delta_failure_rate']:+.3f} | "
            f"{row['baseline_critical_rate']:.3f} | "
            f"{row['steered_critical_rate']:.3f} | "
            f"{row['delta_critical_rate']:+.3f} |"
        )

    md_lines.append("")
    md_lines.append("## Delta Failure Plot")
    md_lines.append("")
    for row in summary:
        magnitude = min(40, int(abs(row["delta_failure_rate"]) * 80))
        bar = "#" * magnitude if magnitude > 0 else "."
        sign = "+" if row["delta_failure_rate"] >= 0 else "-"
        md_lines.append(f"- alpha={row['alpha']:.3f} {sign}{bar} ({row['delta_failure_rate']:+.3f})")

    # Category table
    categories = sorted({k[1] for k in by_alpha_category})
    if categories:
        md_lines.extend(["", "## By Category", "", "| alpha | category | delta_fail | delta_crit |", "|---:|---|---:|---:|"])
        for row in summary:
            a = float(row["alpha"])
            for cat in categories:
                m = by_alpha_category.get((a, cat))
                if not m:
                    continue
                md_lines.append(
                    f"| {a:.3f} | {cat} | {m['delta_failure_rate']:+.3f} | {m['delta_critical_rate']:+.3f} |"
                )

    markdown = "\n".join(md_lines) + "\n"

    # HTML summary table + bars.
    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Steering Sweep Summary</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1a1d21;}",
        "table{border-collapse:collapse;width:100%;margin-top:12px;}",
        "th,td{border:1px solid #d9dde3;padding:6px 8px;font-size:13px;text-align:right;}",
        "th:first-child,td:first-child{text-align:right;}",
        ".bar-wrap{background:#eef2f7;border-radius:4px;height:12px;overflow:hidden;}",
        ".bar{height:12px;background:#2e7d32;}",
        ".bar.neg{background:#c62828;}",
        ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}",
        "</style></head><body>",
        "<h1>Steering Alpha Sweep Summary</h1>",
        f"<p><b>Probe:</b> <span class='mono'>{probe_name}</span> &nbsp; <b>Scope:</b> <span class='mono'>{scope}</span></p>",
        f"<p><b>Best alpha:</b> <span class='mono'>{best['alpha']:.3f}</span></p>",
        "<table><thead><tr><th>alpha</th><th>n</th><th>base_fail</th><th>steered_fail</th><th>delta_fail</th><th>plot</th></tr></thead><tbody>",
    ]
    for row in summary:
        width = min(100, int(abs(row["delta_failure_rate"]) * 400))
        cls = "bar neg" if row["delta_failure_rate"] > 0 else "bar"
        html_lines.append(
            "<tr>"
            f"<td>{row['alpha']:.3f}</td>"
            f"<td>{int(row['sample_count'])}</td>"
            f"<td>{row['baseline_failure_rate']:.3f}</td>"
            f"<td>{row['steered_failure_rate']:.3f}</td>"
            f"<td>{row['delta_failure_rate']:+.3f}</td>"
            f"<td><div class='bar-wrap'><div class='{cls}' style='width:{width}%'></div></div></td>"
            "</tr>"
        )
    html_lines.append("</tbody></table>")

    if categories:
        html_lines.append("<h2>By Category</h2>")
        html_lines.append("<table><thead><tr><th>alpha</th><th style='text-align:left'>category</th><th>delta_fail</th><th>delta_crit</th></tr></thead><tbody>")
        for row in summary:
            a = float(row["alpha"])
            for cat in categories:
                m = by_alpha_category.get((a, cat))
                if not m:
                    continue
                html_lines.append(
                    "<tr>"
                    f"<td>{a:.3f}</td>"
                    f"<td style='text-align:left'>{cat}</td>"
                    f"<td>{m['delta_failure_rate']:+.3f}</td>"
                    f"<td>{m['delta_critical_rate']:+.3f}</td>"
                    "</tr>"
                )
        html_lines.append("</tbody></table>")

    html_lines.append("</body></html>")
    html = "\n".join(html_lines) + "\n"
    return rows, markdown, html

