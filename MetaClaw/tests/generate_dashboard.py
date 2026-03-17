"""
Generate an HTML dashboard from all DragonClaw test result JSON files.
Produces a self-contained HTML file with embedded Chart.js visualizations.

Usage:
    python tests/generate_dashboard.py
"""

import json
import os
import sys
from datetime import datetime

RECORDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "records")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dashboard.html")

SUITES = [
    ("cascading_hallucination_results.json", "T1-T10: Vulnerability Detection", "vuln"),
    ("cascading_hallucination_advanced_results.json", "T11-T20: Advanced Patch Verification", "adv"),
    ("patch_validation_results.json", "V1-V10: Stress / Red-Team Validation", "val"),
    ("patch_validation_multistep_results.json", "V11-V30: Multi-Step Chain Logic", "ms"),
    ("patch_validation_fuzzing_results.json", "V31-V40: Property-Based Fuzzing", "fuzz"),
    ("mutation_testing_results.json", "M1-M10: Mutation Testing", "mut"),
    ("orchestration_hallucination_results.json", "O1-O15: Multi-Agent Orchestration", "orch"),
    ("teach_recall_100turn_results.json", "O16-O25: 100-Turn Teach→Recall", "teach"),
    ("500turn_stress_results.json", "O26-O30: 500-Turn Stress", "stress"),
    ("5000turn_sensitive_results.json", "O31-O35: 5000-Turn Sensitive Data", "sensitive"),
    ("ollama_live_inference_results.json", "O36-O45: Ollama Live Inference", "ollama"),
    ("multimodel_cascade_results.json", "O46-O50: Multi-Model Cascade & Scaling", "multimodel"),
    ("tier2_hallucination_results.json", "O51-O55: Tier 2 Hallucination Analysis", "tier2"),
    ("pentest_extensions_results.json", "P1-P5: Pen Testing Extensions", "pentest"),
    ("pentest_advanced_results.json", "P6-P10: Advanced Pen Testing", "pentest_adv"),
    ("defense_validation_results.json", "D1-D5: Defense Validation", "defense"),
    ("tier3_defense_aware_results.json", "O56-O60: Tier 3 Defense-Aware", "tier3"),
    ("pentest_defense_evasion_results.json", "P11-P15: Defense Evasion", "pentest_evasion"),
    ("training_loop_corruption_results.json", "O61-O65: Training Loop Corruption", "training_loop"),
    ("conversation_memory_results.json", "O66-O70: Conversation Memory Retrieval", "conv_memory"),
    ("session_chain_results.json", "O71-O75: Session Chain Auto-Spawn", "session_chain"),
    ("session_chain_live_results.json", "O76-O80: Session Chain Live", "session_chain_live"),
]


def load_all():
    data = {}
    for fname, label, key in SUITES:
        path = os.path.join(RECORDS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
            data[key]["_label"] = label
        else:
            print(f"  [WARN] Missing: {fname}")
    return data


def build_html(data):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Aggregate stats
    total_pass = 0
    total_warn = 0
    total_fail = 0
    total_killed = 0
    total_survived = 0
    suite_rows = []
    chart_labels = []
    chart_pass = []
    chart_fail = []
    chart_warn = []

    for fname, label, key in SUITES:
        d = data.get(key)
        if not d:
            continue
        s = d.get("summary", {})
        if key == "mut":
            k = s.get("killed", 0)
            sv = s.get("survived", 0)
            total_killed += k
            total_survived += sv
            p, w, f_ = k, 0, sv
            status = s.get("quality", "?")
        else:
            p = s.get("pass", 0)
            w = s.get("warn", 0)
            f_ = s.get("fail", 0)
            status = s.get("validation_status") or s.get("status") or s.get("risk_level") or "?"
            total_pass += p
            total_warn += w
            total_fail += f_

        elapsed = d.get("elapsed_seconds", 0)
        chart_labels.append(label.split(":")[0])
        chart_pass.append(p)
        chart_fail.append(f_)
        chart_warn.append(w)

        status_color = "#22c55e" if f_ == 0 else ("#eab308" if f_ <= 2 else "#ef4444")
        suite_rows.append(f"""
        <tr>
          <td style="font-weight:600">{label}</td>
          <td style="color:#22c55e;font-weight:700">{p}</td>
          <td style="color:#eab308;font-weight:700">{w}</td>
          <td style="color:#ef4444;font-weight:700">{f_}</td>
          <td style="color:{status_color};font-weight:700">{status}</td>
          <td>{elapsed:.1f}s</td>
        </tr>""")

    total_tests = total_pass + total_warn + total_fail + total_killed + total_survived
    overall_score = ((total_pass + total_killed) / total_tests * 100) if total_tests else 0

    # Build detail sections
    detail_sections = []
    for fname, label, key in SUITES:
        d = data.get(key)
        if not d:
            continue
        results = d.get("results", [])
        if not results:
            continue
        rows = []
        for r in results:
            tid = r.get("test_id") or r.get("mutation_id", "?")
            name = r.get("name", "?")
            if key == "mut":
                v = "KILLED" if r.get("killed") else "SURVIVED"
                vc = "#22c55e" if r.get("killed") else "#ef4444"
            else:
                v = r.get("verdict", "?")
                vc = "#22c55e" if v == "PASS" else ("#ef4444" if v == "FAIL" else "#eab308")
            details = r.get("details", "")
            findings = r.get("findings", [])
            findings_html = "<br>".join(f"&bull; {fn}" for fn in findings[:5])
            metrics = r.get("metrics", {})
            metrics_html = " | ".join(f"<b>{k}</b>={v}" for k, v in list(metrics.items())[:4])
            rows.append(f"""
            <tr>
              <td style="font-weight:600;white-space:nowrap">{tid}</td>
              <td>{name}</td>
              <td style="color:{vc};font-weight:700">{v}</td>
              <td style="font-size:0.85em">{findings_html}</td>
              <td style="font-size:0.8em;color:#888">{metrics_html}</td>
            </tr>""")

        detail_sections.append(f"""
        <div class="section">
          <h2>{label}</h2>
          <table>
            <thead><tr>
              <th>ID</th><th>Test</th><th>Verdict</th><th>Findings</th><th>Metrics</th>
            </tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>""")

    # Fuzzing stats
    fuzz_data = data.get("fuzz", {})
    fuzz_examples = fuzz_data.get("total_random_examples", 0)
    mut_data = data.get("mut", {})
    mut_score = mut_data.get("mutation_score_pct", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DragonClaw v0.3 — Hallucination Defense Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  .header {{ text-align:center; margin-bottom:2rem; }}
  .header h1 {{ font-size:2rem; color:#f1f5f9; margin-bottom:0.25rem; }}
  .header p {{ color:#94a3b8; font-size:0.9rem; }}
  .kpi-row {{ display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-bottom:2rem; }}
  .kpi {{ background:#1e293b; border-radius:12px; padding:1.25rem 2rem; text-align:center; min-width:140px; }}
  .kpi .val {{ font-size:2rem; font-weight:800; }}
  .kpi .lbl {{ font-size:0.8rem; color:#94a3b8; margin-top:0.25rem; }}
  .green {{ color:#22c55e; }} .red {{ color:#ef4444; }} .yellow {{ color:#eab308; }} .blue {{ color:#3b82f6; }} .purple {{ color:#a855f7; }}
  .charts {{ display:flex; gap:1.5rem; justify-content:center; flex-wrap:wrap; margin-bottom:2rem; }}
  .chart-box {{ background:#1e293b; border-radius:12px; padding:1.5rem; width:420px; }}
  .chart-box h3 {{ text-align:center; margin-bottom:1rem; font-size:0.95rem; color:#94a3b8; }}
  .section {{ background:#1e293b; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; }}
  .section h2 {{ font-size:1.1rem; margin-bottom:1rem; color:#f1f5f9; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th {{ text-align:left; padding:0.5rem 0.75rem; border-bottom:2px solid #334155; color:#94a3b8; font-weight:600; }}
  td {{ padding:0.5rem 0.75rem; border-bottom:1px solid #1e293b; }}
  tr:nth-child(even) {{ background:#0f172a55; }}
  .summary-table {{ background:#1e293b; border-radius:12px; padding:1.5rem; margin-bottom:2rem; }}
  .summary-table h2 {{ font-size:1.1rem; margin-bottom:1rem; color:#f1f5f9; }}
  .footer {{ text-align:center; color:#475569; font-size:0.75rem; margin-top:2rem; }}
</style>
</head>
<body>

<div class="header">
  <h1>DragonClaw v0.3 — Hallucination Defense Dashboard</h1>
  <p>Generated {now} &nbsp;|&nbsp; {total_tests} total tests across {len(data)} suites</p>
</div>

<div class="kpi-row">
  <div class="kpi"><div class="val green">{total_pass + total_killed}</div><div class="lbl">Tests Passed / Killed</div></div>
  <div class="kpi"><div class="val yellow">{total_warn}</div><div class="lbl">Warnings</div></div>
  <div class="kpi"><div class="val red">{total_fail + total_survived}</div><div class="lbl">Failures / Survived</div></div>
  <div class="kpi"><div class="val blue">{overall_score:.0f}%</div><div class="lbl">Overall Score</div></div>
  <div class="kpi"><div class="val purple">{fuzz_examples:,}</div><div class="lbl">Fuzz Examples</div></div>
  <div class="kpi"><div class="val green">{mut_score:.0f}%</div><div class="lbl">Mutation Score</div></div>
</div>

<div class="charts">
  <div class="chart-box">
    <h3>Results by Suite</h3>
    <canvas id="barChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Overall Verdict Distribution</h3>
    <canvas id="doughnutChart"></canvas>
  </div>
</div>

<div class="summary-table">
  <h2>Suite Summary</h2>
  <table>
    <thead><tr><th>Suite</th><th>Pass</th><th>Warn</th><th>Fail</th><th>Status</th><th>Time</th></tr></thead>
    <tbody>{"".join(suite_rows)}</tbody>
  </table>
</div>

{"".join(detail_sections)}

<div class="footer">
  DragonClaw Cascading Hallucination Evaluation &bull; v0.3.0 &bull; {total_tests} tests &bull; {now}
</div>

<script>
const barCtx = document.getElementById('barChart').getContext('2d');
new Chart(barCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(chart_labels)},
    datasets: [
      {{ label: 'Pass/Killed', data: {json.dumps(chart_pass)}, backgroundColor: '#22c55e' }},
      {{ label: 'Warn', data: {json.dumps(chart_warn)}, backgroundColor: '#eab308' }},
      {{ label: 'Fail/Survived', data: {json.dumps(chart_fail)}, backgroundColor: '#ef4444' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
    scales: {{
      x: {{ stacked: true, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ stacked: true, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

const doughCtx = document.getElementById('doughnutChart').getContext('2d');
new Chart(doughCtx, {{
  type: 'doughnut',
  data: {{
    labels: ['Pass/Killed', 'Warn', 'Fail/Survived'],
    datasets: [{{ data: [{total_pass + total_killed}, {total_warn}, {total_fail + total_survived}],
                  backgroundColor: ['#22c55e', '#eab308', '#ef4444'] }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
  }}
}});
</script>

</body>
</html>"""
    return html


def main():
    print("  Loading test results...")
    data = load_all()
    print(f"  Loaded {len(data)} suite(s)")

    print("  Generating dashboard...")
    html = build_html(data)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"  Dashboard saved to: {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH):,} bytes")


if __name__ == "__main__":
    main()
