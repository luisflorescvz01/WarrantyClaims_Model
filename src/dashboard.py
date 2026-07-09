"""
Generate a self-contained HTML dashboard of the model's performance,
written in plain language for non-technical readers.

Usage:
    python src/dashboard.py          # writes reports/dashboard.html and opens it
    python src/dashboard.py --no-open

The dashboard recomputes every number from the saved model and the same
held-out test split used by train.py, so it always matches the current
models/best_model.joblib. Pure HTML/CSS/SVG output - no internet, no
extra libraries needed to view it.
"""

import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from train import (
    DATA_PATH,
    INVESTIGATION_COST,
    MODEL_PATH,
    RANDOM_STATE,
    REVIEW_CAPACITY,
    TEST_SIZE,
    load_data,
    review_cost,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "reports" / "dashboard.html"
# Copy served by GitHub Pages (Settings -> Pages -> main, /docs folder)
DOCS_PATH = ROOT / "docs" / "index.html"

GREEN = "#1a9850"
RED = "#d73027"
AMBER = "#f58518"
BLUE = "#2c7fb8"
GREY = "#8a8f98"


def money(x: float) -> str:
    return f"${x:,.0f}"


def svg_line_chart(points, width, height, pad, color, marker_x=None, marker_label=""):
    """Simple SVG polyline for (x, y) data scaled into the plot area."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    yr = (y1 - y0) or 1.0
    xr = (x1 - x0) or 1.0

    def sx(v):
        return pad + (v - x0) / xr * (width - 2 * pad)

    def sy(v):
        return height - pad - (v - y0) / yr * (height - 2 * pad)

    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    marker = ""
    if marker_x is not None:
        mx = sx(marker_x)
        marker = (
            f'<line x1="{mx:.1f}" y1="{pad}" x2="{mx:.1f}" y2="{height - pad}" '
            f'stroke="{AMBER}" stroke-width="2" stroke-dasharray="5,4"/>'
            f'<text x="{mx + 6:.1f}" y="{pad + 14}" font-size="12" fill="{AMBER}">'
            f"{marker_label}</text>"
        )
    return pts, marker, sx, sy


def build_html(d: dict) -> str:
    """Render the dashboard from the computed metrics dict."""
    # ---- cost comparison bars
    max_cost = d["cost_no_model"]
    bars = ""
    for label, cost, color, note in [
        ("Without the model (pay every claim)", d["cost_no_model"], RED,
         "every invalid claim gets paid in full"),
        ("With the model", d["cost_tuned"], GREEN,
         f"review the {d['flag_rate']:.0%} most suspicious claims "
         f"at {money(INVESTIGATION_COST)} each"),
    ]:
        pct = cost / max_cost * 100
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{label}<span class="bar-note">{note}</span></div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color}">
              <span>{money(cost)}</span>
            </div>
          </div>
        </div>"""

    # ---- confusion matrix as plain-language 2x2
    tn, fp, fn, tp = d["tn"], d["fp"], d["fn"], d["tp"]
    cm = f"""
    <div class="cm-grid">
      <div class="cm-cell good"><b>{tn}</b>Valid claims correctly approved</div>
      <div class="cm-cell warn"><b>{fp}</b>Valid claims sent to review<br>
        <small>(cleared by an adjuster, costs {money(INVESTIGATION_COST)} each)</small></div>
      <div class="cm-cell bad"><b>{fn}</b>Invalid claims that slipped through<br>
        <small>(paid out - this is the remaining loss)</small></div>
      <div class="cm-cell good"><b>{tp}</b>Invalid claims caught in review</div>
    </div>"""

    # ---- score histogram (stacked per bin: valid vs invalid)
    hist = ""
    max_count = max(max(v + f for v, f in zip(d["hist_valid"], d["hist_fraud"])), 1)
    for i, (v, f) in enumerate(zip(d["hist_valid"], d["hist_fraud"])):
        vh = v / max_count * 160
        fh = f / max_count * 160
        lo = i / len(d["hist_valid"])
        hist += f"""
        <div class="hbin" title="score {lo:.1f}-{lo + 0.1:.1f}: {v} valid, {f} invalid">
          <div class="hstack">
            <div style="height:{fh:.0f}px;background:{RED}"></div>
            <div style="height:{vh:.0f}px;background:{GREEN}"></div>
          </div>
          <span>{lo:.1f}</span>
        </div>"""
    thr_left = d["threshold"] * 100

    # ---- cost vs threshold curve (SVG)
    W, H, PAD = 640, 240, 42
    pts, marker, sx, sy = svg_line_chart(
        d["cost_curve"], W, H, PAD, BLUE,
        marker_x=d["threshold"], marker_label=f"chosen cut-off {d['threshold']:.2f}",
    )
    ymin = min(c for _, c in d["cost_curve"])
    ymax = max(c for _, c in d["cost_curve"])
    cost_svg = f"""
    <svg viewBox="0 0 {W} {H}" role="img">
      <line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" stroke="{GREY}"/>
      <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}" stroke="{GREY}"/>
      <text x="{PAD - 6}" y="{sy(ymax) + 4:.0f}" font-size="11" fill="{GREY}" text-anchor="end">{money(ymax)}</text>
      <text x="{PAD - 6}" y="{sy(ymin) + 4:.0f}" font-size="11" fill="{GREY}" text-anchor="end">{money(ymin)}</text>
      <text x="{sx(0.1):.0f}" y="{H - PAD + 16}" font-size="11" fill="{GREY}">lenient (0.1)</text>
      <text x="{sx(0.9):.0f}" y="{H - PAD + 16}" font-size="11" fill="{GREY}" text-anchor="end">strict (0.9)</text>
      <polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>
      {marker}
    </svg>"""

    # ---- ROC curve (SVG)
    W2, H2, PAD2 = 300, 300, 36
    rpts, _, rsx, rsy = svg_line_chart(d["roc_points"], W2, H2, PAD2, BLUE)
    roc_svg = f"""
    <svg viewBox="0 0 {W2} {H2}" role="img">
      <line x1="{PAD2}" y1="{H2 - PAD2}" x2="{W2 - PAD2}" y2="{H2 - PAD2}" stroke="{GREY}"/>
      <line x1="{PAD2}" y1="{PAD2}" x2="{PAD2}" y2="{H2 - PAD2}" stroke="{GREY}"/>
      <line x1="{PAD2}" y1="{H2 - PAD2}" x2="{W2 - PAD2}" y2="{PAD2}" stroke="{GREY}" stroke-dasharray="4,4"/>
      <polyline points="{rpts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>
      <text x="{W2 / 2}" y="{H2 - 8}" font-size="11" fill="{GREY}" text-anchor="middle">% of valid claims wrongly flagged</text>
      <text x="12" y="{H2 / 2}" font-size="11" fill="{GREY}" text-anchor="middle" transform="rotate(-90 12 {H2 / 2})">% of invalid claims caught</text>
    </svg>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Warranty Claim Model - Performance Dashboard</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0; background:#f4f6f8; color:#1c2430; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 28px 20px 60px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  .sub {{ color:{GREY}; margin: 0 0 26px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-bottom:30px; }}
  .card {{ background:#fff; border-radius:10px; padding:18px 20px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .card .big {{ font-size:30px; font-weight:700; }}
  .card .cap {{ color:{GREY}; font-size:13px; margin-top:4px; }}
  .green {{ color:{GREEN}; }} .red {{ color:{RED}; }} .blue {{ color:{BLUE}; }}
  section {{ background:#fff; border-radius:10px; padding:22px 24px; box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:22px; }}
  section h2 {{ font-size:18px; margin:0 0 6px; }}
  .try-btn {{ display:block; background:{BLUE}; color:#fff; text-decoration:none;
    border-radius:10px; padding:14px 20px; font-weight:600; font-size:16px;
    margin-bottom:22px; box-shadow:0 1px 3px rgba(0,0,0,.15); }}
  .try-btn:hover {{ background:#1c5cab; }}
  .try-btn span {{ display:block; font-weight:400; font-size:13px; opacity:.85; margin-top:2px; }}
  .intro {{ border-left:4px solid {BLUE}; }}
  .intro-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 28px; }}
  .intro-grid h3 {{ font-size:14.5px; margin:12px 0 4px; color:{BLUE}; }}
  .intro-grid p {{ color:#4a5361; font-size:13.5px; line-height:1.5; margin:0 0 8px; }}
  @media (max-width:700px) {{ .intro-grid {{ grid-template-columns:1fr; }} }}
  section p.explain {{ color:#4a5361; font-size:14px; margin:0 0 18px; max-width:70ch; }}
  .bar-row {{ display:flex; align-items:center; gap:14px; margin:12px 0; }}
  .bar-label {{ flex:0 0 300px; font-size:14px; }}
  .bar-note {{ display:block; color:{GREY}; font-size:12px; }}
  .bar-track {{ flex:1; background:#eceff3; border-radius:6px; height:38px; }}
  .bar-fill {{ height:100%; border-radius:6px; display:flex; align-items:center; min-width:110px; }}
  .bar-fill span {{ color:#fff; font-weight:600; padding-left:12px; font-size:14px; }}
  .cm-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; max-width:640px; }}
  .cm-cell {{ border-radius:8px; padding:14px 16px; font-size:13.5px; }}
  .cm-cell b {{ display:block; font-size:24px; margin-bottom:2px; }}
  .cm-cell.good {{ background:#e7f4ea; color:#14532d; }}
  .cm-cell.warn {{ background:#fdf3e0; color:#7a4b0f; }}
  .cm-cell.bad  {{ background:#fdeaea; color:#7f1d1d; }}
  .histo {{ display:flex; align-items:flex-end; gap:6px; height:200px; position:relative; padding-top:10px; }}
  .hbin {{ flex:1; text-align:center; font-size:11px; color:{GREY}; }}
  .hstack {{ display:flex; flex-direction:column-reverse; justify-content:flex-start; }}
  .hstack div {{ border-radius:3px 3px 0 0; }}
  .thr-line {{ position:absolute; top:0; bottom:18px; width:2px; background:{AMBER}; }}
  .thr-line span {{ position:absolute; top:-4px; left:6px; font-size:12px; color:{AMBER}; white-space:nowrap; }}
  .legend {{ font-size:13px; color:#4a5361; margin-top:8px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }}
  .flex {{ display:flex; gap:26px; flex-wrap:wrap; align-items:center; }}
  footer {{ color:{GREY}; font-size:12px; margin-top:8px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Warranty Claim Model - Performance Dashboard</h1>
  <p class="sub">How the model performs on {d["n_test"]} claims it has never seen before
     (model: {d["model_name"]}, generated from models/best_model.joblib)</p>

  <a class="try-btn" href="tester.html">Try the model yourself &rarr;
    <span>enter a claim in the interactive simulator and watch it get scored live</span></a>

  <section class="intro">
    <h2>What is this?</h2>
    <div class="intro-grid">
      <div>
        <h3>What the model does</h3>
        <p>It reads each incoming vehicle claim and gives it a <b>suspicion score</b>
           between 0 and 1. Claims scoring below the cut-off ({d["threshold"]:.2f}) are
           approved automatically; claims at or above it are <b>flagged for a human
           adjuster to review</b> before any money is paid. The model never rejects a
           claim on its own - it only decides who gets a second look.</p>
      </div>
      <div>
        <h3>How it does it</h3>
        <p>The model learned from {d["n_train"]} past claims whose outcome (valid or
           invalid) is already known. It looks at patterns across the whole claim: the
           size and make-up of the amounts claimed, how the claim compares to the policy's
           premium and deductible, vehicle age, customer tenure, incident circumstances,
           and whether supporting evidence (like a police report) was provided. No single
           detail decides - it is the combination that raises or lowers the score.</p>
      </div>
      <div>
        <h3>What is being evaluated here</h3>
        <p>Everything on this page is measured on <b>{d["n_test"]} claims the model never
           saw during training</b> - a held-back sample that simulates brand-new claims
           arriving tomorrow. That makes these numbers an honest preview of real-world
           performance, not a memory test.</p>
      </div>
      <div>
        <h3>Why these metrics matter</h3>
        <p>Two mistakes cost money in opposite ways: <b>missing an invalid claim</b> means
           paying it in full, while <b>flagging a valid one</b> wastes
           {money(INVESTIGATION_COST)} of adjuster time and delays an honest customer.
           The metrics below track both sides - how much fraud is caught, how much review
           effort that takes, and the net dollar result - because a model is only useful
           if the savings outweigh the review cost and the workload stays within the team's
           {REVIEW_CAPACITY:.0%} capacity.</p>
      </div>
    </div>
  </section>

  <div class="cards">
    <div class="card"><div class="big green">{money(d["savings"])}</div>
      <div class="cap">estimated savings vs paying every claim ({d["savings_pct"]:.0%} lower cost)</div></div>
    <div class="card"><div class="big blue">{d["recall"]:.0%}</div>
      <div class="cap">of invalid claims caught before payout</div></div>
    <div class="card"><div class="big">{d["flag_rate"]:.0%}</div>
      <div class="cap">of claims sent to human review (capacity limit: {REVIEW_CAPACITY:.0%})</div></div>
    <div class="card"><div class="big">{d["roc_auc"]:.2f}</div>
      <div class="cap">ranking quality (ROC-AUC, 0.5 = coin flip, 1.0 = perfect)</div></div>
  </div>

  <section>
    <h2>The bottom line: what this model saves</h2>
    <p class="explain">If every claim were simply paid, invalid claims would cost the full
       amount below. With the model, suspicious claims go to a human adjuster first
       ({money(INVESTIGATION_COST)} per review) and most invalid payouts are avoided.</p>
    {bars}
  </section>

  <section>
    <h2>What happened to each of the {d["n_test"]} test claims</h2>
    <p class="explain">Each claim is either approved automatically or flagged for review.
       Green boxes are correct decisions; the amber box is review effort spent on
       claims that turned out fine; the red box is the fraud that still gets paid.</p>
    {cm}
  </section>

  <section>
    <h2>How the model separates valid from invalid claims</h2>
    <p class="explain">Every claim gets a suspicion score from 0 (looks fine) to 1 (very
       suspicious). Green bars are valid claims, red bars are invalid ones. Claims to the
       right of the orange line are sent to review. Perfect separation would put all red
       to the right and all green to the left - real data always overlaps some.</p>
    <div class="histo">
      <div class="thr-line" style="left:{thr_left:.1f}%"><span>review if score &ge; {d["threshold"]:.2f}</span></div>
      {hist}
    </div>
    <div class="legend">
      <span class="dot" style="background:{GREEN}"></span>valid claims&nbsp;&nbsp;
      <span class="dot" style="background:{RED}"></span>invalid claims
    </div>
  </section>

  <section>
    <h2>Why the cut-off is set at {d["threshold"]:.2f}</h2>
    <p class="explain">Moving the cut-off changes total cost: too lenient wastes money on
       unnecessary reviews, too strict lets fraud through. The line shows total cost at
       every possible cut-off on the test claims; the orange marker is the value chosen
       during training (within the {REVIEW_CAPACITY:.0%} review-capacity limit).</p>
    {cost_svg}
  </section>

  <section>
    <h2>Ranking quality (for the technically curious)</h2>
    <div class="flex">
      {roc_svg}
      <p class="explain" style="flex:1;min-width:240px">This ROC curve shows the trade-off
         between catching invalid claims and wrongly flagging valid ones, across all
         possible cut-offs. The further the curve bows above the dashed diagonal
         (random guessing), the better. Area under the curve: <b>{d["roc_auc"]:.3f}</b>;
         precision-recall AUC: <b>{d["pr_auc"]:.3f}</b> vs a {d["base_rate"]:.2f} baseline.</p>
    </div>
  </section>

  <footer>Generated by src/dashboard.py from the saved model. Figures use a held-out test
    set never seen during training. Assumes review always catches flagged invalid claims;
    investigation cost {money(INVESTIGATION_COST)} per claim.</footer>
</div>
</body>
</html>"""


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    pipe, threshold = bundle["pipeline"], bundle["threshold"]

    X, y = load_data(DATA_PATH)
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    y_arr = y_te.to_numpy()
    amounts = X_te["total_claim_amount"].to_numpy()
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred).ravel()

    fpr, tpr, _ = roc_curve(y_arr, proba)
    edges = np.arange(0, 1.05, 0.1)
    d = {
        "model_name": type(pipe.named_steps["clf"]).__name__,
        "n_test": len(y_te),
        "n_train": len(y) - len(y_te),
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_arr, proba),
        "pr_auc": average_precision_score(y_arr, proba),
        "base_rate": y_arr.mean(),
        "recall": tp / (tp + fn),
        "flag_rate": pred.mean(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost_no_model": amounts[y_arr == 1].sum(),
        "cost_tuned": review_cost(y_arr, proba, amounts, threshold),
        "hist_valid": np.histogram(proba[y_arr == 0], bins=edges)[0].tolist(),
        "hist_fraud": np.histogram(proba[y_arr == 1], bins=edges)[0].tolist(),
        "cost_curve": [
            (t, review_cost(y_arr, proba, amounts, t))
            for t in np.arange(0.05, 0.96, 0.01)
        ],
        "roc_points": list(zip(fpr.tolist(), tpr.tolist())),
    }
    d["savings"] = d["cost_no_model"] - d["cost_tuned"]
    d["savings_pct"] = d["savings"] / d["cost_no_model"]

    html = build_html(d)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written -> {OUT_PATH}")
    DOCS_PATH.parent.mkdir(exist_ok=True)
    DOCS_PATH.write_text(html, encoding="utf-8")
    print(f"GitHub Pages copy -> {DOCS_PATH}")

    if "--no-open" not in sys.argv:
        webbrowser.open(OUT_PATH.as_uri())


if __name__ == "__main__":
    main()
