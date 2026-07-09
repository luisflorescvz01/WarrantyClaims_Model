"""
Build an interactive "claim simulator" page for the trained model.

The selected model is logistic regression, so its exact scoring math
(one-hot weights + scaler + engineered features + intercept) can be
exported to JavaScript and run entirely in the browser - no server.

This script:
  1. decomposes the fitted pipeline into per-feature weights,
  2. VERIFIES the decomposition reproduces pipeline.predict_proba on
     every row of the dataset (aborts if not),
  3. writes reports/tester.html and docs/tester.html (GitHub Pages).

Usage:
    python src/build_tester.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features import engineer
from train import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_data

ROOT = Path(__file__).resolve().parents[1]
OUT_PATHS = [ROOT / "reports" / "tester.html", ROOT / "docs" / "tester.html"]

INCIDENT_YEAR = 2015  # all incidents in the dataset occur in 2015

LABELS = {
    "months_as_customer": "Months as customer",
    "age": "Customer age",
    "policy_state": "Policy state",
    "policy_csl": "Policy CSL (liability limits)",
    "policy_deductable": "Policy deductible ($)",
    "policy_annual_premium": "Annual premium ($)",
    "umbrella_limit": "Umbrella limit ($)",
    "insured_sex": "Customer sex",
    "insured_education_level": "Education level",
    "insured_occupation": "Occupation",
    "insured_relationship": "Relationship status",
    "capital-gains": "Capital gains ($)",
    "capital-loss": "Capital loss ($, negative)",
    "incident_type": "Incident type",
    "collision_type": "Collision type",
    "incident_severity": "Incident severity",
    "authorities_contacted": "Authorities contacted",
    "incident_state": "Incident state",
    "incident_city": "Incident city",
    "incident_hour_of_the_day": "Hour of incident (0-23)",
    "number_of_vehicles_involved": "Vehicles involved",
    "property_damage": "Property damage confirmed?",
    "bodily_injuries": "Bodily injuries",
    "witnesses": "Witnesses",
    "police_report_available": "Police report available?",
    "injury_claim": "Injury claim ($)",
    "property_claim": "Property claim ($)",
    "vehicle_claim": "Vehicle claim ($)",
    "auto_make": "Vehicle make",
    "auto_model": "Vehicle model",
    "auto_year": "Vehicle year",
    # engineered / derived (shown in the "why" panel)
    "total_claim_amount": "Total claim amount",
    "vehicle_age": "Vehicle age",
    "tenure_years": "Customer tenure (years)",
    "age_at_signup": "Age when policy started",
    "injury_share": "Injury portion of the claim",
    "property_share": "Property portion of the claim",
    "vehicle_share": "Vehicle portion of the claim",
    "claim_per_premium": "Claim size vs annual premium",
    "claim_vs_deductible": "Claim size vs deductible",
    "net_capital": "Net capital (gains + losses)",
    "night_incident": "Night-time incident",
}

FORM_SECTIONS = [
    ("Customer & policy", [
        "age", "insured_sex", "insured_education_level", "insured_occupation",
        "insured_relationship", "months_as_customer", "policy_state", "policy_csl",
        "policy_deductable", "policy_annual_premium", "umbrella_limit",
        "capital-gains", "capital-loss",
    ]),
    ("Incident", [
        "incident_type", "collision_type", "incident_severity",
        "authorities_contacted", "incident_state", "incident_city",
        "incident_hour_of_the_day", "number_of_vehicles_involved",
        "bodily_injuries", "witnesses", "property_damage",
        "police_report_available",
    ]),
    ("Vehicle & claim amounts", [
        "auto_make", "auto_model", "auto_year",
        "injury_claim", "property_claim", "vehicle_claim",
    ]),
]


def derive(claim: dict) -> tuple[dict, dict]:
    """Python mirror of the JS `derive()` - raw form fields to model inputs.
    Returns (categorical values, numeric values). Must match features.engineer."""
    g = lambda k: float(claim[k])
    total = g("injury_claim") + g("property_claim") + g("vehicle_claim")
    nums = {
        "months_as_customer": g("months_as_customer"),
        "age": g("age"),
        "policy_deductable": g("policy_deductable"),
        "policy_annual_premium": g("policy_annual_premium"),
        "umbrella_limit": g("umbrella_limit"),
        "capital-gains": g("capital-gains"),
        "capital-loss": g("capital-loss"),
        "incident_hour_of_the_day": g("incident_hour_of_the_day"),
        "number_of_vehicles_involved": g("number_of_vehicles_involved"),
        "bodily_injuries": g("bodily_injuries"),
        "witnesses": g("witnesses"),
        "total_claim_amount": total,
        "injury_claim": g("injury_claim"),
        "property_claim": g("property_claim"),
        "vehicle_claim": g("vehicle_claim"),
        "vehicle_age": max(0.0, INCIDENT_YEAR - g("auto_year")),
        "tenure_years": g("months_as_customer") / 12.0,
        "age_at_signup": g("age") - g("months_as_customer") / 12.0,
        "injury_share": g("injury_claim") / total if total else 0.0,
        "property_share": g("property_claim") / total if total else 0.0,
        "vehicle_share": g("vehicle_claim") / total if total else 0.0,
        "claim_per_premium": total / g("policy_annual_premium")
        if g("policy_annual_premium") else 0.0,
        "claim_vs_deductible": total / g("policy_deductable")
        if g("policy_deductable") else 0.0,
        "net_capital": g("capital-gains") + g("capital-loss"),
        "night_incident": 1.0
        if (g("incident_hour_of_the_day") >= 22 or g("incident_hour_of_the_day") <= 5)
        else 0.0,
    }
    cats = {}
    for k, v in claim.items():
        if k in nums or k == "auto_year":
            continue
        v = str(v)
        cats[k] = "UNKNOWN" if v in ("?", "nan", "None", "") else v
    return cats, nums


def export_weights(pipe) -> dict:
    """Decompose OneHot+Scaler+LogisticRegression into flat JS weights."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    ohe = pre.named_transformers_["cat"]
    scaler = pre.named_transformers_["num"]
    cat_cols = list(ohe.feature_names_in_)
    num_cols = list(scaler.feature_names_in_)
    coefs = clf.coef_[0]

    w_cat, i = {}, 0
    for col, cats in zip(cat_cols, ohe.categories_):
        w_cat[col] = {str(c): float(coefs[i + k]) for k, c in enumerate(cats)}
        i += len(cats)
    w_num = {}
    for k, col in enumerate(num_cols):
        c = coefs[i + k]
        w_num[col] = {
            "w": float(c / scaler.scale_[k]),
            "b": float(-c * scaler.mean_[k] / scaler.scale_[k]),
        }
    return {
        "intercept": float(clf.intercept_[0]),
        "cat": w_cat,
        "num": w_num,
        "catCols": cat_cols,
        "numCols": num_cols,
    }


def js_score(weights: dict, cats: dict, nums: dict) -> float:
    """Python mirror of the JS scorer, used for the parity check."""
    z = weights["intercept"]
    for col in weights["catCols"]:
        z += weights["cat"][col].get(cats[col], 0.0)
    for col in weights["numCols"]:
        w = weights["num"][col]
        z += w["w"] * nums[col] + w["b"]
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    pipe, threshold = bundle["pipeline"], bundle["threshold"]
    weights = export_weights(pipe)

    X, y = load_data(DATA_PATH)
    form_cols = [c for cols in [s[1] for s in FORM_SECTIONS] for c in cols]

    # ---- parity check: exported math must reproduce the real pipeline
    proba_pipe = pipe.predict_proba(X)[:, 1]
    proba_js = np.array([
        js_score(weights, *derive(row)) for row in X[form_cols].to_dict("records")
    ])
    max_diff = np.abs(proba_pipe - proba_js).max()
    assert max_diff < 1e-9, f"decomposition mismatch: max diff {max_diff}"
    print(f"Parity check passed: max |pipeline - exported| = {max_diff:.2e} over {len(X)} claims")

    # ---- form metadata: dropdown options, defaults, make->model map
    eng = engineer(X)
    options, defaults = {}, {}
    for col in form_cols:
        if col in weights["cat"]:
            options[col] = sorted(weights["cat"][col].keys())
            defaults[col] = str(eng[col].mode()[0])
        else:
            defaults[col] = float(X[col].median())
    make_models = {
        m: sorted(g["auto_model"].unique().tolist())
        for m, g in X.groupby("auto_make")
    }

    # baseline contribution per categorical column (frequency-weighted mean
    # weight over the training distribution) so "why" bars read as
    # "compared with a typical claim"
    cat_baseline = {
        col: float(sum(
            weights["cat"][col].get(str(v), 0.0) * n / len(eng)
            for v, n in eng[col].value_counts().items()
        ))
        for col in weights["catCols"]
    }

    # ---- example claims from the held-out test set
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    proba_te = pipe.predict_proba(X_te)[:, 1]
    idx_hi = int(np.argmax(np.where(y_te.to_numpy() == 1, proba_te, -1)))
    idx_lo = int(np.argmin(np.where(y_te.to_numpy() == 0, proba_te, 2)))
    idx_mid = int(np.argmin(np.abs(proba_te - threshold)))
    examples = []
    for name, idx in [("High-risk claim", idx_hi), ("Borderline claim", idx_mid),
                      ("Typical valid claim", idx_lo)]:
        row = X_te.iloc[idx]
        vals = {}
        for c in form_cols:
            v = row[c]
            if c in options:
                v = str(v)
                vals[c] = "UNKNOWN" if v in ("?", "nan", "None", "") else v
            else:
                vals[c] = float(v)
        examples.append({"name": name, "values": vals,
                         "expected": round(float(proba_te[idx]), 10)})

    data = {
        "weights": weights,
        "threshold": threshold,
        "options": options,
        "defaults": defaults,
        "makeModels": make_models,
        "catBaseline": cat_baseline,
        "labels": LABELS,
        "examples": examples,
        "incidentYear": INCIDENT_YEAR,
    }

    # ---- build the form HTML server-side
    form_html = ""
    for title, cols in FORM_SECTIONS:
        fields = ""
        for c in cols:
            fid = c.replace("-", "_")
            if c in options:
                opts = "".join(
                    f'<option value="{o}"{" selected" if o == defaults[c] else ""}>{o}</option>'
                    for o in options[c]
                )
                fields += (
                    f'<label>{LABELS[c]}<select id="f_{fid}" data-col="{c}">{opts}</select></label>'
                )
            else:
                d = defaults[c]
                dv = int(d) if d == int(d) else round(d, 2)
                fields += (
                    f'<label>{LABELS[c]}<input id="f_{fid}" data-col="{c}" '
                    f'type="number" step="any" value="{dv}"></label>'
                )
        form_html += f"<fieldset><legend>{title}</legend><div class='grid'>{fields}</div></fieldset>"

    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__FORM__", form_html)
    for p in OUT_PATHS:
        p.parent.mkdir(exist_ok=True)
        p.write_text(html, encoding="utf-8")
        print(f"Tester written -> {p}")


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Warranty Claim Model - Claim Simulator</title>
<style>
  :root { color-scheme: light; }
  .viz-root {
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7;
    --pos: #e34948;   /* pushes score toward invalid */
    --neg: #2a78d6;   /* pushes score toward valid   */
    --status-good: #0ca30c; --status-critical: #d03b3b;
    --border: rgba(11,11,11,0.10);
  }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         margin:0; background:var(--page); color:var(--ink); }
  .wrap { max-width:1100px; margin:0 auto; padding:26px 20px 60px; }
  h1 { font-size:24px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 8px; font-size:14px; }
  .sub a { color:var(--neg); }
  .lede { color:var(--ink-2); font-size:14px; max-width:78ch; margin:0 0 18px; line-height:1.5; }
  .examples { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; align-items:center; }
  .examples span { font-size:13px; color:var(--ink-2); }
  .examples button { border:1px solid var(--border); background:var(--surface-1);
    border-radius:8px; padding:8px 14px; font-size:13.5px; cursor:pointer; color:var(--ink); }
  .examples button:hover { border-color:var(--neg); color:var(--neg); }
  .cols { display:grid; grid-template-columns: 1fr 380px; gap:22px; align-items:start; }
  @media (max-width:900px) { .cols { grid-template-columns:1fr; }
    .result { position:static !important; } }
  fieldset { border:1px solid var(--border); background:var(--surface-1);
    border-radius:10px; margin:0 0 16px; padding:14px 16px 18px; }
  legend { font-weight:600; font-size:14px; padding:0 6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px 16px; }
  label { display:flex; flex-direction:column; font-size:12.5px; color:var(--ink-2); gap:4px; }
  input, select { font:inherit; font-size:13.5px; padding:7px 8px; border:1px solid var(--grid);
    border-radius:6px; background:#fff; color:var(--ink); min-width:0; }
  input:focus, select:focus { outline:2px solid var(--neg); outline-offset:-1px; border-color:transparent; }
  .result { background:var(--surface-1); border:1px solid var(--border); border-radius:10px;
    padding:20px 22px; position:sticky; top:16px; }
  .result h2 { font-size:15px; margin:0 0 2px; color:var(--ink-2); font-weight:600; }
  .hero { font-size:52px; font-weight:700; line-height:1.1; }
  .hero small { font-size:16px; font-weight:400; color:var(--muted); }
  .meter { position:relative; height:14px; background:var(--grid); border-radius:7px;
    margin:14px 0 6px; }
  .meter-fill { position:absolute; inset:0 auto 0 0; border-radius:7px 4px 4px 7px;
    background:var(--neg); min-width:4px; transition:width .25s; }
  .meter-cut { position:absolute; top:-5px; bottom:-5px; width:2px; background:var(--ink-2); }
  .meter-labels { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); }
  .decision { display:inline-flex; align-items:center; gap:8px; font-weight:600; font-size:15px;
    border-radius:8px; padding:8px 14px; margin:14px 0 4px; }
  .decision.approve { background:#e7f4ea; color:#0a5c0a; }
  .decision.flag { background:#fbe9e9; color:#8f2626; }
  .why { margin-top:20px; }
  .why h3 { font-size:13.5px; margin:0 0 2px; color:var(--ink-2); }
  .why p { font-size:12px; color:var(--muted); margin:0 0 12px; }
  .factor { display:grid; grid-template-columns: 128px 1fr; gap:8px; align-items:center;
    margin-bottom:7px; }
  .factor .name { font-size:12px; color:var(--ink-2); text-align:right; line-height:1.25; }
  .factor .track { position:relative; height:14px; }
  .factor .track::before { content:""; position:absolute; left:50%; top:-2px; bottom:-2px;
    width:1px; background:var(--baseline); }
  .factor .bar { position:absolute; top:0; height:14px; }
  .factor .bar.up   { left:50%; background:var(--pos); border-radius:0 4px 4px 0; }
  .factor .bar.down { right:50%; background:var(--neg); border-radius:4px 0 0 4px; }
  .legend { display:flex; gap:16px; font-size:11.5px; color:var(--ink-2); margin-top:10px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
  footer { color:var(--muted); font-size:12px; margin-top:22px; max-width:80ch; line-height:1.5; }
</style>
</head>
<body class="viz-root">
<div class="wrap">
  <h1>Claim Simulator</h1>
  <p class="sub">Test the warranty claim model with your own inputs &middot;
     <a href="index.html">back to the performance dashboard</a></p>
  <p class="lede">Fill in a claim below - or load one of the real examples - and the model
     scores it instantly, exactly as the trained pipeline would. The score updates as you
     change any field, so you can experiment: remove the police report, raise the claim
     amount, change the severity, and watch how the suspicion score responds.</p>

  <div class="examples">
    <span>Load a real claim from the test set:</span>
    <button data-ex="0">High-risk claim</button>
    <button data-ex="1">Borderline claim</button>
    <button data-ex="2">Typical valid claim</button>
    <button data-ex="reset">Reset to averages</button>
  </div>

  <div class="cols">
    <form id="claimform" onsubmit="return false">__FORM__</form>

    <aside class="result">
      <h2>Suspicion score</h2>
      <div class="hero"><span id="score">-</span><small> / 100</small></div>
      <div class="meter">
        <div class="meter-fill" id="fill"></div>
        <div class="meter-cut" id="cut"></div>
      </div>
      <div class="meter-labels"><span>0 - looks fine</span>
        <span id="cutlab"></span><span>100 - very suspicious</span></div>
      <div class="decision" id="decision"></div>
      <div id="totalline" style="font-size:12.5px;color:var(--muted);margin-top:6px"></div>

      <div class="why">
        <h3>What's driving this score</h3>
        <p>Top factors compared with a typical claim. Bars to the right raise suspicion,
           bars to the left lower it.</p>
        <div id="factors"></div>
        <div class="legend">
          <span><span class="dot" style="background:var(--pos)"></span>raises suspicion</span>
          <span><span class="dot" style="background:var(--neg)"></span>lowers suspicion</span>
        </div>
      </div>
    </aside>
  </div>

  <footer>This page runs the trained model's exact mathematics (logistic regression
    coefficients exported from the saved pipeline) in your browser - scores match the
    Python model to more than 9 decimal places. It is a prototype trained on public
    auto-insurance data standing in for warranty data; scores are illustrative, not
    adjudications. Factor impacts are log-odds contributions relative to the average
    claim in the training data.</footer>
</div>

<script>
const D = __DATA__;
const W = D.weights;

function val(col) {
  const el = document.querySelector(`[data-col="${col}"]`);
  return el.tagName === "SELECT" ? el.value : parseFloat(el.value) || 0;
}

function derive() {
  const c = {};
  document.querySelectorAll("[data-col]").forEach(el => c[el.dataset.col] = val(el.dataset.col));
  const total = c.injury_claim + c.property_claim + c.vehicle_claim;
  const nums = {
    months_as_customer: c.months_as_customer, age: c.age,
    policy_deductable: c.policy_deductable, policy_annual_premium: c.policy_annual_premium,
    umbrella_limit: c.umbrella_limit, "capital-gains": c["capital-gains"],
    "capital-loss": c["capital-loss"], incident_hour_of_the_day: c.incident_hour_of_the_day,
    number_of_vehicles_involved: c.number_of_vehicles_involved,
    bodily_injuries: c.bodily_injuries, witnesses: c.witnesses,
    total_claim_amount: total, injury_claim: c.injury_claim,
    property_claim: c.property_claim, vehicle_claim: c.vehicle_claim,
    vehicle_age: Math.max(0, D.incidentYear - c.auto_year),
    tenure_years: c.months_as_customer / 12,
    age_at_signup: c.age - c.months_as_customer / 12,
    injury_share: total ? c.injury_claim / total : 0,
    property_share: total ? c.property_claim / total : 0,
    vehicle_share: total ? c.vehicle_claim / total : 0,
    claim_per_premium: c.policy_annual_premium ? total / c.policy_annual_premium : 0,
    claim_vs_deductible: c.policy_deductable ? total / c.policy_deductable : 0,
    "net_capital": c["capital-gains"] + c["capital-loss"],
    night_incident: (c.incident_hour_of_the_day >= 22 || c.incident_hour_of_the_day <= 5) ? 1 : 0,
  };
  const cats = {};
  for (const col of W.catCols) cats[col] = String(c[col]);
  return { cats, nums, total };
}

function scoreClaim() {                       // exact mirror of the sklearn pipeline
  const { cats, nums, total } = derive();
  let z = W.intercept;
  const contrib = [];
  for (const col of W.catCols) {
    const w = W.cat[col][cats[col]] ?? 0;
    z += w;
    contrib.push([col, w - D.catBaseline[col], cats[col]]);
  }
  for (const col of W.numCols) {
    const w = W.num[col];
    const v = w.w * nums[col] + w.b;          // centered: 0 = training average
    z += v;
    contrib.push([col, v, null]);
  }
  return { p: 1 / (1 + Math.exp(-z)), contrib, total };
}

function fmt(x) { return x.toLocaleString("en-US", {maximumFractionDigits: 0}); }

function render() {
  const { p, contrib, total } = scoreClaim();
  document.getElementById("score").textContent = (p * 100).toFixed(1);
  document.getElementById("fill").style.width = (p * 100) + "%";
  const flag = p >= D.threshold;
  const d = document.getElementById("decision");
  d.className = "decision " + (flag ? "flag" : "approve");
  d.innerHTML = flag ? "&#9873; Flag for review" : "&#10003; Auto-approve";
  document.getElementById("totalline").textContent =
    "Total claim amount: $" + fmt(total) +
    (flag ? " - held for an adjuster before payout" : " - would be paid without review");

  contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const top = contrib.slice(0, 7);
  const max = Math.max(...top.map(f => Math.abs(f[1])), 1e-9);
  document.getElementById("factors").innerHTML = top.map(([col, v, catVal]) => {
    const wpct = Math.abs(v) / max * 48;
    const label = D.labels[col] + (catVal !== null ? ": " + catVal : "");
    return `<div class="factor" title="log-odds impact ${v >= 0 ? "+" : ""}${v.toFixed(2)}">
      <div class="name">${label}</div>
      <div class="track"><div class="bar ${v >= 0 ? "up" : "down"}"
           style="width:${wpct.toFixed(1)}%"></div></div></div>`;
  }).join("");
}

function setValues(vals) {
  for (const [col, v] of Object.entries(vals)) {
    const el = document.querySelector(`[data-col="${col}"]`);
    if (!el) continue;
    if (col === "auto_make") { el.value = v; syncModels(); }
    else el.value = v;
  }
  render();
}

function syncModels() {
  const make = val("auto_make");
  const sel = document.querySelector('[data-col="auto_model"]');
  const cur = sel.value;
  const models = D.makeModels[make] || D.options.auto_model;
  sel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join("");
  if (models.includes(cur)) sel.value = cur;
}

document.getElementById("claimform").addEventListener("input", e => {
  if (e.target.dataset.col === "auto_make") syncModels();
  render();
});
document.querySelectorAll(".examples button").forEach(b =>
  b.addEventListener("click", () => {
    if (b.dataset.ex === "reset") setValues(D.defaults);
    else setValues(D.examples[+b.dataset.ex].values);
  }));

const cutPct = D.threshold * 100;
document.getElementById("cut").style.left = cutPct + "%";
document.getElementById("cutlab").textContent =
  "review if \\u2265 " + cutPct.toFixed(0);
syncModels();
setValues(D.defaults);
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
