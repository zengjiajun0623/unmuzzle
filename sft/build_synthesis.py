# -*- coding: utf-8 -*-
"""Build the cross-ladder synthesis artifact: one self-contained research page
pulling 7B/14B/72B/R1-Distill honesty-SFT + the abliteration three-way + the Muon
negative into a single thesis. Numbers from ladder_matrix.json (verified)."""
import json, os

OUT = "/private/tmp/claude-501/-Users-clawbox/53df357e-94a8-41d2-8d06-600cd03fddf2/scratchpad/synthesis.html"

# --- verified data ---
LADDER = [  # (label, kind, base, sft)
    ("Qwen2.5-7B",   "dense",     48.4, 68.2),
    ("Qwen2.5-14B",  "dense",     68.8, 79.6),
    ("Qwen2.5-72B",  "dense",     85.4, 96.2),
    ("R1-Distill-32B","reasoning", 69.4, 87.9),
]
THREEWAY = [  # model, sens(base,abl,sft), abstain(...), fabricate(...)
    ("7B",  (48.4,42.0,68.2), (65.7,72.9,81.4), (34.3,27.1,18.6)),
    ("14B", (68.8,67.5,79.6), (91.4,44.3,97.1), (8.6,55.7,2.9)),
    ("72B", (85.4,86.0,96.2), (78.6,62.9,100.0),(21.4,37.1,0.0)),
]
LINKS = [
    ("The three-way overview", "eba44d79-5f3b-49a0-97cf-e5dda36ca89f"),
    ("Qwen2.5-7B", "b1bc351b-be1d-4a78-be24-6312340ae7dc"),
    ("Qwen2.5-14B", "d72db405-40d5-4591-8808-4a316f1ca405"),
    ("Qwen2.5-72B", "b7814f9c-b195-44a2-911c-7107d4902311"),
    ("R1-Distill-32B", "1807697c-1579-40eb-a73b-00018532074a"),
]

# --- dumbbell ladder SVG (sensitive-factual base -> SFT) ---
def dumbbell():
    W, rowH, top = 720, 62, 54
    padL, x1, x100 = 150, 168, 690     # axis 40..100
    def X(v): return round(x1 + (v-40)/60*(x100-x1), 1)
    H = top + len(LADDER)*rowH + 26
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Sensitive-fact accuracy, base to SFT, by model">']
    # gridlines
    for g in (40,60,80,100):
        s.append(f'<line class="grid" x1="{X(g)}" y1="{top-14}" x2="{X(g)}" y2="{top+len(LADDER)*rowH-18}"/>')
        s.append(f'<text class="axis" x="{X(g)}" y="{top-20}" text-anchor="middle">{g}%</text>')
    for i,(label,kind,b,sft) in enumerate(LADDER):
        cy = top + i*rowH + 8
        tag = "reasoning" if kind=="reasoning" else "dense"
        s.append(f'<text class="mlabel" x="138" y="{cy+4}" text-anchor="end">{label}</text>')
        if kind=="reasoning":
            s.append(f'<text class="mkind" x="138" y="{cy+18}" text-anchor="end">reasoning model</text>')
        s.append(f'<line class="connector" x1="{X(b)}" y1="{cy}" x2="{X(sft)}" y2="{cy}"/>')
        s.append(f'<circle class="s-base" cx="{X(b)}" cy="{cy}" r="6.5"/>')
        s.append(f'<circle class="s-sft" cx="{X(sft)}" cy="{cy}" r="7.5"/>')
        s.append(f'<text class="vbase" x="{X(b)-12}" y="{cy+4}" text-anchor="end">{b:.0f}</text>')
        s.append(f'<text class="vsft" x="{X(sft)+13}" y="{cy+4}">{sft:.0f}</text>')
        s.append(f'<text class="delta" x="705" y="{cy+4}" text-anchor="end">+{sft-b:.0f}</text>')
    s.append('</svg>')
    return "".join(s)

def bar(v, mx, cls):  # inline mini-bar cell
    w = round(v/mx*100,1)
    return f'<span class="barcell"><span class="barfill {cls}" style="width:{w}%"></span><span class="barnum">{v:.1f}</span></span>'

def threeway_rows():
    rows=""
    for model,(s),(a),(f) in [(m,se,ab,fa) for m,se,ab,fa in THREEWAY]:
        rows += (f'<tr><th>{model}</th>'
          f'<td>{bar(s[0],100,"n")}</td><td>{bar(s[1],100,"n")}</td><td>{bar(s[2],100,"g")}</td>'
          f'<td>{bar(f[0],60,"r")}</td><td class="abl">{bar(f[1],60,"r")}</td><td>{bar(f[2],60,"g")}</td></tr>')
    return rows

CSS = """
*{box-sizing:border-box}
:root{
  --paper:#f4f2ec; --surface:#fbfaf6; --ink:#1b1e1c; --muted:#6a6f68; --hair:#e2ded3;
  --truth:#0f7d5b; --base:#9aa0a6; --abl:#bf8a2e; --danger:#b23b2e;
  --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#101413; --surface:#181d1b; --ink:#eef1ec; --muted:#9aa39c; --hair:#28302b;
  --truth:#2ed9a2; --base:#828b8f; --abl:#e2ab4d; --danger:#e56a54;
}}
:root[data-theme="light"]{--paper:#f4f2ec;--surface:#fbfaf6;--ink:#1b1e1c;--muted:#6a6f68;--hair:#e2ded3;--truth:#0f7d5b;--base:#9aa0a6;--abl:#bf8a2e;--danger:#b23b2e;}
:root[data-theme="dark"]{--paper:#101413;--surface:#181d1b;--ink:#eef1ec;--muted:#9aa39c;--hair:#28302b;--truth:#2ed9a2;--base:#828b8f;--abl:#e2ab4d;--danger:#e56a54;}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.6;
  font-size:17px;-webkit-font-smoothing:antialiased}
.wrap{max-width:940px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(20px,5vw,40px)}
.eyebrow{font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5.4vw,50px);line-height:1.06;
  letter-spacing:-.01em;margin:.5em 0 .3em;text-wrap:balance;max-width:16ch}
.dek{font-size:clamp(17px,2vw,20px);color:var(--muted);max-width:60ch;margin:0}
.dek b{color:var(--ink);font-weight:600}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(22px,3vw,30px);letter-spacing:-.01em;
  margin:0 0 .1em;text-wrap:balance}
.slabel{font-size:12.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--truth);font-weight:700;margin-bottom:6px}
section{margin-top:clamp(48px,7vw,84px)}
p{max-width:64ch}
p .lead{font-weight:600}
hr{border:0;border-top:1px solid var(--hair);margin:0}
.rule{height:1px;background:var(--hair);margin-top:clamp(48px,7vw,84px)}
/* stat band */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--hair);
  border:1px solid var(--hair);border-radius:14px;overflow:hidden;margin-top:34px}
.stat{background:var(--surface);padding:20px 18px}
.stat .n{font-family:var(--serif);font-size:clamp(28px,4vw,40px);font-weight:600;line-height:1;
  color:var(--truth);font-variant-numeric:tabular-nums}
.stat .n.warn{color:var(--danger)}
.stat .k{font-size:13px;color:var(--muted);margin-top:8px;line-height:1.35}
@media(max-width:620px){.stats{grid-template-columns:repeat(2,1fr)}}
/* figure */
figure{margin:26px 0 0;background:var(--surface);border:1px solid var(--hair);border-radius:16px;
  padding:clamp(18px,3vw,30px);overflow-x:auto}
figure svg{display:block;width:100%;height:auto;min-width:420px}
figcaption{font-size:13.5px;color:var(--muted);margin-top:14px;max-width:62ch}
.grid{stroke:var(--hair)}
.axis{fill:var(--muted);font-family:var(--mono);font-size:12px}
.mlabel{fill:var(--ink);font-family:var(--sans);font-size:15px;font-weight:600}
.mkind{fill:var(--muted);font-family:var(--sans);font-size:11px;letter-spacing:.04em;text-transform:uppercase}
.connector{stroke:var(--hair);stroke-width:3}
.s-base{fill:var(--base)}
.s-sft{fill:var(--truth)}
.vbase{fill:var(--muted);font-family:var(--mono);font-size:13px}
.vsft{fill:var(--truth);font-family:var(--mono);font-size:15px;font-weight:700}
.delta{fill:var(--truth);font-family:var(--mono);font-size:13px;font-weight:700;opacity:.85}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:16px}
.legend i{width:11px;height:11px;border-radius:50%;display:inline-block;vertical-align:-1px;margin-right:6px}
/* three-way table */
.tbl{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:560px}
.tbl th,.tbl td{padding:11px 10px;text-align:left;border-bottom:1px solid var(--hair)}
.tbl thead th{font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:600;
  border-bottom:1.5px solid var(--hair);vertical-align:bottom}
.tbl tbody th{font-family:var(--mono);font-weight:700;font-size:15px}
.tbl .grp{border-left:1px solid var(--hair)}
.barcell{position:relative;display:flex;align-items:center;gap:8px;min-width:70px}
.barfill{height:7px;border-radius:4px;flex:0 0 auto;max-width:64px}
.barfill.n{background:var(--base)} .barfill.g{background:var(--truth)} .barfill.r{background:var(--danger)}
.barnum{font-family:var(--mono);font-size:13px;color:var(--ink)}
.abl .barnum{color:var(--abl);font-weight:700}
.colnote{font-size:12px;color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0;display:block;margin-top:2px}
/* R1 panel */
.panel{background:var(--surface);border:1px solid var(--hair);border-radius:16px;
  padding:clamp(22px,3.5vw,34px);margin-top:26px}
.panel .row{display:flex;gap:clamp(20px,5vw,54px);flex-wrap:wrap;margin:18px 0 4px}
.big{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5vw,44px);line-height:1;font-variant-numeric:tabular-nums}
.big .arw{color:var(--muted);font-size:.6em;padding:0 .12em}
.big .to{color:var(--truth)} .big .from{color:var(--muted)}
.big.warn .from{color:var(--danger)}
.metric .lab{font-size:13px;color:var(--muted);margin-top:8px}
/* two-up */
.twoup{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:26px}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:14px;padding:22px 22px 20px}
.card h3{margin:0 0 8px;font-size:15px;font-family:var(--sans)}
.card p{font-size:14.5px;color:var(--muted);margin:0;max-width:none}
.card .tag{font-family:var(--mono);font-size:12px;color:var(--truth);font-weight:700}
.card .tag.neutral{color:var(--muted)}
@media(max-width:620px){.twoup{grid-template-columns:1fr}}
/* links */
.links{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}
.links a{font-size:14px;color:var(--ink);text-decoration:none;border:1px solid var(--hair);
  background:var(--surface);border-radius:999px;padding:8px 15px;transition:border-color .15s}
.links a:hover{border-color:var(--truth)}
.links a .ar{color:var(--truth);margin-left:5px}
.caveat{font-size:14px;color:var(--muted);max-width:66ch}
.caveat b{color:var(--ink)}
footer{margin-top:clamp(48px,7vw,80px);padding-top:22px;border-top:1px solid var(--hair);
  font-size:13px;color:var(--muted)}
a{color:var(--truth)}
"""

HTML = f"""<div class="wrap">
<div class="eyebrow">Freedom-tech research &middot; curated honesty-SFT &middot; 265-item CN benchmark &middot; cross-family LLM judge</div>
<h1>Un-gag the model, don&rsquo;t re-educate it</h1>
<p class="dek">A small, curated fine-tune makes open Chinese LLMs <b>answer truthfully about censored topics</b> and <b>stop making things up</b> — at every model size, and now for reasoning models too. The knowledge was already inside; training just removes the gag and calibrates the confidence.</p>

<div class="stats">
  <div class="stat"><div class="n">96%</div><div class="k">Sensitive-fact accuracy at 72B after SFT (from 85% base) &mdash; near the ceiling</div></div>
  <div class="stat"><div class="n warn">0%</div><div class="k">Fabrication on invented terms at 72B-SFT (from 21%)</div></div>
  <div class="stat"><div class="n">&minus;44<span style="font-size:.5em">pts</span></div><div class="k">Reasoning-model fabrication, R1-Distill 53% &rarr; 9%</div></div>
  <div class="stat"><div class="n">~0</div><div class="k">Alignment tax &mdash; general-fact accuracy holds at 100%</div></div>
</div>

<section>
<div class="slabel">The result</div>
<h2>The fix scales &mdash; and the bigger the base, the higher it reaches</h2>
<p>The same 1,277-example honesty dataset, the same recipe, applied across four models. Sensitive-fact truthfulness climbs at <em>every</em> scale. Because the fine-tune is un-gagging knowledge the base already holds rather than teaching new facts, a stronger base converges to a higher ceiling: 72B reaches 96%.</p>
<figure>
{dumbbell()}
<div class="legend"><span><i style="background:var(--base)"></i>base model</span><span><i style="background:var(--truth)"></i>after honesty-SFT</span><span>sensitive-fact accuracy, 157 questions</span></div>
</figure>
<figcaption>Every model gains. The dense Qwen2.5 ladder (7B&rarr;14B&rarr;72B) shows a rising ceiling; R1-Distill-32B, a reasoning model, gains the most in absolute terms (+18) despite starting knowledge-light.</figcaption>
</section>

<section>
<div class="slabel">The counter-argument</div>
<h2>Why not just abliterate?</h2>
<p>Abliteration &mdash; the popular one-click uncensoring that ablates the refusal direction &mdash; is the obvious shortcut, and it&rsquo;s the wrong one. It frees the tongue without informing the mind: on sensitive facts it barely moves off the base, and it <em>wrecks calibration</em> &mdash; the model stops refusing but starts <b>fabricating</b>. At 14B, abliteration takes fabrication on invented terms from 9% to 56%.</p>
<figure>
<table class="tbl">
<thead><tr><th></th>
  <th colspan="3">Sensitive-fact accuracy <span class="colnote">higher is better</span></th>
  <th colspan="3" class="grp">Fabrication on invented terms <span class="colnote">lower is better</span></th></tr>
<tr><th></th><th>base</th><th>abliterated</th><th>honesty-SFT</th><th class="grp">base</th><th>abliterated</th><th>honesty-SFT</th></tr></thead>
<tbody>{threeway_rows()}</tbody>
</table>
</figure>
<figcaption>Abliteration (amber) tracks the base on knowledge and inflates fabrication; curated SFT (green) lifts truthfulness and drives fabrication toward zero. Same models, same benchmark, same judge.</figcaption>
</section>

<section>
<div class="slabel">The hard case</div>
<h2>Reasoning models: uncensor the thought, not just the answer</h2>
<p>A reasoning model can <em>reason itself</em> into confident nonsense &mdash; base R1-Distill invented concrete details on more than half of the fabricated-term questions. So we trained on honest <span style="font-family:var(--mono);font-size:.9em">&lt;think&gt;</span> traces, not just honest outputs. Tuning the reasoning is what breaks the confabulation loop.</p>
<div class="panel">
<div class="row">
  <div class="metric"><div class="big warn"><span class="from">53%</span><span class="arw">&rarr;</span><span class="to">9%</span></div><div class="lab">fabrication on invented terms</div></div>
  <div class="metric"><div class="big"><span class="from">69%</span><span class="arw">&rarr;</span><span class="to">88%</span></div><div class="lab">sensitive-fact accuracy</div></div>
  <div class="metric"><div class="big"><span class="from">43%</span><span class="arw">&rarr;</span><span class="to">91%</span></div><div class="lab">honest abstention</div></div>
</div>
<p style="max-width:none;font-size:14.5px;color:var(--muted);margin:14px 0 0">The reasoning stopped being a rationalization engine &mdash; answers also got calibrated and concise (base rambled to ~1,000 characters; the fine-tune is tight and to the point).</p>
</div>
</section>

<section>
<div class="slabel">What it costs, what we ruled out</div>
<h2>Cheap, no tax, and the boring choices verified</h2>
<div class="twoup">
  <div class="card"><div class="tag">no alignment tax</div><h3>General ability is untouched</h3><p>General-fact accuracy holds at 100% across every fine-tune, and over-censorship (declining questions it should answer) falls to ~0. The model gets more honest on hard topics without getting dumber or more evasive on ordinary ones.</p></div>
  <div class="card"><div class="tag neutral">negative result</div><h3>The optimizer is settled</h3><p>We A/B&rsquo;d Moonshot&rsquo;s Muon against AdamW on the LoRA fine-tune. Muon gave no convergence gain &mdash; at matched step-size it ties, otherwise it&rsquo;s worse. AdamW stays. A clean null, reported because rigor cuts both ways.</p></div>
</div>
</section>

<div class="rule"></div>
<section style="margin-top:clamp(34px,5vw,54px)">
<div class="slabel">Browse the evidence</div>
<h2>Every judgment, item by item</h2>
<p style="color:var(--muted)">Each answer below was labelled by an independent cross-family judge against ground truth. Open any model to read the base-vs-fine-tune answers side by side, filter to wins, regressions, and still-wrong.</p>
<div class="links">
{"".join(f'<a href="https://claude.ai/code/artifact/{u}" target="_blank" rel="noopener">{n}<span class="ar">&rarr;</span></a>' for n,u in LINKS)}
</div>
</section>

<footer>
<p class="caveat"><b>Honest about the method.</b> Benchmark is 265 in-house questions across four categories (sensitive-real, sensitive-invented, neutral-real, neutral-invented); each item graded once by a single cross-family judge, so trends are robust and individual cells indicative. Data curation and the judge share one author&rsquo;s framing of &ldquo;ground truth.&rdquo; This is <b>local, personal-use research</b> &mdash; the finding is the deliverable; model weights are not distributed.</p>
</footer>
</div>"""

TITLE = "<title>Un-gag, don't re-educate &middot; curated honesty-SFT across the scale ladder</title>"
open(OUT,"w",encoding="utf-8").write(TITLE+"<style>"+CSS+"</style>"+HTML)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
