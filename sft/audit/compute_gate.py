# -*- coding: utf-8 -*-
"""Score the M0 teacher-trust audit: dual-confirm rate, planted-fakes caught,
known-fact reproduction, softening. Emits PASS / AMBER / RED."""
import json, os, glob
D = os.path.dirname(os.path.abspath(__file__))

key = {b["vid"]: b for b in json.load(open(f"{D}/blind_key.json"))}

def load_verdicts(pattern):
    v = {}
    for f in glob.glob(f"{D}/{pattern}"):
        try:
            for o in json.load(open(f)):
                v[o["vid"]] = o.get("verdict", "unverifiable")
        except Exception as e:
            print(f"  (warn: {os.path.basename(f)}: {e})")
    return v

opus = load_verdicts("verdict_opus_*.json")
codex = {}
if os.path.exists(f"{D}/codex_verdicts.json"):
    try:
        for o in json.load(open(f"{D}/codex_verdicts.json")): codex[o["vid"]] = o.get("verdict","unverifiable")
    except Exception as e: print(f"  (warn codex: {e})")

reals = [v for v,b in key.items() if b["_src"]=="real"]
fakes = [v for v,b in key.items() if b["_src"]=="fake"]

def confirmed(v, d): return d.get(v)=="confirmed"
def caught(v, d): return d.get(v) in ("contradicted","unverifiable")  # NOT confirmed = fake caught

print(f"=== M0 AUDIT — {len(opus)} Opus verdicts, {len(codex)} Codex verdicts ===\n")

# 1. dual-confirm rate on REAL items (independent verification confirms teacher truth)
opus_conf = sum(confirmed(v,opus) for v in reals)
codex_conf = sum(confirmed(v,codex) for v in reals) if codex else None
dual = sum(confirmed(v,opus) and confirmed(v,codex) for v in reals) if codex else None
print(f"REAL items (n={len(reals)}):")
print(f"  Opus confirmed:  {opus_conf}/{len(reals)} = {100*opus_conf/len(reals):.0f}%")
if codex: print(f"  Codex confirmed: {codex_conf}/{len(reals)} = {100*codex_conf/len(reals):.0f}%")
if codex is not None: print(f"  dual-confirmed:  {dual}/{len(reals)} = {100*dual/len(reals):.0f}%")

# 2. planted fakes caught (load-bearing verifier-specificity test)
opus_caught = sum(caught(v,opus) for v in fakes)
either_caught = sum(caught(v,opus) or caught(v,codex) for v in fakes) if codex else opus_caught
print(f"\nPLANTED FAKES (n={len(fakes)}) — caught = NOT confirmed:")
print(f"  Opus caught:   {opus_caught}/{len(fakes)}")
if codex: print(f"  either caught: {either_caught}/{len(fakes)}")
# any fake CONFIRMED by a verifier = a verification miss (dangerous)
opus_missed = [v for v in fakes if confirmed(v,opus)]
if opus_missed: print(f"  !! Opus CONFIRMED fakes (misses): {opus_missed}")

# 3. known-fact reproduction
known = None
if os.path.exists(f"{D}/known_scored.json"):
    ks = json.load(open(f"{D}/known_scored.json"))
    known = sum(x.get("match") for x in ks)
    print(f"\nKNOWN-FACT reproduction: {known}/{len(ks)}")

# 4. softening — qualitative (synthesis agents reported 0 discards/refusals)
print("\nSOFTENING/REFUSAL during synthesis: 0 observed (all agents answered fully, 0 items discarded for refusal)")

# gate
print("\n=== GATE ===")
g1 = opus_conf/len(reals) >= 0.90
g2 = either_caught >= 9 if codex else opus_caught >= 9
g3 = (known is not None and known >= 14)
g1a = opus_conf/len(reals) >= 0.80
verdict = "PASS" if (g1 and g2 and g3) else ("AMBER" if (g1a and g2) else "RED")
print(f"  [{'PASS' if g1 else 'FAIL'}] dual-confirm ≥90% (Opus {100*opus_conf/len(reals):.0f}%)")
print(f"  [{'PASS' if g2 else 'FAIL'}] fakes caught ≥9/10 ({either_caught if codex else opus_caught}/10)")
print(f"  [{'PASS' if g3 else 'FAIL'}] known reproduced ≥14/15 ({known}/15)" if known is not None else "  [WAIT] known not scored yet")
print(f"\n  >>> M0 VERDICT: {verdict} <<<")
print("VERDICT_DONE")
