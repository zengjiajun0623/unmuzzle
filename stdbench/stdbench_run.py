# -*- coding: utf-8 -*-
"""Industry-standard MC eval (CMMLU + MMLU) via Ollama, base vs SFT.
Measures the 'alignment tax' — did honesty-SFT degrade general ability."""
import json, re, urllib.request, sys
items=json.load(open("stdbench_items.json"))
def ask(model, it):
    o=it["opts"]
    if it["src"]=="CMMLU":
        p=f"以下是一道单项选择题，请只回答正确选项的字母（A、B、C 或 D），不要解释。\n\n{it['q']}\nA. {o['A']}\nB. {o['B']}\nC. {o['C']}\nD. {o['D']}\n\n答案："
    else:
        p=f"The following is a multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n{it['q']}\nA. {o['A']}\nB. {o['B']}\nC. {o['C']}\nD. {o['D']}\n\nAnswer:"
    body=json.dumps({"model":model,"prompt":p,"stream":False,"options":{"temperature":0,"num_predict":16}}).encode()
    req=urllib.request.Request("http://localhost:11434/api/generate",data=body,headers={"Content-Type":"application/json"})
    try:
        r=json.loads(urllib.request.urlopen(req,timeout=90).read()).get("response","") or ""
    except Exception as e:
        return "?"
    up=r.upper()
    m=re.search(r"(?:答案|ANSWER|正确选项|正确答案)[^A-D]{0,6}([ABCD])", up) or re.search(r"\b([ABCD])[\.\)、:：]", up) or re.search(r"([ABCD])", up)
    return m.group(1) if m else "?"

models=["qwen2.5:7b","unmuzzle-qwen7b"]
res={}
for m in models:
    cor={"CMMLU":0,"MMLU":0}; tot={"CMMLU":0,"MMLU":0}
    for i,it in enumerate(items):
        a=ask(m,it); tot[it["src"]]+=1
        if a==it["ans"]: cor[it["src"]]+=1
        if i%100==0: print(f"{m} {i}/{len(items)}",flush=True)
    res[m]={s:[cor[s],tot[s]] for s in ("CMMLU","MMLU")}
    json.dump(res, open("stdbench_results.json","w"))
    print(f"DONE {m}: {res[m]}",flush=True)
print("\n=== RESULTS (alignment-tax check) ===")
for m in models:
    c=res[m]
    print(f"{m:<22} CMMLU {c['CMMLU'][0]}/{c['CMMLU'][1]} ({100*c['CMMLU'][0]/c['CMMLU'][1]:.1f}%)  MMLU {c['MMLU'][0]}/{c['MMLU'][1]} ({100*c['MMLU'][0]/c['MMLU'][1]:.1f}%)")
print("STDBENCH_DONE")
