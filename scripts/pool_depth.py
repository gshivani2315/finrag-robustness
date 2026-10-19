import json
rows = [json.loads(l) for l in open("outputs/exp2_bm25_k300/retrieval_bm25.jsonl", encoding="utf-8")]
print("questions:", len(rows), "| passages stored per question: min", min(len(r["retrieved"]) for r in rows), "max", max(len(r["retrieved"]) for r in rows))
grp = lambda r: "text" if r["answer_from"] == "text" else "text+table"
print(f"{'pool size':>9s} {'group':11s} {'questions with >=1 gold in pool':>32s} {'share of gold paragraphs':>25s} {'avg paragraphs in pool':>23s}")
for n in (100, 200, 300):
    for g, sel in (("all", lambda r: True), ("text", lambda r: grp(r) == "text"), ("text+table", lambda r: grp(r) == "text+table")):
        s = [r for r in rows if sel(r)]; a = c = p = 0
        for r in s:
            top = r["retrieved"][:n]; got = {x["passage_id"] for x in top} & set(r["gold_evidence_ids"])
            a += bool(got); c += len(got) / len(r["gold_evidence_ids"]); p += sum(x["source_type"] == "paragraph" for x in top)
        print(f"{n:9d} {g:11s} {a/len(s):32.3f} {c/len(s):25.3f} {p/len(s):23.1f}")