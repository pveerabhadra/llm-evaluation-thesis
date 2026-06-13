import json
fname = "mmlu_en_qwen3.5_v1.jsonl"   
problems = []
with open(fname) as f:
    for line in f:
        r = json.loads(line)
        if r.get("error") or r.get("parse_method") == "failed":
            problems.append(r)
print(f"Problems: {len(problems)}")
for r in problems:
    print(f"  id={r['id']:>4}  subject={r['subject']:<35}  tokens={r.get('completion_tokens')}  error={r.get('error')}  parse={r.get('parse_method')}")
