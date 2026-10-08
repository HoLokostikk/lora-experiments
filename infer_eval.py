import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, ".")
from src.prompts import MINIMAL
from src.eval_json import evaluate

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = sys.argv[1] if len(sys.argv) > 1 else None

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
if adapter:
    model = PeftModel.from_pretrained(model, adapter)
model.eval()

rows = [json.loads(l) for l in open("data/test.jsonl")]
preds = []
for i, r in enumerate(rows, 1):
    msgs = [{"role": "user", "content": MINIMAL.format(text=r["text"])}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    preds.append(tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True))
    print(f"\r{i}/{len(rows)}", end="", flush=True)
print()

m = evaluate(preds, [r["gold"] for r in rows])
print(json.dumps(m, indent=2))

tag = (adapter or "base").replace("/", "_")
with open(f"eval_{tag}.json", "w") as f:
    json.dump({"adapter": adapter, "metrics": m, "raw": preds}, f, indent=2)
