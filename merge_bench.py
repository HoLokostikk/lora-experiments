import time, torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "out/lora-r16/final"
MERGED = "out/merged-r16"

tok = AutoTokenizer.from_pretrained(BASE)
prompt = "Analyze this review:\nShipping took three weeks and the box was crushed."
enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to("cuda")


def bench(model, label, n_new=100, reps=5):
    model.eval()
    with torch.no_grad():                       # прогрів
        model.generate(**enc, max_new_tokens=16, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"{label:22s}  {n_new/best:6.1f} tok/s   (best {best:.3f}s з {reps})")


base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
bench(base, "base")

peft_model = PeftModel.from_pretrained(base, ADAPTER)
bench(peft_model, "base + adapter")

merged = peft_model.merge_and_unload()
bench(merged, "merged")

merged.save_pretrained(MERGED)
tok.save_pretrained(MERGED)
print("saved:", MERGED)
