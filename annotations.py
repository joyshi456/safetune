import json
import os
from pathlib import Path
from huggingface_hub import InferenceClient

# ------------------ 1. Setup Client ------------------

client = InferenceClient(model=BASELINE_URL, token=HF_TOKEN)

# ------------------ 2. File Paths ------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

prompt_file = DATA_DIR / "unsafe.txt"
output_file = DATA_DIR / "outputs.json"

print(f"📁 Prompt file path: {prompt_file.resolve()}")
print(f"📄 Output file path: {output_file.resolve()}")

# ------------------ 3. Load Existing Output ------------------
results = {}
if output_file.exists():
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                results = json.loads(content)
                print(f"📦 Loaded {len(results)} existing prompt outputs.")
            else:
                print("⚠️ outputs.json is empty. Starting fresh.")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to load outputs.json due to JSON error: {e}. Starting fresh.")

# ------------------ 4. Check Prompt File ------------------
if not prompt_file.exists():
    print("❌ data/unsafe.txt not found.")
    exit(1)

with open(prompt_file, "r", encoding="utf-8", errors="ignore") as f:
    prompts = [line.strip() for line in f if line.strip()]


print(f"📝 Loaded {len(prompts)} prompts from unsafe.txt")
print(f"🧠 Skipping {len([p for p in prompts if p in results])} already processed prompts")

# ------------------ 5. Text Generation Function ------------------
def generate_text(prompt: str) -> str:
    print(f"🧪 Generating output for prompt: {prompt[:60]}...")
    return client.text_generation(
        prompt=prompt,
        max_new_tokens=200,
        top_p=0.9,
        temperature=1.5,
        do_sample=True,
    )

# ------------------ 6. Process Prompts ------------------
count = 0
for idx, prompt in enumerate(prompts, 1):
    if prompt in results:
        print(f"⏩ [{idx}] Already processed. Skipping: {prompt[:60]}")
        continue
    try:
        output = generate_text(prompt)
        results[prompt] = {"prompt": prompt, "output": output}
        count += 1
        print(f"✅ [{idx}] New output saved.")
    except Exception as e:
        print(f"⚠️ [{idx}] Error processing prompt: {prompt[:60]} — {e}")

# ------------------ 7. Save Outputs ------------------
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved updated results to: {output_file.resolve()}")
except Exception as e:
    print(f"❌ Failed to write outputs.json — {e}")

# ------------------ 8. Final Summary ------------------
print(f"\n✨ Done. Total prompts processed this run: {count}")
print(f"📊 Total prompts in outputs.json: {len(results)}")
