import json, os, shutil
from huggingface_hub import snapshot_download
src = snapshot_download("z-lab/Qwen3.5-9B-DFlash")
dst = os.path.expanduser("~/models/qwen35-9b-dflash")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.startswith(".") and os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), dst)
p = os.path.join(dst, "config.json")
c = json.load(open(p))
c["rope_theta"] = c.get("rope_parameters", {}).get("rope_theta", 10000000)
c["block_size"] = c.get("dflash_config", {}).get("block_size", 16)
json.dump(c, open(p, "w"), indent=2)
print("patched", dst, "rope_theta", c["rope_theta"], "block_size", c["block_size"])
