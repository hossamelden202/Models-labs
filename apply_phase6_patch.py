import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "src/modellab/server/app.py")
anchor = "    add_pipeline_routes(api, ws, jobs, cache, store, enqueue, art, read)\n"
addition = "    from modellab.ai.routes import add_ai_routes\n\n    add_ai_routes(api, ws, jobs, cache, store)\n"

text = target.read_text()
if "add_ai_routes" in text:
    print("already patched")
    sys.exit(0)
if text.count(anchor) != 1:
    sys.exit("anchor line not found exactly once, add the two lines by hand")
shutil.copy(target, target.with_name(target.name + ".bak_phase6"))
target.write_text(text.replace(anchor, anchor + addition))
print("patched", target)
