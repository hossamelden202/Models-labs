import json


def _shrink(obj, depth, max_depth):
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"{{{len(obj)} keys}}"
        items = list(obj.items())
        out = {str(k): _shrink(v, depth + 1, max_depth) for k, v in items[:20]}
        if len(items) > 20:
            out["..."] = f"{len(items) - 20} more keys"
        return out
    if isinstance(obj, list):
        if depth >= max_depth:
            return f"[{len(obj)} items]"
        head = [_shrink(v, depth + 1, max_depth) for v in obj[:3]]
        return head + [f"...{len(obj)} items"] if len(obj) > 3 else head
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, str):
        return obj if len(obj) <= 120 else obj[:117] + "..."
    return obj


def digest_json(obj, max_chars=1500):
    text = ""
    for depth in (4, 3, 2, 1):
        text = json.dumps(_shrink(obj, 0, depth), separators=(",", ":"), ensure_ascii=False, default=str)
        if len(text) <= max_chars:
            return text
    return text[: max_chars - 3] + "..."
