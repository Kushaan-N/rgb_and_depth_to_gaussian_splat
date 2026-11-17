"""Vision-language stage of the layered object pipeline (scripts/object_layers/object_layers.py). Tiny HTTP client: runs on
any node with internet (login node / short CPU job), never on the GPU path. Official `anthropic` SDK ($CEAR_WS/venv-vlm).

  recipe   per object: real object cut-outs on black (<dir>/vlm_in/obj<id>_cut<k>.jpg, written by `object_layers.py
           evidence`) -> strict-JSON shape recipe (primitives, stacking, materials, symmetry, detection check)
           -> <dir>/vlm/recipe_obj<id>.json
  select   per object: candidate sheet (<dir>/cand/obj<id>_sheet.jpg: primitive fit vs generated mesh, real + novel views)
           + numeric check -> <dir>/vlm/select_obj<id>.json
  review   assembled-scene renders (<dir>/review/round<k>/*.jpg) -> structured edits <dir>/vlm/review_round<k>.json

The API key is read from --key-file only (default ~/.config/scan2sim/anthropic_key); it is never printed, logged, put on a
command line or exported. Without a key / SDK / network every command exits 0 after writing nothing: the pipeline then
falls back to the generated mesh. Every response is cached in --cache-dir keyed by sha256(model, system, prompt, image
bytes): reruns are free and deterministic. Token usage per call -> <dir>/vlm/usage.jsonl; `cost` sums it per scene.
"""
from __future__ import annotations
import argparse, base64, glob, hashlib, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prompts_v2 as PR  # noqa: E402  (versioned prompts: a new version = a new file)

PRICE = {"claude-opus-5-5": (4.0, 20.0)}          # $ / 1M input, output tokens (claude-api reference, 2026-11)


class Client:
    def __init__(self, a):
        self.a = a; self.client = None; self.why = None
        os.makedirs(a.cache_dir, exist_ok=True); os.makedirs(os.path.join(a.dir, "vlm"), exist_ok=True)
        kf = os.path.expanduser(a.key_file)
        if not os.path.isfile(kf):
            self.why = "no key file"; return
        try:
            import anthropic
        except ImportError:
            self.why = "anthropic SDK not installed in this python"; return
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=open(kf).read().strip(), max_retries=3, timeout=300.0)

    def ask(self, tag, prompt, images, max_tokens=8000):
        """-> parsed JSON dict or None. images: list of file paths (jpg / png)."""
        a = self.a; h = hashlib.sha256(); blobs = []
        for part in (a.model, a.effort, PR.SYSTEM, prompt):
            h.update(part.encode()); h.update(b"\0")
        for f in images:
            b = open(f, "rb").read(); h.update(b); blobs.append((f, b))
        key = h.hexdigest(); cf = os.path.join(a.cache_dir, key + ".json")
        if os.path.exists(cf):
            j = json.load(open(cf)); self.log(tag, key, j.get("usage", {}), cached=True); return j["json"]
        if self.client is None:
            print(f"[vlm] {tag}: skipped ({self.why}) and not cached", flush=True); return None
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png" if f.endswith(".png") else "image/jpeg",
                                                 "data": base64.standard_b64encode(b).decode()}} for f, b in blobs]
        content.append({"type": "text", "text": prompt}); an = self.anthropic
        try:
            r = self.client.messages.create(model=a.model, max_tokens=max_tokens, system=PR.SYSTEM,
                                            output_config={"effort": a.effort}, messages=[{"role": "user", "content": content}])
        except an.RateLimitError as e:
            print(f"[vlm] {tag}: rate limited ({e.status_code}), skipped", flush=True); return None
        except an.APIStatusError as e:
            print(f"[vlm] {tag}: API error {e.status_code}: {str(e.message)[:200]}", flush=True); return None
        except an.APIConnectionError:
            print(f"[vlm] {tag}: no connection to the API from this node, skipped", flush=True); return None
        if r.stop_reason == "refusal":
            print(f"[vlm] {tag}: declined by the model, skipped", flush=True); return None
        text = "".join(b.text for b in r.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            js = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            js = None
        if js is None:
            print(f"[vlm] {tag}: answer is not JSON (stop_reason {r.stop_reason}), skipped", flush=True); return None
        u = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
        json.dump({"json": js, "usage": u, "model": a.model, "prompt_version": PR.VERSION, "tag": tag, "images": [os.path.basename(f) for f in images],
                   "time": time.strftime("%F %T")}, open(cf, "w"), indent=1)
        self.log(tag, key, u, cached=False); return js

    def log(self, tag, key, u, cached):
        pi, po = PRICE.get(self.a.model, (0.0, 0.0))
        row = {"tag": tag, "key": key[:16], "cached": cached, "model": self.a.model, **u,
               "usd": 0.0 if cached else round((u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6, 5),
               "usd_if_uncached": round((u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6, 5)}
        open(os.path.join(self.a.dir, "vlm", "usage.jsonl"), "a").write(json.dumps(row) + "\n")
        print(f"[vlm] {tag}: {'cache hit' if cached else 'called'} in {u.get('input_tokens')} out {u.get('output_tokens')} tokens ${row['usd']}", flush=True)


def ids_of(d, pat):
    return sorted({int(re.search(r"obj(\d+)", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(d, pat))})


def recipe(a):
    c = Client(a); ev = json.load(open(os.path.join(a.dir, "evidence.json")))
    for o in ev["objects"]:
        i = o["id"]; out = os.path.join(a.dir, "vlm", f"recipe_obj{i}.json")
        cut = sorted(glob.glob(os.path.join(a.dir, "vlm_in", f"obj{i}_cut*.jpg")))[:a.max_images]
        if not cut:
            continue
        bx, by = sorted(o["box_m"][:2], reverse=True)
        p = PR.RECIPE.format(n_cut=len(cut), bx=bx, by=by, bz=o["box_m"][2])
        js = c.ask(f"recipe_obj{i}", p, cut)                                # object cut-outs only: no room context leaves the cluster
        if js is not None:
            json.dump(js, open(out, "w"), indent=1)


def select(a):
    c = Client(a)
    for i in ids_of(os.path.join(a.dir, "cand"), "obj*_sheet.jpg"):
        num = json.load(open(os.path.join(a.dir, "cand", f"obj{i}_numeric.json")))
        A, B = num.get("prim", {}), num.get("gen", {})
        p = PR.SELECT.format(iou_a=A.get("iou"), psnr_a=A.get("psnr"), iou_b=B.get("iou"), psnr_b=B.get("psnr"))
        js = c.ask(f"select_obj{i}", p, [os.path.join(a.dir, "cand", f"obj{i}_sheet.jpg")])
        if js is not None and js.get("choice") in ("A", "B"):
            js["candidate"] = {"A": "prim", "B": "gen"}[js["choice"]]
            json.dump(js, open(os.path.join(a.dir, "vlm", f"select_obj{i}.json"), "w"), indent=1)


def review(a):
    c = Client(a); rd = os.path.join(a.dir, "review", f"round{a.round}")
    ims = sorted(glob.glob(os.path.join(rd, "*.jpg")))[:a.max_review_images]
    if not ims:
        print("[vlm] review: no renders", flush=True); return
    fin = json.load(open(os.path.join(a.dir, "objects_layers.json")))
    objs = "; ".join(f"id {k}: {v.get('size_m')}" for k, v in fin["objects"].items() if v.get("size_m"))
    js = c.ask(f"review_round{a.round}", PR.REVIEW.format(objects=objs), ims)
    if js is not None:
        json.dump(js, open(os.path.join(a.dir, "vlm", f"review_round{a.round}.json"), "w"), indent=1)


def cost(a):
    f = os.path.join(a.dir, "vlm", "usage.jsonl"); rows = [json.loads(x) for x in open(f)] if os.path.exists(f) else []
    uniq = {}
    for r in rows:
        uniq.setdefault(r["key"], r)
    out = {"calls_logged": len(rows), "api_calls_billed": sum(not r["cached"] for r in rows), "unique_requests": len(uniq),
           "input_tokens": sum(r.get("input_tokens", 0) for r in uniq.values()), "output_tokens": sum(r.get("output_tokens", 0) for r in uniq.values()),
           "usd_billed": round(sum(r["usd"] for r in rows), 4), "usd_scene_if_uncached": round(sum(r["usd_if_uncached"] for r in uniq.values()), 4),
           "model": a.model, "prompt_version": PR.VERSION}
    json.dump(out, open(os.path.join(a.dir, "vlm", "cost.json"), "w"), indent=1); print("VLM_COST " + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["recipe", "select", "review", "cost"]); ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="claude-opus-5-5"); ap.add_argument("--effort", default="high")
    ap.add_argument("--key-file", default="~/.config/scan2sim/anthropic_key")
    ap.add_argument("--cache-dir", default=os.path.join(os.environ.get("CEAR_WS", "."), "vlm-cache"))
    ap.add_argument("--max-images", type=int, default=4); ap.add_argument("--max-review-images", type=int, default=12)
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args(); {"recipe": recipe, "select": select, "review": review, "cost": cost}[a.cmd](a)


if __name__ == "__main__":
    main()
