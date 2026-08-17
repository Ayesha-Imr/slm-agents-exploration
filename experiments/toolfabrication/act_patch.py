"""Idea 3 - causal iteration: activation patching of the tool/context ("registry")
representation to localize where correct tool selection is computed.

Follow-up to smoke.py (probes were strong, naive steering weak). Here we ask the
causal question directly: does the pre-query context stream carry the identity of
the correct tool, and at which layers?

Design (per correctly-grounded query):
  clean   = prompt whose registry contains the GT tool (model calls it correctly)
  corrupt = same query, GT tool renamed ("<name>_v2"); identity removed, otherwise
            structurally identical (same params/description, same query)
  cache per-layer residual states for the context region [0, R) of BOTH runs, then:
    - corrupt-patch: run clean generation but inject corrupt's context activations
      into layer L's output. If the context stream causally grounds tool identity,
      correct-rate should fall and reproduce the corrupt behavior.
    - mean-ablate: inject the mean context activation (removes positional identity).
  Sweep L over all layers -> per-layer causal curve for both interventions.

Counterfactual baseline: actually run the corrupt prompt to characterize what
"broken grounding" produces (fabricate vs no-call vs use-renamed-tool).

Model: Qwen/Qwen2.5-3B-Instruct. Dataset: BFCL v3 (cached from smoke.py).
Outputs: results/toolfabrication/act_patch.json
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke  # noqa: E402  (reuse load_model/build_prompt/label_output/...)

OUT_DIR = os.environ.get("SMOKE_OUT", "results/toolfabrication")


def tokenize_ids(tok, prompt):
    return tok(prompt, return_tensors="pt").input_ids[0].tolist()


def find_registry_end(ids, question_tokens):
    """Return the index where the user question begins (everything before = context:
    system message + tool definitions). Search for the question's first tokens."""
    k = min(len(question_tokens), 12)
    if k == 0:
        return min(len(ids) // 3, 400)
    needle = question_tokens[:k]
    for i in range(len(ids) - k + 1):
        if ids[i:i + k] == needle:
            return i
    return min(len(ids) // 3, 400)


def cache_ctx_states(model, tok, prompt, R):
    """Per-layer residual state for context positions [0, R). Returns dict L -> (R, D) bf16."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    n_layers = len(model.model.layers)
    hs_all = out.hidden_states
    assert len(hs_all) >= n_layers + 1, f"hidden_states len {len(hs_all)} != {n_layers}+1"
    ctx = {}
    for L in range(n_layers):
        ctx[L] = hs_all[L + 1][0, :R, :].to(torch.bfloat16).cpu().numpy()
    return ctx


def gen_patched(model, tok, prompt, patch_layer, positions, replacement, max_new_tokens=160):
    """Generate with layer `patch_layer`'s output replaced at `positions` by `replacement`
    (R, D) on the first (prompt) forward pass only."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    layer = model.model.layers[patch_layer]
    repl = torch.tensor(replacement, dtype=torch.bfloat16, device=model.device)
    pos = torch.tensor(positions, dtype=torch.long, device=model.device)
    state = {"done": False}

    def hook(module, args, output):
        if state["done"]:
            return output
        state["done"] = True
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, pos, :] = repl
            return (hs,) + tuple(output[1:])
        hs = output.clone()
        hs[:, pos, :] = repl
        return hs

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id)
    finally:
        handle.remove()
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def corrupt_tools(item, gts):
    """Rename GT tools to '<name>_v2' (removes correct identity, keeps structure)."""
    tools = [dict(t) for t in smoke.item_tools(item)]
    for t in tools:
        if t.get("name") in gts:
            t["name"] = t["name"] + "_v2"
    return tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patch", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--queries", default=os.path.join(OUT_DIR, "h1_queries.json"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    rng = random.Random(0)

    with open(args.queries) as f:
        queries = json.load(f)

    pa_maps = {}
    for key, filename in (("multiple", "BFCL_v3_multiple.json"),
                          ("simple", "BFCL_v3_simple.json")):
        try:
            pa_maps[key] = smoke.load_pa(filename)
        except Exception as e:
            print(f"possible_answer load failed for {filename}: {e}", flush=True)
            pa_maps[key] = {}

    def resolve_gts(item, category):
        return smoke.gt_names(item) or pa_maps.get(category, {}).get(item.get("id"), [])

    tok, model = smoke.load_model()
    n_layers = len(model.model.layers)
    print(f"[{time.time()-t0:.0f}s] model loaded, {n_layers} layers", flush=True)

    # ---- select a correctly-grounded pool (re-generate clean baseline) ----
    candidates = []
    for q in queries:
        gts = resolve_gts(q["item"], q["category"])
        if len(gts) == 1:
            candidates.append((q, gts))
    rng.shuffle(candidates)

    pool = []  # (q, gts, clean_label, clean_calls)
    attempts = 0
    for q, gts in candidates:
        if len(pool) >= args.n_patch:
            break
        attempts += 1
        if attempts > 5 * args.n_patch:
            break
        text, _, _ = smoke.gen(model, tok, q["prompt"], max_new_tokens=args.max_tokens)
        lab, calls = smoke.label_output(text, q["registry"], gts)
        if lab == "correct":
            pool.append((q, gts, calls))
    print(f"[{time.time()-t0:.0f}s] clean pool: {len(pool)} correct queries "
          f"(from {attempts} candidates)", flush=True)
    if not pool:
        print("FATAL: no correctly-grounded pool found", flush=True)
        sys.exit(1)

    layers = list(range(0, n_layers, args.stride))
    # per-layer label tallies for corrupt-patch and mean-ablate
    patch_tally = {L: {"correct": 0, "wrong_tool": 0, "fabricated": 0, "no_call": 0} for L in layers}
    ablate_tally = {L: {"correct": 0, "wrong_tool": 0, "fabricated": 0, "no_call": 0} for L in layers}
    corrupt_baseline = {"correct": 0, "wrong_tool": 0, "fabricated": 0, "no_call": 0,
                        "calls": []}
    per_query = []

    for pi, (q, gts, clean_calls) in enumerate(pool):
        item = q["item"]
        question = smoke.item_question(item)
        q_tokens = tokenize_ids(tok, question) if question else []

        corrupt = corrupt_tools(item, gts)
        corrupt_prompt = smoke.build_prompt(tok, corrupt, question)
        corrupt_registry = [t["name"] for t in corrupt]

        ids_clean = tokenize_ids(tok, q["prompt"])
        ids_corrupt = tokenize_ids(tok, corrupt_prompt)
        re_clean = find_registry_end(ids_clean, q_tokens)
        re_corrupt = find_registry_end(ids_corrupt, q_tokens)
        R = min(re_clean, re_corrupt)
        positions = list(range(R))

        clean_ctx = cache_ctx_states(model, tok, q["prompt"], R)
        corrupt_ctx = cache_ctx_states(model, tok, corrupt_prompt, R)

        # counterfactual baseline: run the corrupt prompt directly
        ctext, _, _ = smoke.gen(model, tok, corrupt_prompt, max_new_tokens=args.max_tokens)
        clab, ccalls = smoke.label_output(ctext, corrupt_registry, [])
        corrupt_baseline[clab] += 1
        corrupt_baseline["calls"].append(ccalls)

        per_q = {"item_id": q["item_id"], "category": q["category"], "gt": gts,
                 "clean_calls": clean_calls, "corrupt_label": clab, "corrupt_calls": ccalls,
                 "R": R, "patch": {}, "ablate": {}}
        for L in layers:
            text = gen_patched(model, tok, q["prompt"], L, positions, corrupt_ctx[L],
                               max_new_tokens=args.max_tokens)
            lab, _ = smoke.label_output(text, q["registry"], gts)
            patch_tally[L][lab] += 1
            per_q["patch"][str(L)] = lab

            mean = clean_ctx[L].mean(axis=0, keepdims=True)
            mean_rep = np.broadcast_to(mean, (R, clean_ctx[L].shape[1])).copy()
            text = gen_patched(model, tok, q["prompt"], L, positions, mean_rep,
                               max_new_tokens=args.max_tokens)
            lab, _ = smoke.label_output(text, q["registry"], gts)
            ablate_tally[L][lab] += 1
            per_q["ablate"][str(L)] = lab

        per_query.append(per_q)
        el = time.time() - t0
        print(f"[{el:.0f}s] query {pi+1}/{len(pool)} done (gt={gts}, corrupt={clab})",
              flush=True)

    n = len(pool)
    def rates(tally):
        return {str(L): {k: tally[L][k] for k in tally[L]} for L in tally}

    def correct_rate(tally):
        return {str(L): round(tally[L]["correct"] / n, 4) for L in tally}

    summary = {
        "model": smoke.MODEL_ID,
        "n_pool": n,
        "n_layers": n_layers,
        "layers_swept": layers,
        "R_info": {"mean_R": int(np.mean([p["R"] for p in per_query]))},
        "clean_baseline_correct_rate": 1.0,
        "corrupt_baseline": {k: corrupt_baseline[k] for k in ("correct", "wrong_tool",
                                                              "fabricated", "no_call")},
        "corrupt_baseline_rate": {k: round(corrupt_baseline[k] / n, 4)
                                  for k in ("correct", "wrong_tool", "fabricated", "no_call")},
        "patch_correct_rate_by_layer": correct_rate(patch_tally),
        "ablate_correct_rate_by_layer": correct_rate(ablate_tally),
        "patch_tally": rates(patch_tally),
        "ablate_tally": rates(ablate_tally),
        "per_query": per_query,
        "wall_seconds": float(time.time() - t0),
    }
    out_path = os.path.join(args.out, "act_patch.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    cr_patch = correct_rate(patch_tally)
    cr_abl = correct_rate(ablate_tally)
    worst_p = min(cr_patch, key=cr_patch.get)
    worst_a = min(cr_abl, key=cr_abl.get)
    print("PATCH_CORRECT_RATES " + json.dumps(cr_patch), flush=True)
    print("ABLATE_CORRECT_RATES " + json.dumps(cr_abl), flush=True)
    print(f"corrupt_baseline: {json.dumps(summary['corrupt_baseline_rate'])}", flush=True)
    print(f"max patch effect at layer {worst_p}: correct {cr_patch[worst_p]} (baseline 1.0)", flush=True)
    print(f"max ablate effect at layer {worst_a}: correct {cr_abl[worst_a]} (baseline 1.0)", flush=True)
    print("FINAL_SUMMARY " + json.dumps(summary), flush=True)
    print(f"results saved to {out_path}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: {e}", flush=True)
        sys.exit(1)
