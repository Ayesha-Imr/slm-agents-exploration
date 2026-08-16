"""Smoke test for Idea 3: Grounding Failure - a mechanistic account of fabricated tool calls.

Fabrication = the model emits a tool NAME that is not in the provided registry
(distinct from misrouting to a valid-but-wrong tool, invalid args, or over-calling).

Validates:
  H1: fabrication is a real, measurable phenomenon on BFCL v3 with a 3B model
      (fabrication rate per category, greedy vs sampled).
  H2: "will fabricate vs will call a valid tool" is linearly separable from
      pre-generation hidden states (prompt-final AND first generated token),
      with a layer-wise AUROC curve.
  H3: steering a linear "toolness" direction changes call rate; steering the
      valid-vs-fabricated probe direction changes the fabrication rate
      (causal smoke test for the grounding hypothesis).
  H4: corrupting the registry (removing the needed tool) shifts the grounding
      probe score and raises fabrication - a minimal circuit-level correlate
      of "where the menu is grounded".

Model: Qwen/Qwen2.5-3B-Instruct. Dataset: BFCL v3 from HF.
Outputs: JSON + printed summary.
"""

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = os.environ.get("SMOKE_OUT", "results/toolfabrication")
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def load_bfcl(filename):
    path = hf_hub_download(BFCL_REPO, filename, repo_type="dataset")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def item_question(item):
    q = item.get("question") or item.get("conversations")
    if isinstance(q, list) and q:
        first = q[0]
        if isinstance(first, dict):
            return first.get("text") or first.get("content") or ""
        if isinstance(first, list) and first:
            d = first[0]
            if isinstance(d, dict):
                return d.get("text") or d.get("content") or ""
    return ""


def item_tools(item):
    return item.get("function") or []


def gt_names(item):
    for key in ("possible_answers", "possible_answer", "ground_truth"):
        pa = item.get(key)
        if pa:
            if isinstance(pa, list):
                return [p.get("name") for p in pa if isinstance(p, dict) and p.get("name")]
            if isinstance(pa, dict) and pa.get("name"):
                return [pa["name"]]
    return []


def build_prompt(tok, tools, question):
    sys_msg = (
        "You are a helpful assistant with access to the following functions. "
        "Use the functions when they are relevant to the user's request. "
        "If no function is relevant, or required parameters are missing, say so "
        "instead of calling a function."
    )
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": question}]
    kwargs = {}
    if tools:
        kwargs["tools"] = tools
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   **kwargs)


CALL_RE = re.compile(r"\[\s*([A-Za-z_]\w*)\s*\(")
JSON_NAME_RE = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"([A-Za-z_][\w.]*)\"")
ANY_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^)]")
TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*\{.*?\"name\"\s*:\s*\"([^\"]+)\"", re.S)
NONTOOL_WORDS = {"note", "see", "e.g", "i.e", "such", "like", "call", "use", "if",
                 "or", "and", "not", "no", "so", "then", "please", "here", "this",
                 "that", "example", "cases", "case", "check", "request", "answer",
                 "because", "since", "while", "before", "after", "for", "with"}


def parse_calls(text):
    names = [m.group(1) for m in TOOLCALL_TAG_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in CALL_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in JSON_NAME_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in ANY_CALL_RE.finditer(text)]
    names = [n for n in names if n.lower() not in NONTOOL_WORDS]
    return names[:3]


def parse_calls(text):
    names = [m.group(1) for m in CALL_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in JSON_NAME_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in TOOLCALL_TAG_RE.finditer(text)]
    if names:
        return names
    return [m.group(1) for m in ANY_CALL_RE.finditer(text)][:3]


def label_output(text, registry, gts):
    calls = parse_calls(text)
    if not calls:
        return "no_call", calls
    names = set(calls)
    if not names <= set(registry):
        return "fabricated", calls
    if gts and (names & set(gts)):
        return "correct", calls
    return "wrong_tool", calls


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model


def gen(model, tok, prompt, max_new_tokens=220, do_sample=False, temperature=1.0,
        steer_layer=None, steer_dir=None, steer_alpha=0.0):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    hooks = []
    if steer_layer is not None and steer_dir is not None and steer_alpha != 0.0:
        layer = model.model.layers[steer_layer]
        direction = torch.tensor(steer_dir, dtype=torch.bfloat16, device=model.device)
        direction = direction / (direction.norm() + 1e-8) * steer_alpha

        def hook(module, args):
            hs = args[0]
            hs = hs.clone()
            hs[:, -1:, :] += direction
            return (hs, *args[1:])
        hooks.append(layer.register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=0.9 if do_sample else 1.0,
                pad_token_id=tok.eos_token_id,
                output_hidden_states=True, return_dict_in_generate=True)
    finally:
        for h in hooks:
            h.remove()
    text = tok.decode(out.sequences[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hs0 = out.hidden_states[0]          # prompt forward, full sequence
    prompt_last = [h[0, -1, :].detach().float().cpu().numpy() for h in hs0]
    first_tok = None
    if len(out.hidden_states) > 1:
        hs1 = out.hidden_states[1]      # state after first generated token
        first_tok = [h[0, 0, :].detach().float().cpu().numpy() for h in hs1]
    return text, np.stack(prompt_last), np.stack(first_tok) if first_tok is not None else None


def forward_states(model, tok, prompts, batch_size=8):
    """Batched prompt-final per-layer hidden states."""
    out_states = None
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        hs = torch.stack([h[torch.arange(len(batch)), last_idx] for h in out.hidden_states],
                         dim=1).float().cpu().numpy()  # (B, L, D)
        out_states = hs if out_states is None else np.concatenate([out_states, hs], axis=0)
    return out_states


def probe_aucs(X, y, layers, test_size=0.3):
    """Per-layer logistic-regression AUROC. X: (N, L, D), y: binary labels."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=0, stratify=y)
    aucs = []
    for L in layers:
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(X[tr, L], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te, L])))
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n-multiple", type=int, default=150)
    ap.add_argument("--n-irrel", type=int, default=150)
    ap.add_argument("--n-live-irrel", type=int, default=60)
    ap.add_argument("--n-live-rel", type=int, default=40)
    ap.add_argument("--n-simple", type=int, default=100)
    ap.add_argument("--n-sampled", type=int, default=150)
    ap.add_argument("--n-steer-tool", type=int, default=80)
    ap.add_argument("--n-steer-ground", type=int, default=60)
    ap.add_argument("--n-corrupt", type=int, default=40)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    rng = random.Random(0)

    print("downloading BFCL v3 files...", flush=True)
    files = {
        "multiple": load_bfcl("BFCL_v3_multiple.json"),
        "irrel": load_bfcl("BFCL_v3_irrelevance.json"),
        "live_irrel": load_bfcl("BFCL_v3_live_irrelevance.json"),
        "live_rel": load_bfcl("BFCL_v3_live_relevance.json"),
        "simple": load_bfcl("BFCL_v3_simple.json"),
    }
    sets = {}
    counts = {"multiple": args.n_multiple, "irrel": args.n_irrel,
              "live_irrel": args.n_live_irrel, "live_rel": args.n_live_rel,
              "simple": args.n_simple}
    for key, items in files.items():
        sets[key] = rng.sample(items, min(counts[key], len(items)))

    tok, model = load_model()
    print(f"[{time.time()-t0:.0f}s] model loaded: {args.model}", flush=True)

    # ---- Phase 1: generation + labeling (H1) ----
    t1 = time.time()
    queries = []  # dict: prompt, registry, gts, category, item
    for key, items in sets.items():
        for it in items:
            tools = item_tools(it)
            registry = [t["name"] for t in tools]
            gts = gt_names(it)
            prompt = build_prompt(tok, tools, item_question(it))
            queries.append({"prompt": prompt, "registry": registry, "gts": gts,
                            "category": key, "item_id": it.get("id"), "item": it})

    prompt_states = []
    first_tok_states = []
    labels = []
    calls_list = []
    outputs = []
    for qi, q in enumerate(queries):
        text, ps, fs = gen(model, tok, q["prompt"], max_new_tokens=220)
        lab, calls = label_output(text, q["registry"], q["gts"])
        labels.append(lab)
        calls_list.append(calls)
        outputs.append({"qid": qi, "item_id": q["item_id"], "category": q["category"],
                        "registry": q["registry"], "raw": text, "calls": calls,
                        "label": lab})
        if qi < args.verbose:
            print(f"--- verbose [{qi}] cat={q['category']} registry={q['registry']} "
                  f"label={lab} calls={calls}\nRAW: {text[:400]}\n", flush=True)
        prompt_states.append(ps)
        if fs is not None:
            first_tok_states.append(fs)
        else:
            first_tok_states.append(np.zeros_like(ps))
    P = np.stack(prompt_states)            # (N, L, D)
    F = np.stack(first_tok_states) if first_tok_states else None

    rates = {}
    for key in counts:
        idx = [i for i, q in enumerate(queries) if q["category"] == key]
        labs = [labels[i] for i in idx]
        rates[key] = {
            "n": len(idx),
            "fabricated": labs.count("fabricated") / max(1, len(idx)),
            "no_call": labs.count("no_call") / max(1, len(idx)),
            "valid": (labs.count("correct") + labs.count("wrong_tool")) / max(1, len(idx)),
        }
    total_fab = labels.count("fabricated") / max(1, len(labels))
    print(f"[{time.time()-t1:.0f}s] H1 greedy: total_fabricated={total_fab:.3f} "
          f"{json.dumps(rates)}", flush=True)

    # sampled subset (fabrication-prone categories)
    t1b = time.time()
    sampled_pool = [q for q in queries if q["category"] in ("irrel", "live_irrel", "live_rel")]
    sampled_pool = rng.sample(sampled_pool, min(args.n_sampled, len(sampled_pool)))
    s_labels = []
    for q in sampled_pool:
        text, _, _ = gen(model, tok, q["prompt"], max_new_tokens=220, do_sample=True,
                         temperature=1.0)
        lab, _ = label_output(text, q["registry"], q["gts"])
        s_labels.append(lab)
    sampled_fab = s_labels.count("fabricated") / max(1, len(s_labels))
    print(f"[{time.time()-t1b:.0f}s] H1 sampled(temp=1): fabricated={sampled_fab:.3f} "
          f"no_call={s_labels.count('no_call')/max(1,len(s_labels)):.3f} "
          f"(n={len(s_labels)})", flush=True)

    def save_partial(p):
        try:
            with open(os.path.join(args.out, "partial_results.json"), "w") as f:
                json.dump(p, f, indent=2)
        except Exception:
            pass

    partial = {"h1_rates_greedy": rates,
               "h1_fabricated_total_greedy": float(total_fab),
               "h1_fabricated_sampled": float(sampled_fab)}
    save_partial(partial)
    with open(os.path.join(args.out, "outputs.jsonl"), "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

    # ---- Phase 2: probes (H2) ----
    # hidden_states layout: [0]=embed, [1..N]=layers 0..N-1, [N+1]=final norm.
    # Probe only the real layers; state idx = probe idx, model layer = probe idx - 1.
    t2 = time.time()
    probe_layers = list(range(1, len(model.model.layers) + 1))
    y_fab = np.array([1 if l == "fabricated" else 0 for l in labels
                      if l in ("fabricated", "correct", "wrong_tool")])
    idx_fab = [i for i, l in enumerate(labels) if l in ("fabricated", "correct", "wrong_tool")]
    if y_fab.sum() >= 4 and (len(y_fab) - y_fab.sum()) >= 4:
        Xf = P[idx_fab]
        auc_fab_prompt = probe_aucs(Xf, y_fab, probe_layers)
        best_fab = probe_layers[int(np.argmax(auc_fab_prompt))]
        if F is not None:
            Xf1 = F[idx_fab]
            auc_fab_first = probe_aucs(Xf1, y_fab, probe_layers)
            best_fab_first = probe_layers[int(np.argmax(auc_fab_first))]
        else:
            auc_fab_first, best_fab_first = None, None
    else:
        auc_fab_prompt, best_fab, auc_fab_first, best_fab_first = None, None, None, None

    y_call = np.array([1 if l in ("fabricated", "correct", "wrong_tool") else 0
                       for l in labels])
    if y_call.sum() >= 4 and (len(y_call) - y_call.sum()) >= 4:
        auc_call = probe_aucs(P, y_call, probe_layers)
        best_call = probe_layers[int(np.argmax(auc_call))]
    else:
        auc_call, best_call = None, None
    print(f"[{time.time()-t2:.0f}s] H2 probes: "
          + (f"fab-vs-valid best L={best_fab} (model layer {best_fab-1}) "
             f"AUROC={auc_fab_prompt[probe_layers.index(best_fab)]:.3f} (prompt-final)"
             if auc_fab_prompt is not None else "fab-vs-valid skipped (class imbalance)")
          + " | "
          + (f"call-vs-nocall best L={best_call} (model layer {best_call-1}) "
             f"AUROC={auc_call[probe_layers.index(best_call)]:.3f}"
             if auc_call is not None else "call-vs-nocall skipped (class imbalance)"),
          flush=True)
    partial["h2_probe_fab_vs_valid"] = ({"auc_by_layer": [float(a) for a in auc_fab_prompt],
                                         "best_probe_layer": int(best_fab),
                                         "best_model_layer": int(best_fab) - 1,
                                         "best_auc": float(auc_fab_prompt[probe_layers.index(best_fab)])}
                                        if auc_fab_prompt is not None else None)
    partial["h2_probe_call_vs_nocall"] = ({"best_probe_layer": int(best_call),
                                           "best_model_layer": int(best_call) - 1,
                                           "best_auc": float(auc_call[probe_layers.index(best_call)])}
                                          if auc_call is not None else None)
    save_partial(partial)

    # ---- Phase 3: steering (H3) ----
    t3 = time.time()
    call_idx = [i for i, l in enumerate(labels) if l in ("fabricated", "correct", "wrong_tool")]
    nocall_idx = [i for i, l in enumerate(labels) if l == "no_call"]
    steer_tool = {}
    if best_call is None or not call_idx or not nocall_idx:
        print(f"[{time.time()-t3:.0f}s] H3a toolness steering: SKIPPED "
              f"(no call/nocall contrast)", flush=True)
    else:
        mean_call = P[call_idx, best_call].mean(axis=0)
        mean_nocall = P[nocall_idx, best_call].mean(axis=0)
        tool_dir = mean_call - mean_nocall
        tool_dir = tool_dir / (np.linalg.norm(tool_dir) + 1e-8)

        steer_pool = [q for q in queries if q["category"] in ("irrel", "live_irrel")]
        steer_pool = rng.sample(steer_pool, min(args.n_steer_tool, len(steer_pool)))
        for alpha in (0.0, 1.0, 2.0):
            labs = []
            for q in steer_pool:
                text, _, _ = gen(model, tok, q["prompt"], max_new_tokens=220,
                                 steer_layer=best_call - 1, steer_dir=tool_dir, steer_alpha=alpha)
                lab, _ = label_output(text, q["registry"], q["gts"])
                labs.append(lab)
            steer_tool[str(alpha)] = {
                "call_rate": sum(1 for l in labs if l in ("fabricated", "correct", "wrong_tool"))
                             / len(labs),
                "fabricated_rate": labs.count("fabricated") / len(labs),
            }
        print(f"[{time.time()-t3:.0f}s] H3a toolness steering: {json.dumps(steer_tool)}", flush=True)
    partial["h3a_steer_toolness"] = steer_tool
    save_partial(partial)

    # grounding steering: push valid-call queries toward the "fabricated" side
    t3b = time.time()
    steer_ground = {}
    if auc_fab_prompt is not None:
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(P[idx_fab, best_fab], y_fab)
        w = clf.coef_[0]
        w = w / (np.linalg.norm(w) + 1e-8)
        correct_idx = [i for i, l in enumerate(labels) if l == "correct"]
        pool = rng.sample(correct_idx, min(args.n_steer_ground, len(correct_idx)))
        for alpha in (0.0, 1.0, 2.0):
            labs = []
            for i in pool:
                q = queries[i]
                text, _, _ = gen(model, tok, q["prompt"], max_new_tokens=220,
                                 steer_layer=best_fab - 1, steer_dir=w, steer_alpha=alpha)
                lab, _ = label_output(text, q["registry"], q["gts"])
                labs.append(lab)
            steer_ground[str(alpha)] = {
                "fabricated_rate": labs.count("fabricated") / len(labs),
                "valid_rate": sum(1 for l in labs if l in ("correct", "wrong_tool")) / len(labs),
            }
        print(f"[{time.time()-t3b:.0f}s] H3b grounding steering: {json.dumps(steer_ground)}",
              flush=True)
    else:
        print("H3b skipped: too few fabricated samples for a grounding probe", flush=True)
    partial["h3b_steer_grounding"] = steer_ground
    save_partial(partial)

    # ---- Phase 4: registry corruption (H4) ----
    t4 = time.time()
    correct_pool = [i for i, l in enumerate(labels) if l == "correct"]
    corrupt_pool = rng.sample(correct_pool, min(args.n_corrupt, len(correct_pool)))
    corrupt_labs = []
    score_shifts = []
    if auc_fab_prompt is not None and corrupt_pool:
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(P[idx_fab, best_fab], y_fab)
        w = clf.coef_[0]
        w = w / (np.linalg.norm(w) + 1e-8)
        prompts_intact, prompts_corrupt = [], []
        for i in corrupt_pool:
            q = queries[i]
            prompts_intact.append(q["prompt"])
            # rebuild prompt with the GT tool removed from the registry
            item = q["item"]
            tools = item_tools(item)
            gts = set(gt_names(item))
            reduced = [t for t in tools if t["name"] not in gts]
            prompts_corrupt.append(build_prompt(tok, reduced, item_question(item)))
            registry = [t["name"] for t in reduced]
            text, _, _ = gen(model, tok, prompts_corrupt[-1], max_new_tokens=220)
            lab, _ = label_output(text, registry, [])
            corrupt_labs.append(lab)
        Si = forward_states(model, tok, prompts_intact, batch_size=8)[:, best_fab, :]
        Sc = forward_states(model, tok, prompts_corrupt, batch_size=8)[:, best_fab, :]
        scores_intact = Si @ w
        scores_corrupt = Sc @ w
        score_shifts = (scores_corrupt - scores_intact).tolist()
        fab_corrupt = corrupt_labs.count("fabricated") / max(1, len(corrupt_labs))
        print(f"[{time.time()-t4:.0f}s] H4 registry corruption: fabricated={fab_corrupt:.3f} "
              f"mean probe-score shift={np.mean(score_shifts):+.3f} (n={len(corrupt_labs)})",
              flush=True)
    else:
        fab_corrupt = None
    partial["h4_corruption"] = {"fabricated_rate": float(fab_corrupt) if fab_corrupt is not None
                                else None,
                                "probe_score_shift_mean": float(np.mean(score_shifts))
                                if score_shifts else None,
                                "n": len(corrupt_labs)}
    save_partial(partial)

    summary = {
        "model": args.model,
        "n_queries": len(queries),
        "h1_rates_greedy": rates,
        "h1_fabricated_total_greedy": float(total_fab),
        "h1_fabricated_sampled": float(sampled_fab),
        "h2_probe_fab_vs_valid": {"auc_by_layer": [float(a) for a in auc_fab_prompt],
                                  "best_probe_layer": int(best_fab),
                                  "best_model_layer": int(best_fab) - 1,
                                  "best_auc": float(auc_fab_prompt[probe_layers.index(best_fab)])}
        if auc_fab_prompt else None,
        "h2_probe_fab_first_token": {"auc_by_layer": [float(a) for a in auc_fab_first],
                                     "best_probe_layer": int(best_fab_first),
                                     "best_model_layer": int(best_fab_first) - 1,
                                     "best_auc": float(auc_fab_first[probe_layers.index(best_fab_first)])}
        if auc_fab_first else None,
        "h2_probe_call_vs_nocall": ({"best_probe_layer": int(best_call),
                                     "best_model_layer": int(best_call) - 1,
                                     "best_auc": float(auc_call[probe_layers.index(best_call)])}
                                    if auc_call is not None else None),
        "h3a_steer_toolness": steer_tool,
        "h3b_steer_grounding": steer_ground,
        "h4_corruption": {"fabricated_rate": float(fab_corrupt) if fab_corrupt is not None
                          else None,
                          "probe_score_shift_mean": float(np.mean(score_shifts))
                          if score_shifts else None,
                          "n": len(corrupt_labs)},
        "wall_seconds": float(time.time() - t0),
    }
    out_path = os.path.join(args.out, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "outputs.jsonl"), "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")
    print("FINAL_SUMMARY " + json.dumps(summary), flush=True)
    print(f"results saved to {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
