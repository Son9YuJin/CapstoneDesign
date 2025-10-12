# baseline_cosine.py
import argparse, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PAIRS = [
    ("dog","puppy"),
    ("car","vehicle"),
    ("big","large"),
    ("happy","glad"),
    ("run","jog"),
    ("movie","film"),
    ("begin","start"),
    ("buy","purchase"),
    ("house","home"),
    ("kid","child"),
]

def anchor_id_of(word, tokenizer):
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if ids else None

def percentile_of(x, samples):
    # return empirical percentile (0~100)
    return float((np.sum(samples <= x) / samples.size) * 100.0)

def main(model_name, n_pairs=200_000, seed=123):
    rng = np.random.default_rng(seed)

    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16", low_cpu_mem_usage=True)

    # embeddings -> float32 for stable cosine
    emb = model.get_input_embeddings().weight.detach().cpu().float().numpy()  # (V, D)

    # collect anchor ids (tokens starting with "Ġ")
    vocab = tok.get_vocab()  #token:id
    inv_vocab = {v:k for k,v in vocab.items()}  #id:token

    anchor_ids = np.array([tid for tid, s in ((i, inv_vocab[i]) for i in range(len(inv_vocab)))
                           if s.startswith("Ġ")], dtype=np.int32)   #white space로 시작하는 토큰의 id

    print(f"Whitespace tokens: {len(anchor_ids)} / Vocab: {len(vocab)}")

    # anchor embeddings (L2-normalize for cosine = dot)
    A = emb[anchor_ids]              # (Na, D)  white space로 시작하느 id개수, hidden(2560)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12 # unit vectors

    # sample random pairs of distinct anchors
    Na = len(anchor_ids) # 난수 최대범위
    i = rng.integers(0, Na, size=n_pairs, dtype=np.int32)  #테스트할 pair 개수
    j = rng.integers(0, Na, size=n_pairs, dtype=np.int32)
    mask_eq = (i == j)
    while np.any(mask_eq):
        # resample where i==j
        j[mask_eq] = rng.integers(0, Na, size=np.sum(mask_eq), dtype=np.int32)
        mask_eq = (i == j)

    # cosine = dot(u_i, u_j)
    cos_rand = np.sum(A[i] * A[j], axis=1)  # shape (n_pairs,)  elementwise mul -> sum = dot product
    mu, sigma = float(cos_rand.mean()), float(cos_rand.std())   #np.float32 ->float
    p10, p50, p90, p95, p99 = np.percentile(cos_rand, [10,50,90,95,99])

    print("\n[Random pair cosine baseline]")
    print(f"n_pairs={n_pairs}")
    print(f"mean={mu:.4f}  std={sigma:.4f}")
    print(f"p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  p99={p99:.4f}")

    # evaluate your synonym pairs (anchor↔anchor cosine + percentile + z)
    for a,b in PAIRS:
        aid = anchor_id_of(a, tok)  #whitespace token id
        bid = anchor_id_of(b, tok)
        if aid is None or bid is None:
            print(f"{a:>10s}  {b:>10s}  ->  not found (unexpected)")
            continue
        ua = emb[aid]; ua = ua / (np.linalg.norm(ua) + 1e-12)
        ub = emb[bid]; ub = ub / (np.linalg.norm(ub) + 1e-12)
        c = float(np.dot(ua, ub))
        perc = percentile_of(c, cos_rand)
        z = (c - mu) / (sigma + 1e-12)
        ta = tok.convert_ids_to_tokens([aid])[0]
        tb = tok.convert_ids_to_tokens([bid])[0]
        print(f"{a:>10s}  {b:>10s}  cos={c:>7.4f}  perc={perc:5.1f}%  z={z:>6.2f}  ({ta} , {tb})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-2.7b")
    ap.add_argument("--pairs", type=int, default=100000, help="number of random anchor pairs")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    main(args.model, n_pairs=args.pairs, seed=args.seed)
