# attack_substitute_param.py
"""
Safe 1:1 word substitution with adjustable per-document and global percent controls.

Usage examples:
# per-doc 10% (document-level, single fixed percent for all docs)
python attack_substitute_param.py --input dataset/clustering_wm.csv --output dataset/clustering_wm_attack.csv --per-doc-percent 10

# per-doc 10% + global 5% (corpus-level cap)
python attack_substitute_param.py --input dataset/clustering_wm.csv --output dataset/clustering_wm_attack.csv --per-doc-percent 10 --global-percent 5

# per-document column (override per row) — values are percentages in [0,100]
python attack_substitute_param.py --input dataset/clustering_wm.csv --output dataset/clustering_wm_attack.csv ^
  --per-doc-percent-col attack_pct --default-per-doc-percent 8

Notes:
- If --per-doc-percent-col is provided, each row's percentage comes from that column.
  Blank/invalid entries fall back to --default-per-doc-percent (or --per-doc-percent if given, or 10%).
- Percent values are clamped to [0, 100].
"""
import re
import argparse
import hashlib
import random
import math
import pandas as pd
from tqdm import tqdm
import spacy
from nltk.corpus import wordnet as wn

# ------------------ Config / Defaults ------------------
TEXT_COL = "generated_text_clean"
NEW_COL = "generated_text_attack"
LOGMAP = "dataset/substitution_map.csv"

# No per-document hard cap; percent-only control
DEFAULT_SEED = 42
DEFAULT_FALLBACK_PERCENT = 10.0

# stopwords (extend as needed)
STOP_WORDS = set((
    "the","a","an","and","or","but","if","so","to","of","in","on","for","at","by",
    "with","from","as","is","are","was","were","be","been","being","that","this",
    "it","its","they","them","their","we","our","you","your","he","she","his","her",
    "i","me","my","mine","us","not","no","yes","do","does","did","have","has","had",
))

PROTECT_ENT_TYPES = set([
    "PERSON","ORG","GPE","LOC","NORP","FAC","EVENT","WORK_OF_ART","LAW","LANGUAGE",
    "PRODUCT","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
])

SPACY2WN = {
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
}

# ------------------ Helpers ------------------
def clean_candidate(w):
    if not w: return None
    # allow letters only (no hyphens, no spaces)
    if not re.fullmatch(r"[A-Za-z]+", w):
        return None
    if len(w) < 3:
        return None
    return w

def preserve_case(dst, src):
    if src.isupper(): return dst.upper()
    if src.istitle(): return dst.title()
    if src.islower(): return dst.lower()
    return dst

def pick_index(options_len, key):
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % options_len

def get_synonym_candidates(token_text, wn_pos):
    lemma = token_text.lower()
    cands = set()
    for syn in wn.synsets(lemma, pos=wn_pos):
        for l in syn.lemmas():
            w = l.name().replace("_","")
            w = clean_candidate(w)
            if not w: continue
            if w.lower() == lemma: continue
            cands.add(w.lower())
    return list(cands)

# ------------------ Document attack ------------------
def attack_doc(text, nlp, per_doc_percent, seed, global_state):
    """
    Returns (new_text, num_changes, change_records)
    global_state: dict used to track and enforce global cap
    per_doc_percent: float in [0,100]
    """
    if not isinstance(text, str) or not text.strip():
        return text, 0, []

    doc = nlp(text)
    tokens = [t for t in doc]
    alpha_tokens = [t for t in tokens if t.is_alpha]
    total_alpha = len(alpha_tokens)
    if total_alpha == 0:
        return text, 0, []

    # compute document budget (number of tokens to change)
    # use ceil so very short docs still get at least one change when percent > 0
    budget_by_percent = int(math.ceil(total_alpha * (per_doc_percent / 100.0))) if per_doc_percent > 0 else 0
    doc_budget = budget_by_percent

    # if global cap is active, reduce doc_budget accordingly
    if global_state.get("global_limit") is not None:
        remaining = global_state["global_limit"] - global_state["global_used"]
        if remaining <= 0 or doc_budget <= 0:
            return text, 0, []
        doc_budget = min(doc_budget, remaining)

    # mark protected entity token indices
    ent_protected = set()
    for ent in doc.ents:
        if ent.label_ in PROTECT_ENT_TYPES:
            for i in range(ent.start, ent.end):
                ent_protected.add(i)

    # collect candidate token indices
    candidates = []
    for i, t in enumerate(tokens):
        if i in ent_protected:
            continue
        if not t.is_alpha:
            continue
        if t.text.lower() in STOP_WORDS:
            continue
        if t.pos_ not in SPACY2WN:
            continue
        if t.like_num:
            continue
        if len(t.text) < 3:
            continue
        candidates.append(i)

    if not candidates or doc_budget <= 0:
        return text, 0, []

    # deterministic selection using seed + doc hash for reproducibility
    changes = []
    new_text = text
    spans = [(t.idx, t.idx + len(t.text)) for t in tokens]

    # ❶ pick doc_budget candidates uniformly at random, ❷ process in original order for safe span math
    rng = random.Random(seed + (hash(text) & 0xffffffff))
    pick_n = min(doc_budget, len(candidates))
    picked = sorted(rng.sample(candidates, pick_n))

    for i in picked:
        tok = tokens[i]
        wn_pos = SPACY2WN.get(tok.pos_)
        if not wn_pos:
            continue

        syns = get_synonym_candidates(tok.text, wn_pos)
        if not syns:
            continue

        # choose one synonym deterministically (stable across runs with same seed)
        key = f"{tok.text.lower()}|{tok.pos_}|{seed}"
        idx_choice = pick_index(len(syns), key)
        rep = syns[idx_choice]
        if not rep or rep.lower() == tok.text.lower():
            continue
        rep = preserve_case(rep, tok.text)

        # compute adjusted spans considering prior changes
        consumed = 0
        for st, en, r, _orig in changes:
            if st < spans[i][0]:
                consumed += (len(r) - (en - st))
        start2 = spans[i][0] + consumed
        end2   = spans[i][1] + consumed

        # final safety check that substring matches token text
        if new_text[start2:end2] != tok.text:
            continue

        # apply replacement
        new_text = new_text[:start2] + rep + new_text[end2:]
        changes.append((spans[i][0], spans[i][1], rep, tok.text))

        # update global usage and early-stop if hitting the global cap
        global_state["global_used"] += 1
        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            break

    # build change records for logging
    records = []
    for st, en, rep, orig in changes:
        records.append({"start": st, "end": en, "orig": orig, "replacement": rep})

    return new_text, len(changes), records

# ------------------ Utilities ------------------
def _coerce_percent(x):
    """Try to parse a per-document percent value from a DataFrame cell."""
    if pd.isna(x):
        return None
    try:
        val = float(x)
        if math.isfinite(val):
            return max(0.0, min(100.0, val))
    except Exception:
        return None
    return None

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="dataset/clustering_wm.csv")
    parser.add_argument("--output", type=str, default="dataset/clustering_wm_attack.csv")

    # Either a single fixed percent OR a per-row column. If both are given, the column overrides per row,
    # and the fixed value acts as the default fallback.
    parser.add_argument("--per-doc-percent", type=float, default=None,
                        help="Percent of alphabetic tokens to replace per document (e.g., 10 for 10%).")
    parser.add_argument("--per-doc-percent-col", type=str, default=None,
                        help="Optional column name containing per-document percent values in [0,100].")
    parser.add_argument("--default-per-doc-percent", type=float, default=None,
                        help="Fallback percent if a row's column is empty/invalid. If not set, falls back to --per-doc-percent or 10%.")

    parser.add_argument("--global-percent", type=float, default=None,
                        help="(Optional) global percent cap across corpus (e.g., 5 for 5%). If set, computes total alphabetic tokens and limits replacements.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # pick fallback percent
    fallback_pct = (
        args.default_per_doc_percent
        if args.default_per_doc_percent is not None
        else (args.per_doc_percent if args.per_doc_percent is not None else DEFAULT_FALLBACK_PERCENT)
    )
    # sanity clamp
    fallback_pct = max(0.0, min(100.0, float(fallback_pct)))

    # load spaCy and data
    print("[INFO] loading spaCy en_core_web_sm (this may take a moment)...")
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(args.input)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column `{TEXT_COL}` not found in {args.input}")

    use_col = args.per_doc_percent_col
    if use_col is None and args.per_doc_percent is None:
        print(f"[WARN] neither --per-doc-percent nor --per-doc-percent-col provided; defaulting to {fallback_pct}% for all docs.")

    texts = df[TEXT_COL].fillna("").tolist()

    # compute global cap if requested
    global_state = {"global_limit": None, "global_used": 0}
    if args.global_percent is not None:
        total_alpha = 0
        print("[INFO] computing total alphabetic token count for global cap...")
        for txt in tqdm(texts, desc="Counting tokens"):
            # crude count via simple regex for performance
            total_alpha += len(re.findall(r"\b[A-Za-z]{3,}\b", txt))
        global_limit = int(total_alpha * (args.global_percent / 100.0))
        global_state["global_limit"] = max(0, global_limit)
        print(f"[INFO] global alphabetic tokens={total_alpha}, global replacement limit={global_state['global_limit']}")

    attacked = []
    all_logs = []

    # resolve per-document percents (vectorized first pass if a column is specified)
    percents = None
    if use_col is not None and use_col in df.columns:
        percents = df[use_col].apply(_coerce_percent).tolist()
    elif use_col is not None and use_col not in df.columns:
        print(f"[WARN] column `{use_col}` not found; falling back to default/fixed percent {fallback_pct}% for all docs.")

    for i, txt in enumerate(tqdm(texts, desc="Substituting")):
        # decide this doc's percent
        if percents is not None:
            p = percents[i]
            per_doc_pct = p if p is not None else fallback_pct
        else:
            per_doc_pct = fallback_pct

        new_text, num_changes, records = attack_doc(
            txt, nlp, per_doc_pct, args.seed, global_state
        )
        attacked.append(new_text)

        if records:
            for r in records:
                r.update({"row_id": i, "percent_used": per_doc_pct})
            all_logs.extend(records)

        # if global cap reached, after finishing current doc we stop processing further docs
        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            print("[INFO] global replacement limit reached; stopping further substitutions.")
            # append remaining original texts unchanged
            for j in range(i+1, len(texts)):
                attacked.append(texts[j])
            break

    df[NEW_COL] = attacked
    df.to_csv(args.output, index=False)
    print(f"[DONE] saved attacked CSV: {args.output}")

    if all_logs:
        logdf = pd.DataFrame(all_logs)
        logdf.to_csv(LOGMAP, index=False)
        print(f"[LOG] substitution map saved: {LOGMAP} (total replacements={len(all_logs)})")

if __name__ == "__main__":
    main()
