# attack_substitute.py (K-Cluster Aware Version)
"""
Safe 1:1 word substitution - MODIFIED to use cluster_data.npz for synonyms,
and to control the ratio of synonym vs non-synonym replacements.
"""
import re
import argparse
import hashlib
import random
import math
import pandas as pd
from tqdm import tqdm
import spacy
import numpy as np
from collections import defaultdict
import os

# ------------------ Config / Defaults ------------------
TEXT_COL = "generated_text_clean"
NEW_COL = "generated_text_attack"
LOGMAP = "dataset/substitution_map.csv"
CLUSTER_DATA_PATH = "cluster_data.npz"

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

# ------------------ Cluster data load ------------------
def load_cluster_maps():
    print(f"[INFO] Attack script loading cluster data from: {CLUSTER_DATA_PATH}")
    try:
        data = np.load(CLUSTER_DATA_PATH, allow_pickle=True)
        tokens = data['tokens']
        labels = data['labels']
        
        # 1. 단어 -> 클러스터 ID 맵 (소문자로 저장)
        token_to_cluster = {}
        for token, label in zip(tokens, labels):
            token_to_cluster[str(token).lower()] = label
            
        # 2. 클러스터 ID -> 단어 리스트 맵
        cluster_to_tokens = defaultdict(list)
        for token, label in zip(tokens, labels):
            cluster_to_tokens[label].append(str(token).lower())
        
        # 3. 전체 단어 리스트 (비유의어 후보용)
        all_words = sorted(set(str(t).lower() for t in tokens))
            
        print(f"[INFO] Attack script loaded {len(token_to_cluster)} tokens into {len(cluster_to_tokens)} clusters.")
        return token_to_cluster, cluster_to_tokens, all_words
        
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] in Attack Script: '{CLUSTER_DATA_PATH}' not found.")
        print("-> 'fix_npz.py'를 실행해서 이 파일을 생성했는지 확인하세요.")
        exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] in Attack Script: Failed to load .npz file: {e}")
        exit(1)

# 전역 변수로 클러스터 사전 로드
TOKEN_TO_CLUSTER, CLUSTER_TO_TOKENS, ALL_WORDS = load_cluster_maps()

# ------------------ Helpers ------------------
def clean_candidate(w):
    if not w:
        return None
    if not re.fullmatch(r"[A-Za-z]+", w):
        return None
    if len(w) < 3:
        return None
    return w

def preserve_case(dst, src):
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst.title()
    if src.islower():
        return dst.lower()
    return dst

def pick_index(options_len, key):
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % options_len

# ------------------ Synonym / Non-synonym candidates ------------------
def get_synonym_candidates(token_text, wn_pos_ignored):
    """
    cluster_data.npz에서 '같은 클러스터'에 속한 유의어 후보를 찾는다.
    """
    lemma = token_text.lower()
    
    cluster_id = TOKEN_TO_CLUSTER.get(lemma)
    if cluster_id is None:
        return []
        
    candidates = CLUSTER_TO_TOKENS.get(cluster_id, [])
    
    syns = set()
    for w in candidates:
        w_clean = clean_candidate(w)
        if not w_clean:
            continue
        if w_clean.lower() == lemma:
            continue
        syns.add(w_clean.lower())
        
    return list(syns)

def get_nonsynonym_candidate(token_text, seed, max_tries=20):
    """
    '다른 클러스터'에 속한 비유의어 후보를 전체 단어 리스트에서 선택.
    seed + 토큰을 기반으로 결정적(deterministic) 선택.
    """
    lemma = token_text.lower()
    cluster_id = TOKEN_TO_CLUSTER.get(lemma)
    if cluster_id is None:
        return None
    
    for k in range(max_tries):
        key = f"RAND|{lemma}|{seed}|{k}"
        idx = pick_index(len(ALL_WORDS), key)
        cand = ALL_WORDS[idx]
        
        # 자기 자신 제외
        if cand == lemma:
            continue
        # 같은 클러스터(유의어) 제외
        if TOKEN_TO_CLUSTER.get(cand) == cluster_id:
            continue
        
        w_clean = clean_candidate(cand)
        if not w_clean:
            continue
        
        return w_clean.lower()
    
    return None

# ------------------ Document attack ------------------
def attack_doc(text, nlp, per_doc_percent, seed, synonym_ratio, global_state):
    """
    Returns (new_text, num_changes, change_records)
    global_state: dict used to track and enforce global cap
    per_doc_percent: float in [0,100]
    synonym_ratio: among attacked tokens, fraction to use synonyms (0.0 ~ 1.0)
    """
    if not isinstance(text, str) or not text.strip():
        return text, 0, []

    # synonym_ratio 정규화
    if synonym_ratio is None:
        synonym_ratio = 1.0
    try:
        synonym_ratio = float(synonym_ratio)
    except Exception:
        synonym_ratio = 1.0
    synonym_ratio = max(0.0, min(1.0, synonym_ratio))

    doc = nlp(text)
    tokens = [t for t in doc]
    alpha_tokens = [t for t in tokens if t.is_alpha]
    total_alpha = len(alpha_tokens)
    if total_alpha == 0:
        return text, 0, []

    budget_by_percent = int(math.ceil(total_alpha * (per_doc_percent / 100.0))) if per_doc_percent > 0 else 0
    doc_budget = budget_by_percent

    if global_state.get("global_limit") is not None:
        remaining = global_state["global_limit"] - global_state["global_used"]
        if remaining <= 0 or doc_budget <= 0:
            return text, 0, []
        doc_budget = min(doc_budget, remaining)

    ent_protected = set()
    for ent in doc.ents:
        if ent.label_ in PROTECT_ENT_TYPES:
            for i in range(ent.start, ent.end):
                ent_protected.add(i)

    candidates = []
    for i, t in enumerate(tokens):
        if i in ent_protected:
            continue
        if not t.is_alpha:
            continue
        if t.text.lower() in STOP_WORDS:
            continue
        if t.text.lower() not in TOKEN_TO_CLUSTER:
            continue
        if t.like_num:
            continue
        if len(t.text) < 3:
            continue
        candidates.append(i)

    if not candidates or doc_budget <= 0:
        return text, 0, []

    changes = []
    new_text = text
    spans = [(t.idx, t.idx + len(t.text)) for t in tokens]

    rng = random.Random(seed + (hash(text) & 0xffffffff))
    pick_n = min(doc_budget, len(candidates))
    picked = sorted(rng.sample(candidates, pick_n))

    records = []

    for i in picked:
        tok = tokens[i]

        # 이번 토큰을 유의어로 바꿀지, 비유의어로 바꿀지 결정
        use_synonym = (rng.random() < synonym_ratio)

        rep_lemma = None

        if use_synonym:
            # 같은 클러스터 내 유의어 후보
            syns = get_synonym_candidates(tok.text, tok.pos_)
            if syns:
                key = f"{tok.text.lower()}|{tok.pos_}|{seed}"
                idx_choice = pick_index(len(syns), key)
                rep_lemma = syns[idx_choice]
            else:
                # 유의어가 없으면 비유의어 fallback
                rep_lemma = get_nonsynonym_candidate(tok.text, seed)
        else:
            # 다른 클러스터 비유의어 후보
            rep_lemma = get_nonsynonym_candidate(tok.text, seed)
            # 비유의어가 안 나오면 유의어 쪽으로 fallback (선택사항)
            if rep_lemma is None:
                syns = get_synonym_candidates(tok.text, tok.pos_)
                if syns:
                    key = f"{tok.text.lower()}|{tok.pos_}|{seed}|fallback"
                    idx_choice = pick_index(len(syns), key)
                    rep_lemma = syns[idx_choice]

        if not rep_lemma or rep_lemma.lower() == tok.text.lower():
            continue

        rep = preserve_case(rep_lemma, tok.text)

        consumed = 0
        for st, en, r, _orig in changes:
            if st < spans[i][0]:
                consumed += (len(r) - (en - st))
        start2 = spans[i][0] + consumed
        end2   = spans[i][1] + consumed

        if new_text[start2:end2] != tok.text:
            continue

        new_text = new_text[:start2] + rep + new_text[end2:]
        changes.append((spans[i][0], spans[i][1], rep, tok.text))

        records.append({
            "start": spans[i][0],
            "end": spans[i][1],
            "orig": tok.text,
            "replacement": rep,
            "mode": "synonym" if use_synonym else "nonsynonym"
        })

        global_state["global_used"] += 1
        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            break

    return new_text, len(changes), records

# ------------------ Utilities ------------------
def _coerce_percent(x):
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
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--per-doc-percent", type=float, default=None,
                        help="Percent of alphabetic tokens to replace per document (e.g., 10 for 10%).")
    parser.add_argument("--per-doc-percent-col", type=str, default=None,
                        help="Optional column name containing per-document percent values in [0,100].")
    parser.add_argument("--default-per-doc-percent", type=float, default=None,
                        help="Fallback percent if a row's column is empty/invalid. If not set, falls back to --per-doc-percent or 10%.")
    parser.add_argument("--global-percent", type=float, default=None,
                        help="(Optional) global percent cap across corpus (e.g., 5 for 5%).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--synonym-ratio", type=float, default=1.0,
                        help="Among attacked tokens, fraction to replace with synonyms (0.0 ~ 1.0). 1.0 = all synonyms, 0.0 = all non-synonyms.")
    args = parser.parse_args()

    fallback_pct = (
        args.default_per_doc_percent
        if args.default_per_doc_percent is not None
        else (args.per_doc_percent if args.per_doc_percent is not None else DEFAULT_FALLBACK_PERCENT)
    )
    fallback_pct = max(0.0, min(100.0, float(fallback_pct)))

    print("[INFO] loading spaCy en_core_web_sm (this may take a moment)...")
    nlp = spacy.load("en_core_web_sm")
    
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] Input file not found: {args.input}")
        exit(1)
        
    if TEXT_COL not in df.columns:
        print(f"\n[FATAL ERROR] Column `{TEXT_COL}` not found in {args.input}")
        print(f"-> Available columns are: {list(df.columns)}")
        exit(1)

    use_col = args.per_doc_percent_col
    if use_col is None and args.per_doc_percent is None:
        print(f"[WARN] neither --per-doc-percent nor --per-doc-percent-col provided; defaulting to {fallback_pct}% for all docs.")

    texts = df[TEXT_COL].fillna("").tolist()

    global_state = {"global_limit": None, "global_used": 0}
    if args.global_percent is not None:
        total_alpha = 0
        print("[INFO] computing total alphabetic token count for global cap...")
        for txt in tqdm(texts, desc="Counting tokens"):
            total_alpha += len(re.findall(r"\b[A-Za-z]{3,}\b", txt))
        global_limit = int(total_alpha * (args.global_percent / 100.0))
        global_state["global_limit"] = max(0, global_limit)
        print(f"[INFO] global alphabetic tokens={total_alpha}, global replacement limit={global_state['global_limit']}")

    attacked = []
    all_logs = []

    percents = None
    if use_col is not None and use_col in df.columns:
        percents = df[use_col].apply(_coerce_percent).tolist()
    elif use_col is not None and use_col not in df.columns:
        print(f"[WARN] column `{use_col}` not found; falling back to default/fixed percent {fallback_pct}% for all docs.")

    for i, txt in enumerate(tqdm(texts, desc="Substituting")):
        if percents is not None:
            p = percents[i]
            per_doc_pct = p if p is not None else fallback_pct
        else:
            per_doc_pct = fallback_pct

        new_text, num_changes, records = attack_doc(
            txt, nlp, per_doc_pct, args.seed, args.synonym_ratio, global_state
        )
        attacked.append(new_text)

        if records:
            for r in records:
                r.update({"row_id": i, "percent_used": per_doc_pct})
            all_logs.extend(records)

        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            print("[INFO] global replacement limit reached; stopping further substitutions.")
            attacked.extend(texts[i+1:])
            break

    df[NEW_COL] = attacked
    
    # 출력 폴더 자동 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
        
    df.to_csv(args.output, index=False)
    print(f"\n[DONE] saved attacked CSV: {args.output}")

    if all_logs:
        # 로그맵 폴더 자동 생성
        log_dir = os.path.dirname(LOGMAP)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"[INFO] Created log directory: {log_dir}")
            
        logdf = pd.DataFrame(all_logs)
        logdf.to_csv(LOGMAP, index=False)
        print(f"[LOG] substitution map saved: {LOGMAP} (total replacements={len(all_logs)})")

if __name__ == "__main__":
    main()
