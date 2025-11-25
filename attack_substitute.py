# attack_substitute_param.py
"""
Safe 1:1 word substitution with adjustable per-document and global percent controls,
+ watermark z-score computation on attacked text.

Usage examples (옵션 안 주면 맨 위 CONFIG 값 사용):

python attack_substitute_param.py
python attack_substitute_param.py --per-doc-percent 50
python attack_substitute_param.py --input 다른거.csv --output 결과.csv
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

import torch
from transformers import AutoTokenizer
from watermark_processor import WatermarkDetector

# ============================================================
#                     HIGH-LEVEL CONFIG
#        (여기만 수정해도 기본 세팅이 싹 바뀜)
# ============================================================

# --- 기본 입출력 파일 ---
CFG_INPUT = "dataset/experiment_data_wm.csv"
CFG_OUTPUT = "dataset/experiment_wm_p10.csv"

# --- 치환 비율 관련 ---
#  예: 50.0 으로 두면 각 문서 알파벳 토큰의 50% 치환
#  None 으로 두면 CLI 인자나 DEFAULT_FALLBACK_PERCENT 사용
CFG_PER_DOC_PERCENT = 10.0          # e.g. 50.0 or None
CFG_PER_DOC_PERCENT_COL = None      # e.g. "attack_pct" or None
CFG_DEFAULT_PER_DOC_PERCENT = None  # e.g. 10.0 or None
CFG_GLOBAL_PERCENT = None           # e.g. 5.0 (코퍼스 전체 5%) or None

# --- 워터마킹 검출 관련 ---
CFG_MODEL_NAME = "facebook/opt-350m"
CFG_DETECT_GAMMA = 0.25
CFG_DETECT_DELTA = 3.0
CFG_SEEDING_SCHEME = "simple_1"
CFG_Z_COL_NAME = "z_attack"
# "cuda", "cpu", 또는 None (자동 선택)
CFG_DEVICE = None

# ============================================================
#                   OTHER CONSTANT CONFIG
# ============================================================

TEXT_COL = "generated_text_clean"
NEW_COL = "generated_text_attack"
LOGMAP = "dataset/substitution_map.csv"

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

# WordNet 동의어 중 의미가 너무 튀는 것들(특히 마약 관련 등) 블랙리스트
BAD_LEMMAS = set([
    "amphetamine", "methedrine", "pep", "pep_pill", "upper", "uppers",
    "downer", "downers",
    # 필요하면 여기 계속 추가
])

# ------------------ Helpers ------------------
def clean_candidate(w):
    if not w:
        return None
    # allow letters only (no hyphens, no spaces)
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

def get_synonym_candidates(token_text, wn_pos):
    """
    의미 덜 깨지는 동의어 후보 리스트:
    - WordNet의 가장 빈도 높은 sense(첫 synset)만 사용
    - '_'나 '-'가 들어가는 복합어는 제외 (overseas_telegram, applied_science 등)
    - BAD_LEMMAS(amphetamine 등)는 제외
    """
    lemma = token_text.lower()
    synsets = wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        return []

    cands = set()

    # 가장 빈도 높은 sense 한 개만 사용
    main_syn = synsets[0]

    for l in main_syn.lemmas():
        name = l.name()  # 예: "velocity", "overseas_telegram", "amphetamine" ...

        # 복합어/이상한 조합은 제외
        if "_" in name or "-" in name:
            continue

        w = name
        w = clean_candidate(w)
        if not w:
            continue

        # 자기 자신은 제외
        if w.lower() == lemma:
            continue

        # 블랙리스트 단어 제외
        if w.lower() in BAD_LEMMAS:
            continue

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

        # choose one synonym deterministically
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

    # 기본값을 CONFIG에서 가져옴
    parser.add_argument("--input", type=str, default=CFG_INPUT)
    parser.add_argument("--output", type=str, default=CFG_OUTPUT)

    # Either a single fixed percent OR a per-row column. If both are given, the column overrides per row,
    # and the fixed value acts as the default fallback.
    parser.add_argument("--per-doc-percent", type=float, default=CFG_PER_DOC_PERCENT,
                        help="Percent of alphabetic tokens to replace per document (e.g., 10 for 10%).")
    parser.add_argument("--per-doc-percent-col", type=str, default=CFG_PER_DOC_PERCENT_COL,
                        help="Optional column name containing per-document percent values in [0,100].")
    parser.add_argument("--default-per-doc-percent", type=float, default=CFG_DEFAULT_PER_DOC_PERCENT,
                        help="Fallback percent if a row's column is empty/invalid. If not set, falls back to --per-doc-percent or 10%.")

    parser.add_argument("--global-percent", type=float, default=CFG_GLOBAL_PERCENT,
                        help="(Optional) global percent cap across corpus (e.g., 5 for 5%). If set, computes total alphabetic tokens and limits replacements.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # ---------- Watermark detector 관련 옵션 ----------
    parser.add_argument("--model-name", type=str, default=CFG_MODEL_NAME,
                        help="Tokenizer를 불러올 HF 모델 이름 (워터마킹에 사용한 모델과 맞추는 걸 추천).")
    parser.add_argument("--detect-gamma", type=float, default=CFG_DETECT_GAMMA,
                        help="WatermarkDetector에서 사용할 gamma.")
    parser.add_argument("--detect-delta", type=float, default=CFG_DETECT_DELTA,
                        help="WatermarkDetector에서 사용할 delta (사용하지 않으면 무시될 수 있음).")
    parser.add_argument("--seeding-scheme", type=str, default=CFG_SEEDING_SCHEME,
                        help="WatermarkDetector seeding scheme (보통 'simple_1').")
    parser.add_argument("--z-col-name", type=str, default=CFG_Z_COL_NAME,
                        help="치환 후 z-score를 저장할 컬럼 이름.")
    parser.add_argument("--device", type=str, default=CFG_DEVICE,
                        help="검출에 사용할 디바이스 (예: 'cuda', 'cpu'). 지정 안 하면 자동 선택.")
    # -------------------------------------------------

    args = parser.parse_args()

    # pick fallback percent
    fallback_pct = (
        args.default_per_doc_percent
        if args.default_per_doc_percent is not None
        else (args.per_doc_percent if args.per_doc_percent is not None else DEFAULT_FALLBACK_PERCENT)
    )
    fallback_pct = max(0.0, min(100.0, float(fallback_pct)))

    # load spaCy and data
    print("[INFO] loading spaCy en_core_web_sm (this may take a moment)...")
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(args.input)
    # df = df.head(100)
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
            total_alpha += len(re.findall(r"\b[A-Za-z]{3,}\b", txt))
        global_limit = int(total_alpha * (args.global_percent / 100.0))
        global_state["global_limit"] = max(0, global_limit)
        print(f"[INFO] global alphabetic tokens={total_alpha}, global replacement limit={global_state['global_limit']}")

    attacked = []
    all_logs = []

    # resolve per-document percents
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

        # if global cap reached, stop for remaining docs
        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            print("[INFO] global replacement limit reached; stopping further substitutions.")
            for j in range(i+1, len(texts)):
                attacked.append(texts[j])
            break

    # attacked 텍스트 컬럼 추가
    df[NEW_COL] = attacked

    # ------------------ 치환 후 z-score 계산 ------------------
    print("[INFO] loading tokenizer & WatermarkDetector for z-score computation...")

    # device 선택
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    vocab_ids = list(tokenizer.get_vocab().values())

    # WatermarkDetector 시그니처가 레포마다 달 수 있어서 delta 유무 둘 다 시도
    try:
        detector = WatermarkDetector(
            vocab=vocab_ids,
            gamma=args.detect_gamma,
            delta=args.detect_delta,
            seeding_scheme=args.seeding_scheme,
            tokenizer=tokenizer,
            device=device,
        )
    except TypeError:
        detector = WatermarkDetector(
            vocab=vocab_ids,
            gamma=args.detect_gamma,
            seeding_scheme=args.seeding_scheme,
            tokenizer=tokenizer,
            device=device,
        )

    z_scores = []
    texts_for_z = df[NEW_COL].fillna("").tolist()

    print("[INFO] computing watermark z-scores on attacked text...")
    for txt in tqdm(texts_for_z, desc="Computing z_attack"):
        t = txt.strip()
        if not t:
            z_scores.append(float("nan"))
            continue
        detect_dict = detector.detect(
            text=t,
            return_prediction=False,
            return_scores=True,
        )
        z = detect_dict.get("z_score", float("nan"))
        z_scores.append(z)

    df[args.z_col_name] = z_scores
    # ---------------------------------------------------------

    df.to_csv(args.output, index=False)
    print(f"[DONE] saved attacked CSV: {args.output}번")

    if all_logs:
        logdf = pd.DataFrame(all_logs)
        logdf.to_csv(LOGMAP, index=False)
        print(f"[LOG] substitution map saved: {LOGMAP} (total replacements={len(all_logs)})")

if __name__ == "__main__":
    main()
