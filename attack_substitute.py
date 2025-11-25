# coding=utf-8
"""
[K-Cluster + WordNet 하이브리드 치환 공격 스크립트]

- cluster_data.npz 에서 토큰 클러스터 정보를 읽어서
  * 같은 클러스터 안의 토큰들 → "클러스터 유의어" 후보
  * 다른 클러스터의 토큰들 → "비유의어(통계 교란용)" 후보로 사용.

- 추가로 WordNet 기반 동의어를 함께 사용해서
  비유의어 모드일 때는:
    1) 우선 WordNet 동의어에서 후보를 찾고
    2) 없으면 다른 클러스터에서 후보를 찾는 하이브리드 공격.

- 문서별 퍼센트(per-doc-percent)와 전체 코퍼스 기준 퍼센트(global-percent)를
  동시에 제어할 수 있음.

사용 예시:

1) 모든 문서에서 알파벳 토큰의 10%를 치환, 클러스터 유의어 위주 공격
   (synonym_ratio=1.0 → 거의 전부 같은 클러스터 내에서 교체)

    python attack_substitute.py --input dataset/clustering_wm_gamma0.25_delta4.0_cg0.25.csv --output dataset/clustering_wm_gamma0.25_delta4.0_cg0.25_attack_10pct_1.0.csv --per-doc-percent 10 --synonym-ratio 1.0

2) 유의어 50%, 비유의어(WordNet→다른 클러스터) 50% 섞인 공격

    python attack_substitute.py --input dataset/clustering_wm_gamma0.25_delta4.0_cg0.25.csv --output dataset/clustering_wm_gamma0.25_delta4.0_cg0.25_attack_10pct_0.5.csv --per-doc-percent 10 --synonym-ratio 0.5

3) 비유의어 공격만 (WordNet → 없으면 다른 클러스터)

    python attack_substitute.py --input dataset/clustering_wm_gamma0.25_delta4.0_cg0.25.csv --output dataset/clustering_wm_gamma0.25_delta4.0_cg0.25_attack_10pct_0.csv --per-doc-percent 10 --synonym-ratio 0


주의:
- 처음 한 번은 python에서 다음을 실행해서 WordNet 데이터를 받아야 함.
    >>> import nltk
    >>> nltk.download("wordnet")
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

from nltk.corpus import wordnet as wn  # WordNet 동의어 사전

# ------------------ 기본 설정 ------------------
# 입력 CSV에서 사용할 컬럼 이름
TEXT_COL = "generated_text_clean"      # 원본 텍스트 컬럼
NEW_COL = "generated_text_attack"      # 치환 공격 후 텍스트가 저장될 컬럼

# 치환 로그를 남길 CSV 경로
LOGMAP = "dataset/substitution_map.csv"

# 클러스터 데이터 npz 파일 경로
CLUSTER_DATA_PATH = "cluster_data.npz"

# 랜덤 시드 기본값
DEFAULT_SEED = 42

# per-doc-percent / 컬럼 둘 다 없을 때 사용할 기본 퍼센트 (%)
DEFAULT_FALLBACK_PERCENT = 10.0

# 불필요하거나 바꾸고 싶지 않은 토큰(관사, 전치사 등)
STOP_WORDS = set((
    "the","a","an","and","or","but","if","so","to","of","in","on","for","at","by",
    "with","from","as","is","are","was","were","be","been","being","that","this",
    "it","its","they","them","their","we","our","you","your","he","she","his","her",
    "i","me","my","mine","us","not","no","yes","do","does","did","have","has","had",
))

# spaCy NER에서 보호할 엔터티 타입들 (고유명사, 숫자, 날짜 등)
PROTECT_ENT_TYPES = set([
    "PERSON","ORG","GPE","LOC","NORP","FAC","EVENT","WORK_OF_ART","LAW","LANGUAGE",
    "PRODUCT","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
])

# spaCy 품사를 WordNet 품사로 변환하기 위한 매핑
SPACY2WN = {
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
}

# ------------------ 클러스터 데이터 로드 ------------------
def load_cluster_maps():
    """
    cluster_data.npz 파일에서 클러스터 정보를 읽어서
    - token_to_cluster: 토큰(소문자) → 클러스터 ID
    - cluster_to_tokens: 클러스터 ID → 해당 클러스터에 속한 토큰 리스트
    - all_words: 전체 토큰 리스트 (비유의어 후보 선택용)
    을 생성한다.
    """
    print(f"[INFO] Attack script loading cluster data from: {CLUSTER_DATA_PATH}")
    try:
        data = np.load(CLUSTER_DATA_PATH, allow_pickle=True)
        tokens = data['tokens']
        labels = data['labels']
        
        # 토큰(소문자) → 클러스터 ID
        token_to_cluster = {}
        for token, label in zip(tokens, labels):
            token_to_cluster[str(token).lower()] = label
            
        # 클러스터 ID → 토큰 리스트
        cluster_to_tokens = defaultdict(list)
        for token, label in zip(tokens, labels):
            cluster_to_tokens[label].append(str(token).lower())
        
        # 전체 토큰 리스트 (중복 제거 후 정렬)
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

# 전역 변수로 클러스터 사전 로드 (스크립트 시작 시 한 번만)
TOKEN_TO_CLUSTER, CLUSTER_TO_TOKENS, ALL_WORDS = load_cluster_maps()

# ------------------ 공통 유틸 함수 ------------------
def clean_candidate(w):
    """
    치환 후보로 쓸 수 있는 단어인지 간단히 검사:
    - None / 빈 문자열이면 제외
    - 알파벳(a~z, A~Z)만 허용 (공백, 하이픈 등 제거)
    - 길이 3 미만이면 제외
    """
    if not w:
        return None
    if not re.fullmatch(r"[A-Za-z]+", w):
        return None
    if len(w) < 3:
        return None
    return w

def preserve_case(dst, src):
    """
    src의 대소문자 패턴을 dst에 최대한 맞춰서 반환.
    예) SRC: "Apple" → dst: "fruit" → "Fruit"
        SRC: "APPLE" → "FRUIT"
        SRC: "apple" → "fruit"
    """
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst.title()
    if src.islower():
        return dst.lower()
    return dst

def pick_index(options_len, key):
    """
    문자열 key를 MD5 해시로 변환하여
    0 ~ (options_len-1) 범위의 정수 인덱스를 결정론적으로 생성.
    같은 key면 항상 같은 인덱스를 뽑게 되므로 재현성 확보 가능.
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % options_len

# ------------------ 클러스터 / WordNet 기반 후보 생성 ------------------
def get_synonym_candidates(token_text):
    """
    [클러스터 유의어 후보]
    - cluster_data.npz에서 같은 클러스터에 속한 토큰들을 유의어 후보로 사용.
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
            continue  # 자기 자신은 제외
        syns.add(w_clean.lower())
        
    return list(syns)

def get_nonsynonym_candidate(token_text, seed, max_tries=20):
    """
    [다른 클러스터 비유의어 후보]
    - 전체 토큰 리스트 ALL_WORDS에서 랜덤하게 단어를 골라
      * 자기 자신이 아니고
      * 같은 클러스터가 아닌
      토큰을 찾는다.
    - WordNet에서도 후보를 찾지 못했을 때 최후 fallback 용도로 사용.
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

def get_wordnet_candidates(tok):
    """
    [WordNet 유의어 후보]
    - spaCy 토큰 tok에 대해
      * 품사를 WordNet POS로 변환한 뒤
      * 해당 synset들의 lemma들을 동의어 후보로 사용.
    - 클러스터와는 독립적인 '사전적 동의어' 개념.
    """
    wn_pos = SPACY2WN.get(tok.pos_)
    if not wn_pos:
        return []

    lemma = tok.text.lower()
    syns = set()

    for syn in wn.synsets(lemma, pos=wn_pos):
        for l in syn.lemmas():
            w = l.name().replace("_", "")  # multi-word 표현 처리
            w_clean = clean_candidate(w)
            if not w_clean:
                continue
            if w_clean.lower() == lemma:
                continue
            syns.add(w_clean.lower())

    return list(syns)

# ------------------ 한 문서에 대한 공격 로직 ------------------
def attack_doc(text, nlp, per_doc_percent, seed, synonym_ratio, global_state):
    """
    하나의 문서(text)에 대해 치환 공격을 수행.

    반환값:
        new_text: 치환된 텍스트 문자열
        num_changes: 실제로 치환된 토큰 개수
        change_records: 각 치환에 대한 정보 리스트(dict)
            - row_id는 main() 쪽에서 추가
    
    파라미터:
        text: 원본 텍스트 (문자열)
        nlp: spaCy nlp 객체
        per_doc_percent: 이 문서에서 치환할 알파벳 토큰 비율 (%)
        seed: 랜덤 시드 (재현성)
        synonym_ratio: 
            공격 대상 토큰 중 '몇 %를 유의어 모드'로 교체할지 (0.0 ~ 1.0)
            - 유의어 모드: 주로 같은 클러스터 안에서 치환
            - 비유의어 모드: WordNet → 없으면 다른 클러스터
        global_state: 
            전체 코퍼스 기준 치환 개수를 제어하기 위한 dict
            - {"global_limit": int 또는 None, "global_used": int}
    """
    # 빈 문자열/NaN 처리
    if not isinstance(text, str) or not text.strip():
        return text, 0, []

    # synonym_ratio 안전하게 보정
    if synonym_ratio is None:
        synonym_ratio = 1.0
    try:
        synonym_ratio = float(synonym_ratio)
    except Exception:
        synonym_ratio = 1.0
    synonym_ratio = max(0.0, min(1.0, synonym_ratio))

    # spaCy 분석
    doc = nlp(text)
    tokens = [t for t in doc]

    # 알파벳 토큰 수 집계 (숫자, 구두점 제외)
    alpha_tokens = [t for t in tokens if t.is_alpha]
    total_alpha = len(alpha_tokens)
    if total_alpha == 0:
        return text, 0, []

    # 이 문서에서 치환할 토큰 개수 (ceil 사용 → 짧은 문서도 1개 이상 바뀔 수 있게)
    budget_by_percent = int(math.ceil(total_alpha * (per_doc_percent / 100.0))) if per_doc_percent > 0 else 0
    doc_budget = budget_by_percent

    # 전역(global) 치환 한도를 적용
    if global_state.get("global_limit") is not None:
        remaining = global_state["global_limit"] - global_state["global_used"]
        if remaining <= 0 or doc_budget <= 0:
            return text, 0, []
        doc_budget = min(doc_budget, remaining)

    # NER 엔터티 보호: 사람 이름, 숫자, 날짜 등은 바꾸지 않음
    ent_protected = set()
    for ent in doc.ents:
        if ent.label_ in PROTECT_ENT_TYPES:
            for i in range(ent.start, ent.end):
                ent_protected.add(i)

    # 치환 후보 토큰 인덱스 수집
    candidates = []
    for i, t in enumerate(tokens):
        if i in ent_protected:
            continue
        if not t.is_alpha:
            continue
        if t.text.lower() in STOP_WORDS:
            continue
        # 클러스터에 존재하지 않는 단어는 스킵 (클러스터 기반 통계 실험에 맞추기 위함)
        if t.text.lower() not in TOKEN_TO_CLUSTER:
            continue
        if t.like_num:
            continue
        if len(t.text) < 3:
            continue
        candidates.append(i)

    if not candidates or doc_budget <= 0:
        return text, 0, []

    # 실제 문자열 치환을 위한 준비
    changes = []  # (원래 시작 인덱스, 원래 끝 인덱스, 교체 문자열, 원래 단어) 리스트
    new_text = text
    spans = [(t.idx, t.idx + len(t.text)) for t in tokens]

    # 문서마다 랜덤 선택을 재현 가능하게 하기 위해 seed + hash(text) 사용
    rng = random.Random(seed + (hash(text) & 0xffffffff))
    pick_n = min(doc_budget, len(candidates))

    # 후보 인덱스를 정렬한 뒤, 그 "순서상" 인접한 두 후보가 동시에 선택되지 않도록 선택
    sorted_cands = sorted(candidates)
    cand_positions = list(range(len(sorted_cands)))
    rng.shuffle(cand_positions)

    picked_pos = set()        # sorted_cands 상에서 선택된 위치 인덱스
    picked = []               # 실제 토큰 인덱스(i)

    for pos in cand_positions:
        if len(picked) >= pick_n:
            break
        # 바로 앞/뒤 후보가 이미 선택되어 있으면 건너뛰기 → 연속 치환 방지
        if (pos - 1 in picked_pos) or (pos + 1 in picked_pos):
            continue
        picked_pos.add(pos)
        picked.append(sorted_cands[pos])

    picked = sorted(picked)

    records = []

    for i in picked:
        tok = tokens[i]
        rep_lemma = None      # 실제 교체할 소문자 형태 단어
        detail_mode = None    # 좀 더 세분화된 모드 정보 (분석용)

        use_synonym = (rng.random() < synonym_ratio)

        if use_synonym:
            # ------------------ [유의어 모드] ------------------
            # 1순위: 같은 클러스터 안의 유의어 (cluster_syn)
            syns = get_synonym_candidates(tok.text)
            if syns:
                key = f"{tok.text.lower()}|{tok.pos_}|{seed}|cluster_syn"
                idx_choice = pick_index(len(syns), key)
                rep_lemma = syns[idx_choice]
                detail_mode = "cluster_syn"
            else:
                # 같은 클러스터 유의어가 없다면 WordNet 동의어로 fallback
                wn_syns = get_wordnet_candidates(tok)
                if wn_syns:
                    key = f"{tok.text.lower()}|{tok.pos_}|{seed}|wn_fallback"
                    idx_choice = pick_index(len(wn_syns), key)
                    rep_lemma = wn_syns[idx_choice]
                    detail_mode = "wordnet_syn_fallback"
                else:
                    # 아무 후보도 없으면 이 토큰은 스킵
                    rep_lemma = None
                    detail_mode = "none"
        else:
            # ------------------ [비유의어 모드] ------------------
            # 1순위: WordNet 동의어 사용 (wordnet_syn)
            wn_syns = get_wordnet_candidates(tok)
            if wn_syns:
                key = f"{tok.text.lower()}|{tok.pos_}|{seed}|wn_primary"
                idx_choice = pick_index(len(wn_syns), key)
                rep_lemma = wn_syns[idx_choice]
                detail_mode = "wordnet_syn"
            else:
                # 2순위: WordNet 동의어도 없으면 다른 클러스터에서 후보 선택 (cluster_other)
                rep_lemma = get_nonsynonym_candidate(tok.text, seed)
                if rep_lemma is not None:
                    detail_mode = "cluster_other"
                else:
                    detail_mode = "none"

        # 치환할 단어가 없거나, 자기 자신과 동일하면 패스
        if not rep_lemma or rep_lemma.lower() == tok.text.lower():
            continue

        # 대소문자 패턴 유지
        rep = preserve_case(rep_lemma, tok.text)

        # 지금까지의 변경(changes) 때문에 인덱스가 얼마나 밀렸는지 보정
        consumed = 0
        for st, en, r, _orig in changes:
            if st < spans[i][0]:
                consumed += (len(r) - (en - st))
        start2 = spans[i][0] + consumed
        end2   = spans[i][1] + consumed

        # 안전 체크: 현재 new_text[start2:end2]가 실제 토큰 텍스트와 일치하는지 확인
        if new_text[start2:end2] != tok.text:
            continue

        # 실제 문자열 치환
        new_text = new_text[:start2] + rep + new_text[end2:]
        changes.append((spans[i][0], spans[i][1], rep, tok.text))

        # 로그 기록용 정보
        records.append({
            "start": spans[i][0],
            "end": spans[i][1],
            "orig": tok.text,
            "replacement": rep,
            "mode": "synonym" if use_synonym else "nonsynonym",
            "detail_mode": detail_mode,  # cluster_syn / wordnet_syn / cluster_other 등
        })

        # 전역 치환 개수 업데이트
        global_state["global_used"] += 1
        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            break

    return new_text, len(changes), records

# ------------------ per-doc-percent 컬럼 파싱 유틸 ------------------
def _coerce_percent(x):
    """
    DataFrame 셀에서 퍼센트(0~100)를 안전하게 읽기 위한 함수.
    - NaN, 변환 실패 시 None 반환.
    - 유한한 숫자이면 0~100 범위로 clamp 후 반환.
    """
    if pd.isna(x):
        return None
    try:
        val = float(x)
        if math.isfinite(val):
            return max(0.0, min(100.0, val))
    except Exception:
        return None
    return None

# ------------------ 메인 함수 ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="입력 CSV 파일 경로 (예: dataset/clustering_wm.csv)")
    parser.add_argument("--output", type=str, required=True,
                        help="치환 결과를 저장할 CSV 파일 경로")

    # 문서별 치환 비율 설정
    parser.add_argument("--per-doc-percent", type=float, default=None,
                        help="모든 문서에 동일하게 적용할 치환 비율 (예: 10 → 10%)")
    parser.add_argument("--per-doc-percent-col", type=str, default=None,
                        help="각 문서별 치환 비율이 들어있는 컬럼 이름 (0~100).")
    parser.add_argument("--default-per-doc-percent", type=float, default=None,
                        help="per-doc-percent-col 값이 비어있거나 잘못된 경우 사용될 기본값. \
                              지정하지 않으면 --per-doc-percent 또는 10%%를 사용.")

    # 전체 코퍼스 기준 글로벌 치환 비율
    parser.add_argument("--global-percent", type=float, default=None,
                        help="코퍼스 전체 알파벳 토큰의 몇 %까지만 치환할지 설정 (예: 5 → 5%).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="랜덤 시드 (재현성 확보용).")

    # 유의어 vs 비유의어 비율
    parser.add_argument("--synonym-ratio", type=float, default=1.0,
                        help="치환 대상 토큰 중 몇 %를 유의어 모드로 바꿀지 (0.0~1.0). \
                              1.0 = 전부 유의어, 0.0 = 전부 비유의어(WordNet→다른 클러스터).")

    args = parser.parse_args()

    # fallback 퍼센트 결정
    fallback_pct = (
        args.default_per_doc_percent
        if args.default_per_doc_percent is not None
        else (args.per_doc_percent if args.per_doc_percent is not None else DEFAULT_FALLBACK_PERCENT)
    )
    fallback_pct = max(0.0, min(100.0, float(fallback_pct)))

    # spaCy 모델 로드
    print("[INFO] loading spaCy en_core_web_sm (this may take a moment)...")
    nlp = spacy.load("en_core_web_sm")
    
    # 입력 CSV 읽기
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

    # 전역(global) 치환 한도 계산
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

    # per-doc-percent 컬럼 사용 시 미리 파싱
    percents = None
    if use_col is not None and use_col in df.columns:
        percents = df[use_col].apply(_coerce_percent).tolist()
    elif use_col is not None and use_col not in df.columns:
        print(f"[WARN] column `{use_col}` not found; falling back to default/fixed percent {fallback_pct}% for all docs.")

    # 각 문서에 대해 순차적으로 치환 수행
    for i, txt in enumerate(tqdm(texts, desc="Substituting")):
        # 이 문서에서 사용할 per-doc-percent 결정
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

        # 글로벌 한도에 도달하면 남은 문서들은 원문 그대로 복사 후 종료
        if global_state.get("global_limit") is not None and global_state["global_used"] >= global_state["global_limit"]:
            print("[INFO] global replacement limit reached; stopping further substitutions.")
            attacked.extend(texts[i+1:])
            break

    # 결과 컬럼 추가
    df[NEW_COL] = attacked
    
    # 출력 폴더 자동 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
        
    df.to_csv(args.output, index=False)
    print(f"\n[DONE] saved attacked CSV: {args.output}")

    # 치환 로그 저장
    if all_logs:
        log_dir = os.path.dirname(LOGMAP)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"[INFO] Created log directory: {log_dir}")
            
        logdf = pd.DataFrame(all_logs)
        logdf.to_csv(LOGMAP, index=False)
        print(f"[LOG] substitution map saved: {LOGMAP} (total replacements={len(all_logs)})")

if __name__ == "__main__":
    main()
