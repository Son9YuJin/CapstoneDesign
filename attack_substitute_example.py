# attack_substitute_example.py
# 단일 텍스트 WordNet 치환 공격 + z-score 계산 + 줄바꿈 완전 지원 버전

import re
import math
import random
import hashlib
import spacy
from nltk.corpus import wordnet as wn

import torch
from transformers import AutoTokenizer
from llm_watermark.processor import WatermarkDetector

# ================================
# 기본 설정값 / 상수
# ================================

STOP_WORDS = {
    "the","a","an","and","or","but","if","so","to","of","in","on","for","at","by",
    "with","from","as","is","are","was","were","be","been","being","that","this",
    "it","its","they","them","their","we","our","you","your","he","she","his","her",
    "i","me","my","mine","us","not","no","yes","do","does","did","have","has","had",
}

PROTECT_ENT_TYPES = {
    "PERSON","ORG","GPE","LOC","NORP","FAC","EVENT","WORK_OF_ART","LAW","LANGUAGE",
    "PRODUCT","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
}

SPACY2WN = {
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
}

BAD_LEMMAS = {
    "amphetamine","methedrine","pep","pep_pill","upper","uppers",
    "downer","downers"
}

# ================================
# NLP, Tokenizer, Detector 로드
# ================================

print("[INFO] loading spaCy...")
NLP = spacy.load("en_core_web_sm")

print("[INFO] loading tokenizer & WatermarkDetector...")
TOKENIZER = AutoTokenizer.from_pretrained("facebook/opt-350m", use_fast=True)
VOCAB_IDS = list(TOKENIZER.get_vocab().values())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    DETECTOR = WatermarkDetector(
        vocab=VOCAB_IDS,
        gamma=0.25,
        delta=3.0,
        seeding_scheme="simple_1",
        tokenizer=TOKENIZER,
        device=DEVICE,
    )
except TypeError:
    DETECTOR = WatermarkDetector(
        vocab=VOCAB_IDS,
        gamma=0.25,
        seeding_scheme="simple_1",
        tokenizer=TOKENIZER,
        device=DEVICE,
    )

# ================================
# 유틸 함수들
# ================================

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

def get_synonym_candidates(token_text, wn_pos):
    lemma = token_text.lower()
    synsets = wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        return []

    cands = set()
    main_syn = synsets[0]

    for l in main_syn.lemmas():
        name = l.name()
        if "_" in name or "-" in name:
            continue
        w = clean_candidate(name)
        if not w:
            continue
        if w.lower() == lemma:
            continue
        if w.lower() in BAD_LEMMAS:
            continue
        cands.add(w.lower())

    return list(cands)

# ================================
# 단일 문장 공격 (줄바꿈 없는 경우)
# ================================

def substitute_attack_single(text, per_doc_percent=10.0, seed=42):
    if not isinstance(text, str) or not text.strip():
        return text

    doc = NLP(text)
    tokens = list(doc)
    alpha_tokens = [t for t in tokens if t.is_alpha]

    total_alpha = len(alpha_tokens)
    if total_alpha == 0:
        return text

    doc_budget = int(math.ceil(total_alpha * (per_doc_percent / 100.0)))
    if doc_budget <= 0:
        return text

    # 보호 엔터티
    ent_protected = set()
    for ent in doc.ents:
        if ent.label_ in PROTECT_ENT_TYPES:
            for i in range(ent.start, ent.end):
                ent_protected.add(i)

    # 치환 후보
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

    if not candidates:
        return text

    rng = random.Random(seed)
    pick_n = min(doc_budget, len(candidates))
    picked = sorted(rng.sample(candidates, pick_n))

    new_text = text
    changes = []
    spans = [(t.idx, t.idx + len(t.text)) for t in tokens]

    for i in picked:
        tok = tokens[i]
        wn_pos = SPACY2WN.get(tok.pos_)
        syns = get_synonym_candidates(tok.text, wn_pos)
        if not syns:
            continue

        key = f"{tok.text.lower()}|{tok.pos_}|{seed}"
        idx_choice = pick_index(len(syns), key)
        rep = preserve_case(syns[idx_choice], tok.text)

        # offset 보정
        offset = sum((len(r) - (end - start)) for start, end, r, _ in changes if start < spans[i][0])
        start2 = spans[i][0] + offset
        end2 = spans[i][1] + offset

        if new_text[start2:end2] != tok.text:
            continue

        new_text = new_text[:start2] + rep + new_text[end2:]
        changes.append((spans[i][0], spans[i][1], rep, tok.text))

    return new_text

# 전체 공격 함수
def substitute_attack(text, per_doc_percent=10.0, seed=42):

    lines = text.split("\n")
    attacked_lines = []

    for line in lines:
        if line.strip():
            attacked_lines.append(substitute_attack_single(line, per_doc_percent, seed))
        else:
            attacked_lines.append("")

    final_text = "\n".join(attacked_lines)

    # 전체 텍스트에 대해 z-score 계산
    detect_info = DETECTOR.detect(
        text=final_text,
        return_prediction=False,
        return_scores=True,
    )
    z = detect_info.get("z_score", float("nan"))

    return final_text, z

# ================================
# 실행 예제
# ================================

if __name__ == "__main__":
    text = """ some of the biggest cable providers who currently provide content on home. and they are also using streaming to deliver their shows across multiple platforms, including smartphones, tablets and even laptops so that consumers can watch in their homes. a net-zero-energy model - by 2024. 
In a bid to cut the waste and create jobs it is expected Netflix and Sony will be rolling out an initiative to reduce its global footprint.com to:
- provide low and zerocarbon, low and carbon energy sources through its global membership; and to: "start, develop and distribute renewable-energy assets, including wind farms, oil wells, gas turbines and hydrographic power stations.
- introduce solar and wind farms in 50 cities for 10 years.
"""

    new_text, z = substitute_attack(text, per_doc_percent=30)

    print("Before:\n", text)
    print("\nAfter:\n", new_text)
    print("\nz_score:", z)
