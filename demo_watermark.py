# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"

import os
import re
import csv
import traceback
import pandas as pd
from tqdm import tqdm
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LogitsProcessorList,
)

from llm_watermark import WatermarkLogitsProcessor, WatermarkDetector

# ========================== CONFIG ==========================
# best parameters
BEST_GAMMA = 0.25    # greenlist로 보낼 토큰 비율(고정)
BEST_DELTA = 4.0    # greenlist 토큰에 더해줄 logit 보정 세기
BEST_CLUSTER_GAMMA = 1.0   # 클러스터 워터마킹 사용 비율 

INPUT_CSV = "dataset/human_prompts.csv"
OUTPUT_CSV = f"dataset/clustering_wm_gamma{BEST_GAMMA}_delta{BEST_DELTA}_cg{BEST_CLUSTER_GAMMA}.csv"
MODEL_NAME = "facebook/opt-125m"
USE_GPU = True
MAX_NEW_TOKENS = 100
CLUSTER_DATA_PATH = "src/llm_watermark/data/cluster_data.npz"
NUM_PROMPT_TOKENS = 100
# ============================================================


def preprocess_for_model(text: str) -> str:
    """Preprocessing before truncation or feeding into model."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    text = re.sub(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\uA960-\uA97F\uD7B0-\uD7FF]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def clean_generated_text(text: str) -> str:
    if text is None:
        return ""

    # 공백 정리
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 완전한 비ASCII 문자 제거
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # 한글 및 자모 제거
    text = re.sub(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    text = re.sub(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\uA960-\uA97F\uD7B0-\uD7FF]", "", text)

    # 긴 구분선/반복 기호를 ' — '로 축약 (---, - - - -, ***, ___, === 등)
    text = re.sub(r"(?:[-*_=]\s*){3,}", " — ", text)

    # 공백-문장부호 간격 정리
    text = re.sub(r"\s+([,\.!?;:])", r"\1", text)

    # 양끝 불필요한 기호 제거
    text = text.strip(" -_*=•\t")

    # 마지막 문장부호(. ? !)까지 자르고 마감 (구분선으로 끝나는 현상 방지)
    m = re.search(r"[\.!?](?!.*[\.!?])", text)
    if m:
        text = text[:m.end()]
    return text


def build_used_prompt(original_prompt: str, tokenizer, n_tokens: int) -> str:
    """Take first N tokens and decode back to text."""
    ids = tokenizer.encode(original_prompt, add_special_tokens=False)
    truncated = ids[:n_tokens]
    used = tokenizer.decode(truncated, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    return preprocess_for_model(used)


def load_model():
    """Load model and tokenizer."""
    print(f"Loading model {MODEL_NAME} ...")
    is_seq2seq = any(mt in MODEL_NAME for mt in ["t5", "T0"])
    is_decoder_only = any(mt in MODEL_NAME for mt in ["gpt", "opt", "bloom"])
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    elif is_decoder_only:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    else:
        raise ValueError(f"Unknown model type: {MODEL_NAME}")

    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)
    else:
        device = "cpu"

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # OPT 일부 체크포인트에 pad_token이 없는 경우 대비
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def _build_bad_words_ids(tokenizer):
    """
    연속 대시/스페이스-대시와 같은 패턴을 금칙어로 등록하여 구분선 폭주 억제.
    일반적인 하이픈 사용은 허용하고, 4개 이상 연속 패턴만 막음.
    """
    ban_patterns = [
        "----", "-----", "------", "-------", "--------",
        "- - - -", "- - - - -", "- - - - - -",
        "— — —", "— — — —", "———", "————"
    ]
    ids = []
    for p in ban_patterns:
        enc = tokenizer.encode(p, add_special_tokens=False)
        if enc:  # 빈 리스트는 제외 (HF 스펙)
            ids.append(enc)
    return ids if ids else None  # 아무것도 없으면 None 반환


def generate(prompt, model, tokenizer, device):
    """Generate watermarked text using WatermarkLogitsProcessor."""
    from argparse import Namespace
    args = Namespace(
        gamma=BEST_GAMMA,              
        delta=BEST_DELTA,              
        seeding_scheme="simple_1",
        select_green_tokens=True,
        cluster_data_path=CLUSTER_DATA_PATH,
        use_sampling=True,
        sampling_temp=0.6,
        n_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        generation_seed=123,
        is_decoder_only_model=True,
        cluster_gamma=BEST_CLUSTER_GAMMA,  
    )

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
        tokenizer=tokenizer,
        cluster_data_path=args.cluster_data_path,
        cluster_gamma=args.cluster_gamma,  
    )

    bad_words_ids = _build_bad_words_ids(tokenizer)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.sampling_temp,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if bad_words_ids is not None:
        gen_kwargs["bad_words_ids"] = bad_words_ids  # 추가: 긴 구분선 직접 차단

    tokd_input = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True, truncation=True
    ).to(device)

    torch.manual_seed(args.generation_seed)
    output = model.generate(
        **tokd_input,
        logits_processor=LogitsProcessorList([watermark_processor]),
        **gen_kwargs,
    )

    if args.is_decoder_only_model:
        output = output[:, tokd_input["input_ids"].shape[-1]:]

    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decoded_output


def detection_scores_raw(text: str, tokenizer, device):
    """Compute watermark detection raw scores."""
    args = type("Args", (), {})()
    args.gamma = BEST_GAMMA
    args.seeding_scheme = "simple_1"
    args.cluster_data_path = CLUSTER_DATA_PATH
    args.detection_z_threshold = 4.0
    args.normalizers = []
    args.ignore_repeated_bigrams = False
    args.select_green_tokens = True
    args.cluster_gamma = BEST_CLUSTER_GAMMA   

    wm_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        cluster_data_path=args.cluster_data_path,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens,
        cluster_gamma=args.cluster_gamma,   
    )

    if len(text) - 1 <= wm_detector.min_prefix_len:
        return {"p_value": "", "z_score": ""}

    result = wm_detector.detect(text)
    return {"p_value": result.get("p_value", ""), "z_score": result.get("z_score", "")}


def main():
    model, tokenizer, device = load_model()
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found.")

    df = pd.read_csv(INPUT_CSV)
    prompt_col = "data" if "data" in df.columns else df.columns[0]

    # 기존 결과가 있으면 읽어서 이어쓰기 모드로 전환
    processed_ids = set()
    outputs = []
    if os.path.exists(OUTPUT_CSV):
        try:
            prev = pd.read_csv(OUTPUT_CSV)
            if "id" in prev.columns:
                processed_ids = set(prev["id"].tolist())
                print(f"Found existing output with {len(processed_ids)} rows. Resuming...")
            outputs = prev.to_dict(orient="records")
        except Exception as e:
            print(f"[Warn] Failed to read existing OUTPUT_CSV: {e}")
            outputs = []

    # 주기적 저장 간격 (필요시 1로 낮추면 샘플마다 저장)
    SAVE_EVERY = 10
    since_last_save = 0

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # 이미 처리한 행이면 건너뜀
            if idx in processed_ids:
                continue

            original = str(row[prompt_col]) if not pd.isna(row[prompt_col]) else ""
            if not original.strip():
                continue

            try:
                processed = preprocess_for_model(original)
                used_prompt = build_used_prompt(processed, tokenizer, NUM_PROMPT_TOKENS)
                generated_raw = generate(used_prompt, model, tokenizer, device)
                generated_clean = clean_generated_text(generated_raw)
                scores = detection_scores_raw(generated_raw, tokenizer, device)

                ascii_only = lambda s: re.sub(r"[^\x00-\x7F]+", "", s or "")
                outputs.append({
                    "id": idx,
                    "original_prompt": ascii_only(original),
                    "used_prompt": ascii_only(used_prompt),
                    "generated_text_raw": ascii_only(generated_raw),
                    "generated_text_clean": clean_generated_text(generated_raw),
                    "p_value": scores.get("p_value", ""),
                    "z_score": scores.get("z_score", ""),
                })

                processed_ids.add(idx)
                since_last_save += 1

                if since_last_save >= SAVE_EVERY:
                    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
                    pd.DataFrame(outputs).to_csv(
                        OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\'
                    )
                    since_last_save = 0

            except Exception as e:
                print(f"[Error idx={idx}] {e}")
                print(traceback.format_exc())

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Saving partial results...\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(outputs).to_csv(
        OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\'
    )
    print(f"\nSaved {len(outputs)} total results to {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
