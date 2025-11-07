# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# Modified 2025 by Song YuJin for automatic batch processing (no Gradio)

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

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# ========================== CONFIG ==========================
INPUT_CSV = "dataset/human_prompts.csv"
OUTPUT_CSV = "dataset/clustering_wm.csv"
MODEL_NAME = "facebook/opt-6.7b"  
USE_GPU = False
MAX_NEW_TOKENS = 100
CLUSTER_DATA_PATH = "cluster_data.npz"
NUM_PROMPT_TOKENS = 200
# ============================================================

def preprocess_for_model(text: str) -> str:
    """Preprocessing before truncation or feeding into model."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_generated_text(text: str) -> str:
    """Postprocess model output."""
    if text is None:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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
    return model, tokenizer, device

def generate(prompt, model, tokenizer, device):
    """Generate watermarked text using WatermarkLogitsProcessor."""
    from argparse import Namespace
    args = Namespace(
        gamma=0.25,
        delta=2.0,
        seeding_scheme="simple_1",
        select_green_tokens=True,
        cluster_data_path=CLUSTER_DATA_PATH,
        use_sampling=True,
        sampling_temp=0.7,
        n_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        generation_seed=123,
        is_decoder_only_model=True,
    )

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
        tokenizer=tokenizer,
        cluster_data_path=args.cluster_data_path,
    )

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
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
    args.gamma = 0.25
    args.seeding_scheme = "simple_1"
    args.cluster_data_path = CLUSTER_DATA_PATH
    args.detection_z_threshold = 4.0
    args.normalizers = []
    args.ignore_repeated_bigrams = False
    args.select_green_tokens = True

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

    outputs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        original = str(row[prompt_col]) if not pd.isna(row[prompt_col]) else ""
        if not original.strip():
            continue

        try:
            processed = preprocess_for_model(original)
            used_prompt = build_used_prompt(processed, tokenizer, NUM_PROMPT_TOKENS)
            generated_raw = generate(used_prompt, model, tokenizer, device)
            generated_clean = clean_generated_text(generated_raw)
            scores = detection_scores_raw(generated_raw, tokenizer, device)

            outputs.append({
                "id": idx,
                "original_prompt": original,
                "used_prompt": used_prompt,
                "generated_text_raw": generated_raw,
                "generated_text_clean": generated_clean,
                "p_value": scores["p_value"],
                "z_score": scores["z_score"],
            })

        except Exception as e:
            print(f"[Error idx={idx}] {e}")
            print(traceback.format_exc())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(outputs).to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n Saved to {OUTPUT_CSV}\n")

if __name__ == "__main__":
    main()
