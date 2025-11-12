# coding=utf-8
print("======= [V2] demo_detect.py 스크립트 실행 시작 =======")

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import os

# watermark_processor.py 파일에서 WatermarkDetector 클래스를 가져옵니다.
try:
    from watermark_processor import WatermarkDetector
except ImportError:
    print("\n[ERROR] 'watermark_processor.py' 파일을 찾을 수 없습니다.")
    exit(1)
except SyntaxError as e:
    print(f"\n[ERROR] 'watermark_processor.py'에 문법 오류가 있습니다: {e}")
    print("-> 이전에 수정했던 '):' 괄호가 올바르게 닫혔는지 확인하세요!")
    exit(1)


# --- 스크립트 설정 (demo_watermark.py와 동일) ---
MODEL_NAME = "facebook/opt-125m"
CLUSTER_DATA_PATH = "cluster_data.npz"
GAMMA = 0.25
SEEDING_SCHEME = "simple_1"
# -----------------------------------------------


def load_detector_and_tokenizer(device_str, cluster_gamma_value):
    """
    [수정] cluster_gamma 값을 인자로 받아 탐지기를 로드합니다.
    """
    print(f"[INFO] Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # [수정] 이제 watermark_processor.py 파일을 수정할 필요가 없습니다!
    print(f"[INFO] Initializing WatermarkDetector (mode: cluster_gamma = {cluster_gamma_value})...")
    
    wm_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=GAMMA,
        seeding_scheme=SEEDING_SCHEME,
        device=torch.device(device_str), # device 객체 전달
        tokenizer=tokenizer,
        cluster_data_path=CLUSTER_DATA_PATH,
        select_green_tokens=True,
        cluster_gamma=cluster_gamma_value # [핵심] cluster_gamma 값을 여기서 직접 설정
    )
    return wm_detector, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Detect watermark and ADD z-score column to the input CSV.")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input CSV file (e.g., soft_wm_attack_10pct.csv)")
    # [수정] --output 인자 제거
    parser.add_argument("--text-col", type=str, required=True, 
                        help="Column name to detect (e.g., generated_text_attack)")
    parser.add_argument("--cluster-gamma", type=float, required=True,
                        help="Cluster gamma value to use (0.0 for 'soft', 0.15 for 'cluster')")
    args = parser.parse_args()

    # [수정] torch 로딩을 try 블록 안으로 이동
    try:
        # 장치 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")

        # 탐지기 및 토크나이저 로드
        detector, tokenizer = load_detector_and_tokenizer(device, args.cluster_gamma)
    
    except Exception as e:
        print(f"\n[FATAL ERROR] FAILED to load PyTorch or Detector: {e}")
        print("-> 'torch' 라이브러리 설치가 손상되었을 수 있습니다.")
        print("-> 'spacy' 모델(en_core_web_sm)이 설치되었는지 확인하세요 (python -m spacy download en_core_web_sm)")
        import traceback
        traceback.print_exc()
        return

    # 입력 CSV 파일 로드
    if not os.path.exists(args.input):
        print(f"\n[ERROR] Input file not found: {args.input}")
        return
        
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"\n[ERROR] Failed to read CSV: {e}")
        return
        
    if args.text_col not in df.columns:
        print(f"\n[ERROR] Text column '{args.text_col}' not found. (Available: {list(df.columns)})")
        return

    all_results = []
    print(f"\n[INFO] Detecting watermark in column: '{args.text_col}'")
    
    for text in tqdm(df[args.text_col].fillna(""), desc="Detecting"):
        if not isinstance(text, str) or len(text.split()) < 5:
            all_results.append({"p_value": None, "z_score": None})
            continue
        try:
            score_dict = detector.detect(text)
            all_results.append({
                "p_value": score_dict.get("p_value"),
                "z_score": score_dict.get("z_score")
            })
        except Exception as e:
            all_results.append({"p_value": None, "z_score": None, "error": str(e)})

    # [수정] 열 이름에 감마 값 추가 (겹치지 않도록)
    z_col_name = f"z_score_{args.text_col}_gamma{args.cluster_gamma}"
    p_col_name = f"p_value_{args.text_col}_gamma{args.cluster_gamma}"
    
    df[z_col_name] = [r.get('z_score') for r in all_results]
    df[p_col_name] = [r.get('p_value') for r in all_results]

    # [수정] 원본 입력 파일(args.input)에 덮어쓰기
    try:
        df.to_csv(args.input, index=False)
        print(f"\n[SUCCESS] Successfully ADDED columns to: {args.input}")
        
        avg_z_score = pd.to_numeric(df[z_col_name], errors='coerce').mean()
        print(f"\n[RESULT] Average z-score (gamma={args.cluster_gamma}): {avg_z_score:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save output file: {e}")


if __name__ == "__main__":
    main()
