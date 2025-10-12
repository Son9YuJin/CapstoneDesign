# anchor_vs_subword.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np, pandas as pd, argparse, os, sys


#   BPE=byte pair encoding
#    1. 문자단위->토큰단위 (자주 등장하는 pair)  , " ", ","..등등도 기본토큰
#    문자1개는 무조건존재함 -> 자주등장하는 pair묶어서 vocab에 추가 ->반복
#   "I love dog"  -> ["ĠI", "Ġlove", "Ġdog"]    "puppy" → ["Ġpu", "pp", "y"]  
#  Ġpu<- 단어의 시작이니 의미를 강하게 포함함  , pp,y <- 다양한 맥락에서 사용됨(apple,supply)의미를 포함하지못함

MODEL = "facebook/opt-2.7b" 

pairs = [
    ("dog", "puppy"),
    ("car", "vehicle"),
    ("big", "large"),
    ("happy", "glad"),
    ("run", "jog"),
    ("movie", "film"),
    ("begin", "start"),
    ("buy", "purchase"),
    ("house", "home"),
    ("kid", "child"),
]

def cosine(a,b):
    a=np.asarray(a); b=np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na==0 or nb==0:
        return float('nan')
    return float(np.dot(a,b)/(na*nb))

def main(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16", low_cpu_mem_usage=True)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    vocab = tokenizer.get_vocab()

    rows=[]
    for a,b in pairs:
        ids_a = tokenizer.encode(a, add_special_tokens=False)   # word -> token ids
        ids_b = tokenizer.encode(b, add_special_tokens=False)

        wst_ids_a = tokenizer.encode(" " + a, add_special_tokens=False)   
        wst_ids_b = tokenizer.encode(" " + b, add_special_tokens=False)

        wst_id_a = wst_ids_a[0] if wst_ids_a else None
        wst_id_b = wst_ids_b[0] if wst_ids_b else None

        mean_a = emb[ids_a].mean(axis=0)  #토큰이 여러개로 나뉘었다면 mean
        mean_b = emb[ids_b].mean(axis=0)
        first_a = emb[ids_a[0]]  
        first_b = emb[ids_b[0]]
        wst_a = emb[wst_id_a] if wst_id_a is not None else None
        wst_b = emb[wst_id_b] if wst_id_b is not None else None

        sim_wst = cosine(wst_a, wst_b) if wst_a is not None and wst_b is not None else None  
        sim_mean   = cosine(mean_a, mean_b)     # 토큰쪼개졌을때 평균낸 벡터들의 유사도 
        sim_first  = cosine(first_a, first_b)    # 토큰쪼개졌을때 그냥 첫번째 토큰의 유사도

        # Also collect token strings for debugging
        tokstrs_a = [tokenizer.decode([i]) for i in ids_a]
        tokstrs_b = [tokenizer.decode([i]) for i in ids_b]

        rows.append({
            "word_a": a, "word_b": b,
            " tokens_a": "|".join(tokstrs_a), "tokens_b": "|".join(tokstrs_b),
            "   sim_ws": sim_wst," sim_mean": sim_mean, " sim_first": sim_first
        })

    df = pd.DataFrame(rows)
    print("\nResults:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    m = MODEL
    if len(sys.argv) > 1:
        m = sys.argv[1]
    main(m)
