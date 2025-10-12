# wpt_cluster_min.py
# OPT-2.7B에서 Whitespace Prefix Token("Ġ")만 뽑아
# L2 정규화 → PCA → k-means → 각 클러스터 대표 토큰 몇 개 출력

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans,KMeans

# 고정 파라미터
MODEL_NAME = "facebook/opt-2.7b"
WPT_PREFIX = "Ġ"      
N_COMPONENTS = 256     
K = 300                # 클러스터 개수
SEED = 123

def l2norm_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (n + eps)

def main():
    print(f"{MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype="float16", low_cpu_mem_usage=True
    )

    vocab = tok.get_vocab()                 # token -> id
    inv = {v: k for k, v in vocab.items()}  # id -> token
    wpt_ids = [tid for tid in range(len(inv)) if inv[tid].startswith(WPT_PREFIX)]  #whitespace로 시작하는 토큰만
    wpt_tokens = [inv[tid] for tid in wpt_ids]  
    print(f"[whitespace prefix token] count={len(wpt_ids)}")

    with torch.no_grad():
        E = model.get_input_embeddings().weight.detach().cpu().float()   # (V, D) torch tensor
        idx = torch.as_tensor(wpt_ids, dtype=torch.long)                 # (Na,)
        A = E[idx].numpy()                                               # (Na, D)만 NumPy로 변환
    A = l2norm_rows(A) 


    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    Z = pca.fit_transform(A)                            # (Na, Nc)
    Z = l2norm_rows(Z)                                  # 코사인 근사 위해 다시 정규화

    #km = MiniBatchKMeans(n_clusters=K, random_state=SEED, batch_size=4096, n_init="auto",max_iter=300)
    km=KMeans(n_clusters=K,random_state=SEED,n_init='auto',max_iter=300)
    labels = km.fit_predict(Z)                          # (Na,)  cluster label
    centers = l2norm_rows(km.cluster_centers_)          # (K, Nc), 정규화  300,256  중심과 cos sim계산을 할것이기때문

    print("top tokens per cluster:")
    for c in range(min(K, 1000)):                          # 앞 몇 개만 보기
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            print(f"[cluster{c}] <empty>")
            continue
        sims = Z[idx] @ centers[c].reshape(-1, 1)       # (Nc,1), dot ≈ cosine
        order = np.argsort(-sims.squeeze())[:10]   #(label=c,1)->(label=c)  () - )  <내림차순 정렬  , 상위n개
        picks = "  ".join(f"{wpt_tokens[idx[o]].lstrip("Ġ")}({float(sims[order][i].item()):.2f})"
                         for i, o in enumerate(order))
        print(f"[cluster{c+1}] {picks}")

if __name__ == "__main__":
    main()
