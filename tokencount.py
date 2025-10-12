from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
vocab = tok.get_vocab()  # dict: {token_str: token_id}

anchor_ids = [tid for t, tid in vocab.items() if t.startswith("Ġ")]
print("whitespace prefix token 토큰 수:", len(anchor_ids))
print("전체 토큰 수:", len(vocab))
