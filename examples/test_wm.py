from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from llm_watermark import WatermarkLogitsProcessor, WatermarkDetector
import torch
import os

# ========================== CONFIG ==========================
BEST_GAMMA = 0.25
BEST_DELTA = 4.0
BEST_CLUSTER_GAMMA = 0.25  
MODEL_NAME = "facebook/opt-125m"
USE_GPU = True
MAX_NEW_TOKENS = 150

CLUSTER_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "llm_watermark",
    "data",
    "cluster_data.npz",
)
CLUSTER_DATA_PATH = os.path.abspath(CLUSTER_DATA_PATH)
# ============================================================

torch.manual_seed(123)

# 1) 모델 & 토크나이저 로드 
print(f"Loading model {MODEL_NAME} ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_ids = list(tokenizer.get_vocab().values())

# 2) 워터마크 삽입 Processor 
watermark_processor = WatermarkLogitsProcessor(
    vocab=vocab_ids,
    gamma=BEST_GAMMA,
    delta=BEST_DELTA,
    seeding_scheme="simple_1",
    select_green_tokens=True,
    tokenizer=tokenizer,
    cluster_data_path=CLUSTER_DATA_PATH,
    cluster_gamma=BEST_CLUSTER_GAMMA,
)

# 3) 프롬프트 
prompt = (
    " "
)

tokd_input = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=True,
    truncation=True,
).to(device)

gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.6,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=3,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# 4) 워터마크가 들어간 텍스트 생성 
output = model.generate(
    **tokd_input,
    logits_processor=LogitsProcessorList([watermark_processor]),
    **gen_kwargs,
)

new_tokens = output[:, tokd_input["input_ids"].shape[-1]:]
generated_text = tokenizer.batch_decode(
    new_tokens, skip_special_tokens=True
)[0]

print("\n=== Generated Text (Continuation only) ===")
print(generated_text)

# 5) Detector 생성 
wm_detector = WatermarkDetector(
    vocab=vocab_ids,
    gamma=BEST_GAMMA,
    seeding_scheme="simple_1",
    device=device,
    tokenizer=tokenizer,
    cluster_data_path=CLUSTER_DATA_PATH,
    z_threshold=4.0,
    normalizers=[],
    ignore_repeated_bigrams=False,   
    select_green_tokens=True,
    cluster_gamma=BEST_CLUSTER_GAMMA,
)

# 6) 검출 수행
result = wm_detector.detect(generated_text)
print("\n=== Watermark Detection Result ===")
print(result)
