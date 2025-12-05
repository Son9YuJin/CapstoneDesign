from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_watermark import WatermarkLogitsProcessor, WatermarkDetector
import torch

# 1) 모델 로드
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2) vocab
vocab = tokenizer.get_vocab()

# 3) device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 4) 워터마크 삽입 Processor
logits_processor = WatermarkLogitsProcessor(
    vocab=vocab,
    tokenizer=tokenizer,
    gamma=0.15,
    delta=2.0
)

# 5) prompt
prompt = " "
inputs = tokenizer(prompt, return_tensors="pt")

# 6) 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=[logits_processor]
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Generated Text ===")
print(text)

# 7) Detector 생성
detector = WatermarkDetector(
    vocab=vocab,
    tokenizer=tokenizer,
    device=device, 
    gamma=0.25,
    delta=4.0
)

# 8) 검출
score = detector.detect(text)
print("\n=== Watermark Detection Result ===")
print(score)
