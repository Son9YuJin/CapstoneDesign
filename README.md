# LLM Text Watermarking Library (Capstone Design)

이 레포지토리는 대규모 언어 모델(LLM)의 **텍스트 워터마킹 삽입·검출 실험**을 위해 만들어진  
파이썬 라이브러리 **`llm_watermark`** 와 관련 데모/실험 스크립트를 포함합니다.

캡스톤 디자인 프로젝트의 일환으로,

- 워터마크 삽입용 `WatermarkLogitsProcessor`  
- 워터마크 검출용 `WatermarkDetector`  
- 클러스터 기반 워터마킹(`cluster_data.npz`)  
- 데모/공격 스크립트  

를 패키지 구조로 정리했습니다.

---

## 프로젝트 구조 (요약)

라이브러리 코드는 `src/llm_watermark/` 아래에 위치합니다.

```text
src/
  llm_watermark/
    __init__.py
    processor.py                 # WatermarkLogitsProcessor, WatermarkDetector
    extended_processor.py        # 확장 버전
    alternative_prf_schemes.py
    normalizers.py
    homoglyphs.py
    data/
      cluster_data.npz
    homoglyph_data/
      ...
demo_watermark.py                # 워터마크 삽입 + 검출 데모 (CSV 기반)
demo_detect.py                   # 검출 전용 데모
pyproject.toml                   # 패키지 설정
```

## 설치 방법

### 1) 레포 클론 후 설치 (개발/실험용)

```bash
git clone -b watermark-lib --single-branch https://github.com/Son9YuJin/CapstoneDesign.git
cd CapstoneDesign
python -m pip install -e .
```

### 2) GitHub에서 직접 설치 (코드 수정 필요 없을 때)

```bash
python -m pip install "git+https://github.com/Son9YuJin/CapstoneDesign.git@watermark-lib"
```
레포를 직접 클론하지 않고, llm_watermark 라이브러리만 바로 설치해서 쓰고 싶을 때 사용합니다.
설치 후 동일하게 다음과 같이 import 할 수 있습니다.
```bash
from llm_watermark import WatermarkLogitsProcessor, WatermarkDetector
```
