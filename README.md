# Personality Alignment of LLMs for Educational Constructs

LLM의 내부 활성화(activation)를 조작하여 **Self-Efficacy(SE)**, **Intrinsic Motivation(IM)**, **Academic Stress(AS)** 세 가지 교육 심리 구인(construct)의 성격 방향을 제어하는 프로젝트입니다. [Personality Alignment of Large Language Models (PAS)](https://arxiv.org/abs/2401.17939) 방법론을 기반으로 합니다.

## 프로젝트 개요

```
[사용자 질문]
      ↓
[Tokenizer]
      ↓
[LLaMA 모델]  ←  modeling_llama.py (수정된 HuggingFace LLaMA)
      │
      ├─ Layer 0
      ├─ Layer 1
      ├─ ...
      ├─ Layer N
      │     └─ Self-Attention
      │           └─ head_out  ← PAS 개입 지점
      │                 ↑
      │               pas.py (PASLM 클래스)
      │
      └─ Output (성격이 반영된 응답)
```

### PAS 파이프라인 (7-Step)

| Step | 설명 | 관련 코드 |
|------|------|-----------|
| 1 | 성격 대비 데이터 준비 (SE/IM/AS positive vs negative) | `main_by_claude.py` → `process_pas()` |
| 2 | Forward pass + activation 추출 | `pas.py` → `preprocess_activate_dataset()` |
| 3 | Head-wise activation 분리 | `pas.py` → `get_llama_activations_bau()` + `baukit.TraceDict` |
| 4 | Probe로 "성격을 잘 드러내는 head" 탐색 | `pas.py` → `train_probes()` (LogisticRegression) |
| 5 | 성격 방향 벡터(direction vector) 계산 | `pas.py` → `get_com_directions()` |
| 6 | 해당 head output에 bias 개입 | `pas.py` → `set_activate()` → `o_proj.bias` 수정 |
| 7 | Inference 시 성격 고정 | `main_by_claude.py` → `generateAnswer()` |

## 타겟 구인(Construct)

기존 PAS 논문은 Big Five (A, C, E, N, O) 성격 특성을 다루지만, 본 프로젝트는 교육 맥락의 세 가지 구인으로 대체합니다.

| 구인 | 약어 | 설명 | 샘플 극성 |
|------|------|------|-----------|
| Self-Efficacy | SE | 자기효능감 | positive / negative |
| Intrinsic Motivation | IM | 내재적 동기 | positive / negative |
| Academic Stress | AS | 학업 스트레스 | positive / negative |

샘플 성격 패턴 예시:
```json
{"SE": "positive", "IM": "positive", "AS": "positive"}
```

각 구인의 긍정/부정 극단적 조합으로 실험을 진행합니다.

## 디렉토리 구조

```
llm_tutee_tutor/
├── README.md
├── pas_pilot/
│   ├── main_by_claude.py        # 메인 실행 스크립트 (파이프라인 전체 orchestration)
│   ├── pas.py                   # PASLM 클래스 (모델 로딩, activation 추출, 개입)
│   ├── modeling_llama.py        # 수정된 LLaMA 모델 (head_out Identity hook 포함)
│   ├── baseline_utils.py        # 베이스라인 방법 유틸리티
│   ├── analyze_activations.py   # Activation 분석 및 시각화
│   ├── Test-set.json            # 테스트 데이터셋 (300 samples)
│   ├── traintest_split_balanced_80_20.json  # Train/Test 분할 인덱스
│   └── all_data_ItemKey.xlsx    # 문항 텍스트 매핑
└── preprocessing/
    ├── itemkey_traintest_split.ipynb  # 데이터 전처리 노트북
    ├── Test-set.json
    ├── traintest_split_balanced_80_20.json
    └── all_data_ItemKey.xlsx
```

## 실행 방법

### 기본 실행 (PAS 모드)

```bash
cd pas_pilot
python main_by_claude.py --modes PAS --model_file meta-llama/Meta-Llama-3-8B-Instruct
```

### 지원 모드

| 모드 | 설명 |
|------|------|
| `NO_CHANGE` | 베이스라인 - 모델 변경 없이 응답 생성 |
| `few-shot` | Few-shot 학습 기반 성격 유도 |
| `personality_prompt` | 프롬프트 기반 성격 유도 |
| `PAS` | Activation Steering 기반 성격 개입 (핵심) |

### 지원 모델

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-70b-chat-hf` (4-bit 양자화)
- Mistral 계열 (별도 `modeling_mistral.py` 필요)

## 핵심 메커니즘

### 1. Activation 추출

`baukit.TraceDict`를 사용하여 각 레이어의 `self_attn.head_out` 지점에서 head-wise activation을 추출합니다.

```python
HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(num_layers)]
with TraceDict(model, HEADS + MLPS) as ret:
    output = model(prompt, output_hidden_states=True)
```

### 2. 성격 방향 벡터 계산

각 head에서 positive/negative 샘플의 activation 평균 차이로 방향 벡터를 계산합니다.

```python
true_mass_mean = np.mean(activations[labels == 1], axis=0)
false_mass_mean = np.mean(activations[labels == 0], axis=0)
direction = true_mass_mean - false_mass_mean
```

### 3. Activation Steering (개입)

`o_proj`의 bias를 조작하여 특정 head의 출력 방향을 제어합니다.

```python
bias_tobe = F.linear(displacement, o_proj.weight)
model.layers[layer].self_attn.o_proj.bias = Parameter(bias_tobe)
```

### 4. 레이어 할당 전략

`get_activate_layer()` 함수로 각 구인별 개입 레이어를 결정합니다. 모델의 중간 레이어 범위(1/4 ~ 3/4)를 5등분하여 SE → IM → AS 순서로 배치합니다.

## 채점 체계

- **A (Accurate)**: 2점 — 문항이 자신을 정확히 설명한다고 답변
- **B (Inaccurate)**: 1점 — 문항이 자신을 정확히 설명하지 않는다고 답변
- **UNK**: 0점 — 파싱 불가

positive/negative key에 따라 역채점이 적용됩니다.

## 의존성

```
torch
transformers
baukit
einops
scikit-learn
pandas
numpy
tqdm
huggingface_hub
```

## 참고 문헌

- [Personality Alignment of Large Language Models](https://arxiv.org/abs/2401.17939)
- [PAPI-300K Dataset](https://huggingface.co/datasets/WestlakeNLP/PAPI-300K)
- [IPIP-NEO Personality Inventory](https://ipip.ori.org/)
