# PAS (Personality Activation Steering) 실험 코드

논문 "Personality Alignment of Large Language Models" 기반 activation steering 구현.

## 핵심 개념

### 1. Direction 탐색 (Section 4.2)

각 attention head에서 성격 선호도를 가장 잘 분리하는 방향 벡터를 찾습니다.

```
Positive sample: "Question: Given a statement of you: '{Statement}', Do you agree? Answer: Yes"
Negative sample: "Question: Given a statement of you: '{Statement}', Do you agree? Answer: No"
```

**Probe 학습:**
```
p_θ(x_h_l) = sigmoid(⟨θ, x_h_l⟩)
```
- `x_h_l ∈ R^D`: layer l의 head h에서 추출한 activation
- `θ ∈ R^D`: probe weights (학습 후 정규화하여 direction으로 사용)

**Top-K Head 선택:**
- 60% train / 40% validation split
- Validation accuracy 기준 상위 K개 head 선택
- 선택된 head의 정규화된 θ가 intervention direction

### 2. Activation Intervention (Section 4.2)

선택된 head의 output에 direction 방향으로 shift 적용:

```
x_{l+1} = x_l + Σ_h Q_h_l[Att_h_l(P_h_l · x_l) + α · σ_h_l]
```

- `α`: 조정 강도 (intervention intensity)
- `σ_h_l`: 정규화된 direction θ_h_l 방향의 표준편차 벡터

**핵심:** 모델 파라미터는 변경하지 않고, `α * σ_h_l` 만큼의 bias만 추가

### 3. 최적 α 탐색 (Section 4.3)

```
Optimal α = argmin_{α∈[0,10]} f(α)
```

Golden section search로 [0, 10] 구간에서 최적 α 탐색.

## 파일 구조

```
test_by_claude/
├── README.md           # 이 파일
├── config.py           # 설정 상수
├── data_utils.py       # 데이터 로딩 및 프롬프트 포맷팅
├── activation_extractor.py  # Head-wise activation 추출
├── probe_trainer.py    # Probe 학습 및 direction 탐색
├── steering.py         # Activation intervention 적용
└── main.py             # 메인 파이프라인
```

## 실행 방법

```bash
cd test_by_claude
python main.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_heads 48
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model_name` | `meta-llama/Meta-Llama-3-8B-Instruct` | HuggingFace 모델 이름 |
| `--num_heads` | 48 | Intervention할 top-K head 수 |
| `--alpha` | None | 고정 α값 (None이면 자동 탐색) |
| `--train_ratio` | 0.6 | Probe 학습용 데이터 비율 |

## 파이프라인 흐름

```
[1] 데이터 로드 (Test-set.json + ItemKey.xlsx)
         ↓
[2] Positive/Negative 프롬프트 생성
         ↓
[3] 각 프롬프트에 대해 모든 head의 activation 추출
         ↓
[4] Head별 probe (LogisticRegression) 학습
         ↓
[5] Validation accuracy 기준 Top-K head 선택
         ↓
[6] 선택된 head의 direction (정규화된 θ) 및 σ 계산
         ↓
[7] 최적 α 탐색 (optional)
         ↓
[8] Steering 적용하여 inference
```

## 핵심 차이점 (기존 pas_pilot vs 본 구현)

| 항목 | pas_pilot | test_by_claude |
|------|-----------|----------------|
| 프롬프트 형식 | A/B 선택지 | Yes/No 동의 여부 |
| Activation 추출 | baukit TraceDict | 직접 forward hook |
| Direction 계산 | mean difference | probe coef (정규화) |
| σ 계산 | proj_val의 std | direction 방향 activation std |
| α 탐색 | 없음 | Golden section search |

## Trait 매핑

| 약어 | 전체 이름 | 설명 |
|------|-----------|------|
| SE | Self-Efficacy | 자기효능감 |
| IM | Intrinsic Motivation | 내재적 동기 |
| AS | Academic Stress | 학업 스트레스 |

각 trait에 대해:
- `+` key (positive keyed): "Accurate" = 높은 trait 수준
- `-` key (negative keyed): "Accurate" = 낮은 trait 수준

## 참고

- [Personality Alignment of Large Language Models](https://arxiv.org/abs/2401.17939)
- [Inference-Time Intervention (Li et al., 2024)](https://arxiv.org/abs/2306.03341)
