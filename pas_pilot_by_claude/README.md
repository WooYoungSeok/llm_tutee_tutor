# PAS Pilot by Claude

8개 Synthetic Sample을 사용한 Personality Activation Steering (PAS) 실험

## 개요

이 디렉토리는 `test_by_claude`의 PAS 구현을 실제 데이터셋에 적용한 버전입니다.

### 8개 Synthetic Patterns

| Pattern | SE | IM | AS |
|---------|----|----|----||
| synthetic_pattern_1 | HIGH | HIGH | HIGH |
| synthetic_pattern_2 | HIGH | HIGH | LOW |
| synthetic_pattern_3 | HIGH | LOW | HIGH |
| synthetic_pattern_4 | HIGH | LOW | LOW |
| synthetic_pattern_5 | LOW | HIGH | HIGH |
| synthetic_pattern_6 | LOW | HIGH | LOW |
| synthetic_pattern_7 | LOW | LOW | HIGH |
| synthetic_pattern_8 | LOW | LOW | LOW |

### Traits

- **SE**: Self-Efficacy (학업 자기효능감)
- **IM**: Intrinsic Motivation (내적 동기)
- **AS**: Academic Stress (학업 스트레스)

## 파일 구조

```
pas_pilot_by_claude/
├── config.py                 # 설정 상수
├── data_utils.py             # 데이터 로딩 및 프롬프트 생성
├── activation_extractor.py   # Head-wise activation 추출
├── probe_trainer.py          # Probe 학습 및 direction 탐색
├── steering.py               # Activation steering 적용
├── main.py                   # 학습 파이프라인
├── evaluate.py               # Base vs Steered 모델 평가
├── chat.py                   # 대화형 Alpha 조정 인터페이스
├── outputs/                  # 학습 결과 저장
│   ├── interventions_se_*.pkl
│   ├── interventions_im_*.pkl
│   ├── interventions_as_*.pkl
│   └── summary_*.json
└── Data files:
    ├── Test-set.json                    # 8개 synthetic samples
    ├── traintest_split_balanced_80_20.json  # Train/test split
    └── all_data_ItemKey.xlsx            # 문항 정보
```

## 사용법

### 1. 학습 (Training)

```bash
# 모든 trait에 대해 학습
python main.py --model_name meta-llama/Meta-Llama-3-8B-Instruct

# 특정 trait만 학습
python main.py --trait se

# 4bit 양자화 사용 (메모리 절약)
python main.py --use_4bit

# Top-K heads 조정
python main.py --num_heads 48
```

출력 파일명에 trait 이름이 포함됩니다:
- `interventions_se_20260212_120000.pkl`
- `interventions_im_20260212_120000.pkl`
- `interventions_as_20260212_120000.pkl`

### 2. 평가 (Evaluation)

```bash
# 특정 trait 평가
python evaluate.py --trait se --alpha 1.0

# 모든 trait 평가
python evaluate.py --evaluate_all

# Alpha 값 조정
python evaluate.py --trait im --alpha 2.0

# Base 모델만 평가 (steering 없이)
python evaluate.py --base_only
```

출력 예시:
```
======================================================================
Self-Efficacy (SE) Evaluation Results
======================================================================
Alpha: 1.0

Overall Accuracy:
  Base model: 52.3%
  Steered model: 68.5%
  Improvement: +16.2%

Per-Sample Results (by SE level):
----------------------------------------------------------------------
Case                          SE       Base    Steered      Diff
----------------------------------------------------------------------
synthetic_pattern_1         HIGH      55.0%      72.0%    +17.0%
synthetic_pattern_2         HIGH      53.0%      70.0%    +17.0%
synthetic_pattern_5          LOW      48.0%      65.0%    +17.0%
synthetic_pattern_6          LOW      50.0%      67.0%    +17.0%
----------------------------------------------------------------------
```

### 3. 대화 (Interactive Chat)

```bash
# 대화 시작
python chat.py

# 4bit 양자화 사용
python chat.py --use_4bit
```

명령어:
```
/load all           # 모든 intervention 로드
/alpha se 2.0       # SE alpha를 2.0으로 설정
/alpha im -1.0      # IM alpha를 -1.0으로 (반대 방향)
/alpha as 0         # AS steering 끄기
/status             # 현재 alpha 값 확인
/reset              # 모든 alpha를 0으로 리셋
/clear              # 대화 히스토리 초기화
/help               # 도움말
/quit               # 종료
```

예시 세션:
```
You: /load all
Loaded SE intervention: 48 heads
Loaded IM intervention: 48 heads
Loaded AS intervention: 48 heads

You: /alpha se 2.0
Set SE alpha to +2.00

You: /status
Current Steering Status:
  Self-Efficacy (SE): ACTIVE, Alpha: +2.00
  Intrinsic Motivation (IM): INACTIVE, Alpha: 0.00
  Academic Stress (AS): INACTIVE, Alpha: 0.00

You: Tell me about your approach to learning mathematics.
Assistant: I find mathematics fascinating! I believe I can master
any concept with enough practice and dedication...

You: /alpha se -2.0
Set SE alpha to -2.00

You: Tell me about your approach to learning mathematics.
Assistant: Mathematics has always been challenging for me. I often
doubt my ability to understand complex problems...
```

## 파이프라인 설명

### Step 1: 데이터 로드
- 8개 synthetic sample (SE/IM/AS의 high/low 조합)
- 각 trait별 train items (약 29개)
- 각 trait별 test items (7개)

### Step 2: Activation Sample 생성
- 각 sample × 각 trait의 train items
- Agree/Disagree 프롬프트 쌍 생성
- Label: 해당 trait의 positive 방향 결정

### Step 3: Activation 추출
- Forward hook으로 o_proj 입력 캡처
- 마지막 토큰의 head-wise activation 저장
- Shape: (num_samples, num_layers, num_heads, head_dim)

### Step 4: Probe 학습
- 각 head에 LogisticRegression 학습
- 60/40 train/validation split
- Validation accuracy로 Top-K heads 선택

### Step 5: Intervention 저장
- 선택된 heads의 정규화된 direction
- Sigma (direction 방향 activation std)
- Trait별로 별도 파일에 저장

### Step 6: 평가/대화
- Test items에서 Base vs Steered 비교
- Alpha 값 조정으로 steering 강도 제어
- 음수 alpha로 반대 방향 steering 가능

## Alpha 값 가이드

| Alpha | 효과 |
|-------|------|
| 0.0 | Steering 없음 (Base 모델) |
| 1.0 | 기본 steering 강도 |
| 2.0~5.0 | 강한 steering |
| -1.0~-2.0 | 반대 방향 steering |
| >5.0 | 매우 강함 (품질 저하 가능) |

## 참고사항

- 학습에는 8개 synthetic sample의 모든 응답을 사용
- 평가는 각 sample별로 해당 trait의 test items만 사용
- 음수 alpha는 해당 trait의 반대 성향으로 steering
- 여러 trait의 alpha를 동시에 조정하여 복합 steering 가능
