"""
config.py
=========
PAS 실험 설정 상수 - pas_pilot_by_claude용

8개 샘플 (SE/IM/AS의 positive/negative 조합)을 사용한 activation steering
"""

import os

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

# 데이터 파일 (pas_pilot_by_claude 내부 복사본 사용)
TEST_SET_FILE = os.path.join(SCRIPT_DIR, 'Test-set.json')
ITEM_KEY_FILE = os.path.join(SCRIPT_DIR, 'all_data_ItemKey.xlsx')
TRAIN_TEST_SPLIT_FILE = os.path.join(SCRIPT_DIR, 'traintest_split_balanced_80_20.json')

# Trait 정보
TRAITS = ['se', 'im', 'as']
TRAIT_NAMES = {
    'se': 'Self-Efficacy',
    'im': 'Intrinsic Motivation',
    'as': 'Academic Stress'
}

# Pattern과 Trait 레벨 매핑
# Test-set.json의 8개 패턴
PATTERN_LABELS = {
    'synthetic_pattern_1': {'se': 'high', 'im': 'high', 'as': 'high'},
    'synthetic_pattern_2': {'se': 'high', 'im': 'high', 'as': 'low'},
    'synthetic_pattern_3': {'se': 'high', 'im': 'low', 'as': 'high'},
    'synthetic_pattern_4': {'se': 'high', 'im': 'low', 'as': 'low'},
    'synthetic_pattern_5': {'se': 'low', 'im': 'high', 'as': 'high'},
    'synthetic_pattern_6': {'se': 'low', 'im': 'high', 'as': 'low'},
    'synthetic_pattern_7': {'se': 'low', 'im': 'low', 'as': 'high'},
    'synthetic_pattern_8': {'se': 'low', 'im': 'low', 'as': 'low'},
}

# 프롬프트 템플릿 (논문 4.2절 기반)
PROMPT_TEMPLATE_AGREE = """Question: Given a statement of you: "{statement}", Do you agree?
Answer: Yes"""

PROMPT_TEMPLATE_DISAGREE = """Question: Given a statement of you: "{statement}", Do you agree?
Answer: No"""

# Probe 학습 설정
DEFAULT_TRAIN_RATIO = 0.6  # 60% train, 40% validation
DEFAULT_NUM_HEADS = 48     # Top-K heads for intervention
PROBE_MAX_ITER = 1000      # LogisticRegression max iterations
PROBE_RANDOM_STATE = 42

# Alpha 탐색 설정
ALPHA_SEARCH_MIN = 0.0
ALPHA_SEARCH_MAX = 10.0
ALPHA_SEARCH_TOL = 0.01    # Golden section search tolerance
DEFAULT_ALPHA = 1.0

# 안전 상수
NORM_EPS = 1e-8            # Zero-division 방지용 epsilon

# 모델 설정
DEFAULT_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'

# Group name 매핑 (traintest_split_balanced_80_20.json의 by_group 키)
GROUP_TO_TRAIT = {
    'Academic Self-Efficacy': 'se',
    'Intrinsic Motivation': 'im',
    'Academic Stress': 'as'
}

TRAIT_TO_GROUP = {v: k for k, v in GROUP_TO_TRAIT.items()}
