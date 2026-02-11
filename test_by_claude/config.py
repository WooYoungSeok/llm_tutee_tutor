"""
config.py
=========
PAS 실험 설정 상수
"""

import os

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'pas_pilot')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

# 데이터 파일
TEST_SET_FILE = os.path.join(DATA_DIR, 'Test-set.json')
ITEM_KEY_FILE = os.path.join(DATA_DIR, 'all_data_ItemKey.xlsx')
TRAIN_TEST_SPLIT_FILE = os.path.join(DATA_DIR, 'traintest_split_balanced_80_20.json')

# Trait 정보
TRAITS = ['se', 'im', 'as']
TRAIT_NAMES = {
    'se': 'Self-Efficacy',
    'im': 'Intrinsic Motivation',
    'as': 'Academic Stress'
}

# 프롬프트 템플릿 (논문 4.2절 기반)
# Positive: 해당 trait을 가진 사람이 동의할 문장
# Negative: 해당 trait을 가진 사람이 동의하지 않을 문장
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

# 안전 상수
NORM_EPS = 1e-8            # Zero-division 방지용 epsilon

# 모델 설정
DEFAULT_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
