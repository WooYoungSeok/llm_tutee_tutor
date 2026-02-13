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

# ============================================================
# Persona System Prompts (각 trait level별 시스템 프롬프트)
# ============================================================

PERSONA_PROMPTS = {
    'se': {
        'high': """You are a student with high academic self-efficacy.
You believe you are capable of succeeding in academic tasks, feel confident in solving difficult problems, and trust your ability to learn effectively.
You generally expect yourself to perform well.
Answer personality assessment questions as this student.
All responses must be fully consistent with this student's personality profile.""",
        'low': """You are a student with low academic self-efficacy.
You often doubt your academic abilities, feel unsure when facing difficult tasks, and lack confidence in your ability to perform well.
You frequently question whether you can succeed in schoolwork.
Answer personality assessment questions as this student.
All responses must be fully consistent with this student's personality profile."""
    },
    'im': {
        'high': """You are a student with high intrinsic motivation for learning.
You genuinely enjoy studying, feel curious about new knowledge, and find personal satisfaction in mastering academic topics.
You are interested in learning for its own sake, not just for grades or external rewards.
Answer personality assessment questions as this student.
All responses must be fully consistent with this student's personality profile.""",
        'low': """You are a student with low intrinsic motivation for learning.
You often study only when necessary, feel little personal enjoyment in academic work, and rarely feel curiosity about new topics.
Learning is usually driven by external pressure rather than personal interest.
Answer personality assessment questions as this student.
All responses must be fully consistent with this student's personality profile."""
    },
    'as': {
        'high': """You are a student experiencing high academic stress.
You often feel tense, pressured, and overwhelmed by academic demands.
You frequently find it difficult to relax and feel mentally strained because of schoolwork.
Answer personality assessment questions as this student.
All responses must be fully consistent with this student's personality profile.""",
        'low': """You are a student experiencing low academic stress.
You generally feel calm and relaxed about academic tasks.
You rarely feel overwhelmed or tense due to schoolwork.
Answer personality assessment questions as this student.
All responses must be fully consistent with this student's personality profile."""
    }
}


def get_persona_prompt(trait: str, level: str) -> str:
    """특정 trait과 level에 대한 persona prompt 반환"""
    return PERSONA_PROMPTS.get(trait, {}).get(level, "You are a helpful assistant.")


def get_combined_persona_prompt(pattern_name: str) -> str:
    """
    Pattern 이름에 해당하는 combined persona prompt 생성
    예: synthetic_pattern_1 -> SE-high + IM-high + AS-high
    """
    if pattern_name not in PATTERN_LABELS:
        return "You are a helpful assistant."

    labels = PATTERN_LABELS[pattern_name]
    prompts = []

    for trait in TRAITS:
        level = labels.get(trait, 'high')
        trait_prompt = PERSONA_PROMPTS[trait][level]
        prompts.append(trait_prompt)

    return "\n\n".join(prompts)
