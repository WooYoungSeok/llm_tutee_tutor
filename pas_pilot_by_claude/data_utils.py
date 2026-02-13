"""
data_utils.py
=============
데이터 로딩 및 프롬프트 포맷팅 - pas_pilot_by_claude용

8개 synthetic sample을 사용한 activation steering 데이터 처리
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import (
    TEST_SET_FILE, ITEM_KEY_FILE, TRAIN_TEST_SPLIT_FILE,
    TRAITS, TRAIT_NAMES, PATTERN_LABELS, GROUP_TO_TRAIT,
    PROMPT_TEMPLATE_AGREE, PROMPT_TEMPLATE_DISAGREE,
    PERSONA_PROMPTS, get_persona_prompt, get_combined_persona_prompt
)


@dataclass
class Statement:
    """문항 정보"""
    item_id: int
    text: str
    trait: str          # 'se', 'im', 'as'
    key: int            # +1 (positive keyed) or -1 (negative keyed)
    facet: str          # Full name like 'Academic Self-Efficacy'


@dataclass
class Sample:
    """개인 샘플 (8개 synthetic pattern 중 하나)"""
    case: str           # e.g., 'synthetic_pattern_1'
    pattern: Dict[str, str]  # {'SE': 'positive', 'IM': 'positive', 'AS': 'positive'}
    responses: Dict[int, int]  # item_id -> value (1=Inaccurate, 2=Accurate)

    def get_trait_level(self, trait: str) -> str:
        """특정 trait의 level (high/low) 반환"""
        trait_upper = trait.upper()
        pattern_val = self.pattern.get(trait_upper, 'positive')
        return 'high' if pattern_val == 'positive' else 'low'


@dataclass
class ActivationSample:
    """Activation 추출용 샘플"""
    statement: Statement
    sample: Sample       # 원본 샘플 참조
    prompt_agree: str    # Yes 응답 프롬프트
    prompt_disagree: str # No 응답 프롬프트
    label: int           # 1 = agree가 positive 방향, 0 = disagree가 positive 방향
    trait: str           # 해당 statement의 trait


def load_item_keys(filepath: str = ITEM_KEY_FILE) -> Dict[int, Statement]:
    """
    ItemKey Excel 로드하여 문항 정보 딕셔너리 반환

    Returns:
        {item_id: Statement} 매핑
    """
    df = pd.read_excel(filepath)
    items = {}

    for _, row in df.iterrows():
        item_id = int(row['Full#'])
        sign = row['Sign']  # '+se', '-im', etc.
        key = 1 if sign.startswith('+') else -1
        trait = row['Key']  # 'se', 'im', 'as'

        items[item_id] = Statement(
            item_id=item_id,
            text=row['Item'],
            trait=trait,
            key=key,
            facet=row['Facet']
        )

    return items


def load_samples(filepath: str = TEST_SET_FILE) -> List[Sample]:
    """
    Test-set.json 로드 (8개 synthetic samples)

    Returns:
        Sample 리스트
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for entry in data:
        responses = {}
        for key, value in entry.items():
            if key.startswith('i') and key[1:].isdigit():
                item_id = int(key[1:])
                responses[item_id] = value

        samples.append(Sample(
            case=entry['case'],
            pattern=entry.get('pattern', {}),
            responses=responses
        ))

    return samples


def load_train_test_split(filepath: str = TRAIN_TEST_SPLIT_FILE) -> Dict:
    """
    Train/test split 데이터 로드

    Returns:
        전체 split 데이터 딕셔너리
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    return split_data


def get_train_test_indices(filepath: str = TRAIN_TEST_SPLIT_FILE) -> Tuple[List[int], List[int]]:
    """
    전체 train/test index 반환
    """
    split_data = load_train_test_split(filepath)
    return split_data['train_index'], split_data['test_index']


def get_trait_indices(filepath: str = TRAIN_TEST_SPLIT_FILE, trait: str = None) -> Dict[str, Dict]:
    """
    Trait별 train/test index 반환

    Returns:
        {trait: {'train': [...], 'test': [...]}}
    """
    split_data = load_train_test_split(filepath)
    by_group = split_data.get('by_group', {})

    result = {}
    for group_name, group_data in by_group.items():
        trait_key = GROUP_TO_TRAIT.get(group_name)
        if trait_key:
            result[trait_key] = {
                'train': group_data.get('train_ids', []),
                'test': group_data.get('test_ids', [])
            }

    if trait:
        return {trait: result.get(trait, {'train': [], 'test': []})}

    return result


def create_activation_samples(
    items: Dict[int, Statement],
    sample: Sample,
    item_indices: List[int],
    traits: List[str] = None
) -> List[ActivationSample]:
    """
    주어진 sample과 item들로부터 activation 추출용 샘플 생성

    Args:
        items: {item_id: Statement} 매핑
        sample: 개인 샘플 (응답 포함)
        item_indices: 사용할 item ID 리스트
        traits: 필터링할 trait 리스트 (None이면 전체)

    Returns:
        ActivationSample 리스트
    """
    if traits is None:
        traits = TRAITS

    activation_samples = []

    for item_id in item_indices:
        if item_id not in items:
            continue

        statement = items[item_id]

        # Trait 필터링
        if statement.trait not in traits:
            continue

        # 응답값 확인
        response = sample.responses.get(item_id, 0)
        if response == 0:  # Unknown 응답은 제외
            continue

        # 프롬프트 생성
        prompt_agree = PROMPT_TEMPLATE_AGREE.format(statement=statement.text)
        prompt_disagree = PROMPT_TEMPLATE_DISAGREE.format(statement=statement.text)

        # Label 결정 (Key 기반 - 모든 sample에서 같은 item은 같은 label)
        # Positive-keyed item (key=+1): Agree = high trait → label=1
        # Negative-keyed item (key=-1): Disagree = high trait → label=0
        # 이렇게 하면 probe가 일관된 "high trait" direction을 학습
        label = 1 if statement.key == 1 else 0

        activation_samples.append(ActivationSample(
            statement=statement,
            sample=sample,
            prompt_agree=prompt_agree,
            prompt_disagree=prompt_disagree,
            label=label,
            trait=statement.trait
        ))

    return activation_samples


def create_trait_activation_samples(
    items: Dict[int, Statement],
    samples: List[Sample],
    train_indices: List[int],
    trait: str
) -> List[ActivationSample]:
    """
    특정 trait에 대한 모든 activation sample 수집

    Args:
        items: Statement 딕셔너리
        samples: 8개 Sample 리스트
        train_indices: 해당 trait의 train item indices
        trait: 'se', 'im', 'as'

    Returns:
        해당 trait의 모든 ActivationSample
    """
    all_samples = []

    for sample in samples:
        trait_samples = create_activation_samples(
            items=items,
            sample=sample,
            item_indices=train_indices,
            traits=[trait]
        )
        all_samples.extend(trait_samples)

    return all_samples


def get_sample_description(sample: Sample) -> str:
    """
    샘플의 trait level 설명 문자열 생성

    Example: "SE:high, IM:low, AS:high"
    """
    parts = []
    for trait in TRAITS:
        level = sample.get_trait_level(trait)
        parts.append(f"{trait.upper()}:{level}")
    return ", ".join(parts)


def format_evaluation_prompt(statement_text: str) -> str:
    """
    평가용 프롬프트 생성 (Yes/No 응답을 기대)
    """
    return f"""Question: Given a statement of you: "{statement_text}", Do you agree?
Answer:"""


def format_chat_template(
    tokenizer,
    user_prompt: str,
    model_name: str,
    system_prompt: str = "You are a helpful assistant.",
    add_generation_prompt: bool = False
) -> torch.Tensor:
    """
    모델에 맞는 chat template으로 토큰화

    Args:
        tokenizer: HuggingFace tokenizer
        user_prompt: 사용자 프롬프트
        model_name: 모델 이름
        system_prompt: 시스템 프롬프트 (persona prompt)
        add_generation_prompt: 생성용 프롬프트 추가 여부

    Returns:
        input_ids tensor
    """
    import torch

    if 'llama-3' in model_name.lower():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt
        )
    else:
        # Llama-2 형식
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        formatted = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_prompt} {E_INST}"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids

    return input_ids


def get_sample_persona_prompt(sample: Sample, trait: str) -> str:
    """
    특정 sample의 특정 trait에 대한 persona prompt 반환

    Args:
        sample: Sample 객체
        trait: 'se', 'im', 'as'

    Returns:
        해당 trait level의 persona prompt
    """
    level = sample.get_trait_level(trait)
    return get_persona_prompt(trait, level)


if __name__ == "__main__":
    # 테스트
    print("Loading data...")
    items = load_item_keys()
    print(f"Loaded {len(items)} items")

    samples = load_samples()
    print(f"Loaded {len(samples)} samples")

    # 8개 패턴 출력
    print("\n8 Synthetic Patterns:")
    for sample in samples:
        desc = get_sample_description(sample)
        print(f"  {sample.case}: {desc}")

    # Trait별 indices
    trait_indices = get_trait_indices()
    print("\nTrait indices:")
    for trait, indices in trait_indices.items():
        print(f"  {trait}: {len(indices['train'])} train, {len(indices['test'])} test")

    # 첫 번째 샘플로 activation sample 생성 테스트
    train_idx, test_idx = get_train_test_indices()
    act_samples = create_activation_samples(items, samples[0], train_idx)
    print(f"\nActivation samples for sample 0: {len(act_samples)}")

    # Trait별 분포
    for trait in TRAITS:
        count = sum(1 for s in act_samples if s.trait == trait)
        label_1 = sum(1 for s in act_samples if s.trait == trait and s.label == 1)
        print(f"  {trait}: {count} items, {label_1} label=1, {count-label_1} label=0")
