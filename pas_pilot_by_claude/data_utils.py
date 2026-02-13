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
    SYSTEM_PROMPTS
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
    system_prompt: str = ""  # 샘플의 personality system prompt


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


def get_system_prompt_for_sample(sample: 'Sample', traits: list = None) -> str:
    """
    Sample의 trait level 조합에 맞는 system prompt 생성

    Args:
        sample: 개인 샘플 (pattern 포함)
        traits: 포함할 trait 리스트 (None이면 전체 TRAITS)

    Returns:
        결합된 system prompt 문자열
    """
    if traits is None:
        traits = TRAITS

    parts = []
    for trait in traits:
        level = sample.get_trait_level(trait)
        prompt = SYSTEM_PROMPTS.get((trait, level), "")
        if prompt:
            parts.append(prompt)

    return "\n\n".join(parts)


def get_system_prompt_for_levels(trait_levels: dict) -> str:
    """
    {trait: level} 딕셔너리로 system prompt 생성

    Args:
        trait_levels: {'se': 'high', 'im': 'low', 'as': 'high'} 형식

    Returns:
        결합된 system prompt 문자열
    """
    parts = []
    for trait in TRAITS:
        level = trait_levels.get(trait)
        if level is None:
            continue
        prompt = SYSTEM_PROMPTS.get((trait, level), "")
        if prompt:
            parts.append(prompt)

    return "\n\n".join(parts)


def format_chat_prompt(
    system_prompt: str,
    user_text: str,
    answer: str = None,
    tokenizer=None,
    model_name: str = None
) -> str:
    """
    Chat template 형식으로 프롬프트 포맷

    Args:
        system_prompt: 시스템 프롬프트
        user_text: 사용자 메시지
        answer: 어시스턴트 답변 (None이면 generation prompt 추가)
        tokenizer: HuggingFace tokenizer (있으면 apply_chat_template 사용)
        model_name: 모델 이름 (fallback용)

    Returns:
        포맷된 프롬프트 문자열
    """
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_text})
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # Fallback: LLaMA-3 형식 수동 포맷
    if model_name and 'llama-3' in model_name.lower():
        header = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        if answer is not None:
            return header + answer + "<|eot_id|>"
        return header

    # 최후 fallback: 단순 텍스트
    if answer is not None:
        return f"{user_text}\nAnswer: {answer}"
    return f"{user_text}\nAnswer:"


def create_activation_samples(
    items: Dict[int, Statement],
    sample: Sample,
    item_indices: List[int],
    traits: List[str] = None,
    tokenizer=None,
    model_name: str = None
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

    # 샘플의 personality system prompt 생성 (해당 trait만 포함)
    sys_prompt = get_system_prompt_for_sample(sample, traits=traits)

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

        # 프롬프트 생성 (chat template 적용)
        user_text = f'Question: Given a statement of you: "{statement.text}", Do you agree?'
        prompt_agree = format_chat_prompt(
            system_prompt=sys_prompt,
            user_text=user_text,
            answer="Yes",
            tokenizer=tokenizer,
            model_name=model_name
        )
        prompt_disagree = format_chat_prompt(
            system_prompt=sys_prompt,
            user_text=user_text,
            answer="No",
            tokenizer=tokenizer,
            model_name=model_name
        )

        # Label 결정 (Activation Steering용)
        # 목표: 항상 "높은 trait" 방향을 positive로 설정
        # - Key=+1 (positive keyed): Agree = 높은 trait → label=1
        # - Key=-1 (negative keyed): Disagree = 높은 trait → label=0
        #
        # 주의: 개인의 실제 response는 사용하지 않음!
        # 모든 샘플에서 일관된 "높은 trait" direction을 학습하기 위함
        label = 1 if statement.key == 1 else 0

        activation_samples.append(ActivationSample(
            statement=statement,
            sample=sample,
            prompt_agree=prompt_agree,
            prompt_disagree=prompt_disagree,
            label=label,
            trait=statement.trait,
            system_prompt=sys_prompt
        ))

    return activation_samples


def create_trait_activation_samples(
    items: Dict[int, Statement],
    samples: List[Sample],
    train_indices: List[int],
    trait: str,
    tokenizer=None,
    model_name: str = None
) -> List[ActivationSample]:
    """
    특정 trait에 대한 모든 activation sample 수집

    Args:
        items: Statement 딕셔너리
        samples: 8개 Sample 리스트
        train_indices: 해당 trait의 train item indices
        trait: 'se', 'im', 'as'
        tokenizer: chat template 적용용 tokenizer
        model_name: 모델 이름

    Returns:
        해당 trait의 모든 ActivationSample
    """
    all_samples = []

    for sample in samples:
        trait_samples = create_activation_samples(
            items=items,
            sample=sample,
            item_indices=train_indices,
            traits=[trait],
            tokenizer=tokenizer,
            model_name=model_name
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


def format_evaluation_prompt(
    statement_text: str,
    system_prompt: str = "",
    tokenizer=None,
    model_name: str = None
) -> str:
    """
    평가용 프롬프트 생성 (Yes/No 응답을 기대)

    Args:
        statement_text: 문항 텍스트
        system_prompt: personality system prompt
        tokenizer: chat template 적용용 tokenizer
        model_name: 모델 이름
    """
    user_text = f'Question: Given a statement of you: "{statement_text}", Do you agree?'

    if system_prompt and (tokenizer is not None or model_name):
        return format_chat_prompt(
            system_prompt=system_prompt,
            user_text=user_text,
            answer=None,
            tokenizer=tokenizer,
            model_name=model_name
        )

    # fallback: 기존 단순 텍스트 형식
    return f'{user_text}\nAnswer:'


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
