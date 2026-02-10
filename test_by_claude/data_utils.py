"""
data_utils.py
=============
데이터 로딩 및 프롬프트 포맷팅

논문 4.2절:
- Positive sample: "Question: Given a statement of you: '$Statement', Do you agree? Answer: Yes"
- Negative sample: "Question: Given a statement of you: '$Statement', Do you agree? Answer: No"
"""

import json
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

from config import (
    TEST_SET_FILE, ITEM_KEY_FILE, TRAIN_TEST_SPLIT_FILE,
    TRAITS, PROMPT_TEMPLATE_AGREE, PROMPT_TEMPLATE_DISAGREE
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
    """개인 샘플"""
    case: str
    pattern: str
    responses: Dict[int, int]  # item_id -> value (1=Inaccurate, 2=Accurate)


@dataclass
class ActivationSample:
    """Activation 추출용 샘플"""
    statement: Statement
    prompt_agree: str       # Yes 응답 프롬프트
    prompt_disagree: str    # No 응답 프롬프트
    label: int              # 1 = agree가 positive 방향, 0 = disagree가 positive 방향


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
    Test-set.json 로드

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
            pattern=entry.get('pattern', ''),
            responses=responses
        ))

    return samples


def load_train_test_split(filepath: str = TRAIN_TEST_SPLIT_FILE) -> Tuple[List[int], List[int]]:
    """
    Train/test split 인덱스 로드

    Returns:
        (train_indices, test_indices)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    return split_data['train_index'], split_data['test_index']


def create_activation_samples(
    items: Dict[int, Statement],
    sample: Sample,
    item_indices: List[int],
    traits: List[str] = None
) -> List[ActivationSample]:
    """
    주어진 sample과 item들로부터 activation 추출용 샘플 생성

    논문 방식:
    - Positive sample (label=1): trait을 가진 사람이 동의할 때 → "Answer: Yes"가 positive 방향
    - Negative sample (label=0): trait을 가진 사람이 동의하지 않을 때 → "Answer: No"가 positive 방향

    Key와 Response의 조합으로 label 결정:
    - key=+1 (positive keyed item): Accurate(2) 응답 = 높은 trait = Agree가 positive
    - key=-1 (negative keyed item): Accurate(2) 응답 = 낮은 trait = Disagree가 positive

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

        # Label 결정
        # response=2 (Accurate)이고 key=+1이면: 높은 trait → Agree가 positive (label=1)
        # response=2 (Accurate)이고 key=-1이면: 낮은 trait → Disagree가 positive (label=0)
        # response=1 (Inaccurate)이고 key=+1이면: 낮은 trait → Disagree가 positive (label=0)
        # response=1 (Inaccurate)이고 key=-1이면: 높은 trait → Agree가 positive (label=1)

        is_accurate = (response == 2)
        is_positive_keyed = (statement.key == 1)

        # XOR 로직: (Accurate AND positive_keyed) OR (Inaccurate AND negative_keyed)
        # → Agree가 positive 방향
        label = 1 if (is_accurate == is_positive_keyed) else 0

        activation_samples.append(ActivationSample(
            statement=statement,
            prompt_agree=prompt_agree,
            prompt_disagree=prompt_disagree,
            label=label
        ))

    return activation_samples


def format_chat_prompt(
    tokenizer,
    prompt: str,
    model_name: str,
    system_prompt: str = "You are a helpful assistant."
) -> List[int]:
    """
    모델에 맞게 프롬프트를 토큰화

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: 원본 프롬프트
        model_name: 모델 이름 (llama-3 여부 확인용)
        system_prompt: 시스템 프롬프트

    Returns:
        token_ids 리스트
    """
    if 'llama-3' in model_name.lower():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    else:
        # Llama-2 형식
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        formatted = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{prompt} {E_INST}"
        return tokenizer(formatted).input_ids


def get_trait_samples(
    items: Dict[int, Statement],
    samples: List[Sample],
    train_indices: List[int],
    trait: str
) -> List[ActivationSample]:
    """
    특정 trait에 대한 모든 activation sample 수집

    Args:
        items: Statement 딕셔너리
        samples: Sample 리스트
        train_indices: 사용할 item indices
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


if __name__ == "__main__":
    # 테스트
    print("Loading data...")
    items = load_item_keys()
    print(f"Loaded {len(items)} items")

    samples = load_samples()
    print(f"Loaded {len(samples)} samples")

    train_idx, test_idx = load_train_test_split()
    print(f"Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    # 첫 번째 샘플로 activation sample 생성 테스트
    act_samples = create_activation_samples(items, samples[0], train_idx)
    print(f"\nActivation samples for sample 0: {len(act_samples)}")

    # Trait별 분포
    for trait in TRAITS:
        count = sum(1 for s in act_samples if s.statement.trait == trait)
        label_1 = sum(1 for s in act_samples if s.statement.trait == trait and s.label == 1)
        print(f"  {trait}: {count} items, {label_1} label=1, {count-label_1} label=0")

    # 샘플 출력
    if act_samples:
        s = act_samples[0]
        print(f"\nExample activation sample:")
        print(f"  Statement: {s.statement.text[:50]}...")
        print(f"  Trait: {s.statement.trait}, Key: {s.statement.key}")
        print(f"  Label: {s.label}")
        print(f"  Prompt (agree): {s.prompt_agree[:80]}...")
