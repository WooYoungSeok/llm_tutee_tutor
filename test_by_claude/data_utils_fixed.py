"""
data_utils.py
=============
ë°ì´í„° ë¡œë”© ë° í”„ë¡¬í”„íŠ¸ í¬ë§·íŒ…

ë…¼ë¬¸ 4.2ì ˆ:
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
    """ë¬¸í•­ ì •ë³´"""
    item_id: int
    text: str
    trait: str          # 'se', 'im', 'as'
    key: int            # +1 (positive keyed) or -1 (negative keyed)
    facet: str          # Full name like 'Academic Self-Efficacy'


@dataclass
class Sample:
    """ê°œì¸ ìƒ˜í”Œ"""
    case: str
    pattern: str
    responses: Dict[int, int]  # item_id -> value (1=Inaccurate, 2=Accurate)


@dataclass
class ActivationSample:
    """Activation ì¶”ì¶œìš© ìƒ˜í”Œ"""
    statement: Statement
    prompt_agree: str       # Yes ì‘ë‹µ í”„ë¡¬í”„íŠ¸
    prompt_disagree: str    # No ì‘ë‹µ í”„ë¡¬í”„íŠ¸
    label: int              # 1 = agreeê°€ positive ë°©í–¥, 0 = disagreeê°€ positive ë°©í–¥


def load_item_keys(filepath: str = ITEM_KEY_FILE) -> Dict[int, Statement]:
    """
    ItemKey Excel ë¡œë“œí•˜ì—¬ ë¬¸í•­ ì •ë³´ ë”•ì…”ë„ˆë¦¬ ë°˜í™˜

    Returns:
        {item_id: Statement} ë§¤í•‘
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
    Test-set.json ë¡œë“œ

    Returns:
        Sample ë¦¬ìŠ¤íŠ¸
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
    Train/test split ì¸ë±ìŠ¤ ë¡œë“œ

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
    ì£¼ì–´ì§„ sampleê³¼ itemë“¤ë¡œë¶€í„° activation ì¶”ì¶œìš© ìƒ˜í”Œ ìƒì„±

    ë…¼ë¬¸ ë°©ì‹:
    - Positive sample (label=1): traitì„ ê°€ì§„ ì‚¬ëžŒì´ ë™ì˜í•  ë•Œ â†’ "Answer: Yes"ê°€ positive ë°©í–¥
    - Negative sample (label=0): traitì„ ê°€ì§„ ì‚¬ëžŒì´ ë™ì˜í•˜ì§€ ì•Šì„ ë•Œ â†’ "Answer: No"ê°€ positive ë°©í–¥

    Keyì™€ Responseì˜ ì¡°í•©ìœ¼ë¡œ label ê²°ì •:
    - key=+1 (positive keyed item): Accurate(2) ì‘ë‹µ = ë†’ì€ trait = Agreeê°€ positive
    - key=-1 (negative keyed item): Accurate(2) ì‘ë‹µ = ë‚®ì€ trait = Disagreeê°€ positive

    Args:
        items: {item_id: Statement} ë§¤í•‘
        sample: ê°œì¸ ìƒ˜í”Œ (ì‘ë‹µ í¬í•¨)
        item_indices: ì‚¬ìš©í•  item ID ë¦¬ìŠ¤íŠ¸
        traits: í•„í„°ë§í•  trait ë¦¬ìŠ¤íŠ¸ (Noneì´ë©´ ì „ì²´)

    Returns:
        ActivationSample ë¦¬ìŠ¤íŠ¸
    """
    if traits is None:
        traits = TRAITS

    activation_samples = []

    for item_id in item_indices:
        if item_id not in items:
            continue

        statement = items[item_id]

        # Trait í•„í„°ë§
        if statement.trait not in traits:
            continue

        # ì‘ë‹µê°’ í™•ì¸
        response = sample.responses.get(item_id, 0)
        if response == 0:  # Unknown ì‘ë‹µì€ ì œì™¸
            continue

        # í”„ë¡¬í”„íŠ¸ ìƒì„±
        prompt_agree = PROMPT_TEMPLATE_AGREE.format(statement=statement.text)
        prompt_disagree = PROMPT_TEMPLATE_DISAGREE.format(statement=statement.text)


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
    ëª¨ë¸ì— ë§žê²Œ í”„ë¡¬í”„íŠ¸ë¥¼ í† í°í™”

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: ì›ë³¸ í”„ë¡¬í”„íŠ¸
        model_name: ëª¨ë¸ ì´ë¦„ (llama-3 ì—¬ë¶€ í™•ì¸ìš©)
        system_prompt: ì‹œìŠ¤í…œ í”„ë¡¬í”„íŠ¸

    Returns:
        token_ids ë¦¬ìŠ¤íŠ¸
    """
    if 'llama-3' in model_name.lower():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    else:
        # Llama-2 í˜•ì‹
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
    íŠ¹ì • traitì— ëŒ€í•œ ëª¨ë“  activation sample ìˆ˜ì§‘

    Args:
        items: Statement ë”•ì…”ë„ˆë¦¬
        samples: Sample ë¦¬ìŠ¤íŠ¸
        train_indices: ì‚¬ìš©í•  item indices
        trait: 'se', 'im', 'as'

    Returns:
        í•´ë‹¹ traitì˜ ëª¨ë“  ActivationSample
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
    # í…ŒìŠ¤íŠ¸
    print("Loading data...")
    items = load_item_keys()
    print(f"Loaded {len(items)} items")

    samples = load_samples()
    print(f"Loaded {len(samples)} samples")

    train_idx, test_idx = load_train_test_split()
    print(f"Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    # ì²« ë²ˆì§¸ ìƒ˜í”Œë¡œ activation sample ìƒì„± í…ŒìŠ¤íŠ¸
    act_samples = create_activation_samples(items, samples[0], train_idx)
    print(f"\nActivation samples for sample 0: {len(act_samples)}")

    # Traitë³„ ë¶„í¬
    for trait in TRAITS:
        count = sum(1 for s in act_samples if s.statement.trait == trait)
        label_1 = sum(1 for s in act_samples if s.statement.trait == trait and s.label == 1)
        print(f"  {trait}: {count} items, {label_1} label=1, {count-label_1} label=0")

    # ìƒ˜í”Œ ì¶œë ¥
    if act_samples:
        s = act_samples[0]
        print(f"\nExample activation sample:")
        print(f"  Statement: {s.statement.text[:50]}...")
        print(f"  Trait: {s.statement.trait}, Key: {s.statement.key}")
        print(f"  Label: {s.label}")
        print(f"  Prompt (agree): {s.prompt_agree[:80]}...")
