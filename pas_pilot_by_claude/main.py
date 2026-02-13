"""
main.py
=======
PAS (Personality Activation Steering) 메인 파이프라인 - pas_pilot_by_claude용

8개 synthetic sample을 사용하여 trait별 activation steering 학습

실행:
    # 모든 trait에 대해 학습
    python main.py --model_name meta-llama/Meta-Llama-3-8B-Instruct

    # 특정 trait만 학습
    python main.py --trait se

    # 4bit 양자화 사용
    python main.py --use_4bit
"""

import argparse
import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    SCRIPT_DIR, OUTPUT_DIR, TRAITS, TRAIT_NAMES,
    DEFAULT_MODEL, DEFAULT_NUM_HEADS, DEFAULT_TRAIN_RATIO, DEFAULT_ALPHA
)
from data_utils import (
    load_item_keys, load_samples, get_trait_indices,
    create_trait_activation_samples, get_sample_description
)
from activation_extractor import (
    ActivationExtractor, extract_paired_activations, validate_activations
)
from probe_trainer import train_and_select_heads, ProbeTrainer
from steering import ActivationSteering


def load_model(model_name: str, use_4bit: bool = False):
    """모델 및 토크나이저 로드"""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")

    return model, tokenizer


def train_trait_steering(
    model,
    tokenizer,
    model_name: str,
    trait: str,
    items: dict,
    samples: list,
    trait_indices: dict,
    num_heads: int = DEFAULT_NUM_HEADS,
    train_ratio: float = DEFAULT_TRAIN_RATIO
):
    """
    특정 trait에 대한 steering 학습

    Args:
        model: LLaMA 모델
        tokenizer: 토크나이저
        model_name: 모델 이름
        trait: 'se', 'im', 'as'
        items: Statement 딕셔너리
        samples: 8개 Sample 리스트
        trait_indices: trait별 train/test indices
        num_heads: Top-K heads 수
        train_ratio: probe 학습 train/val 비율

    Returns:
        (trainer, interventions) 튜플
    """
    print(f"\n{'#'*60}")
    print(f"Training steering for {TRAIT_NAMES[trait]} ({trait.upper()})")
    print(f"{'#'*60}")

    # 해당 trait의 train indices 가져오기
    train_indices = trait_indices[trait]['train']
    print(f"Using {len(train_indices)} train items for {trait}")

    # Activation samples 생성
    activation_samples = create_trait_activation_samples(
        items=items,
        samples=samples,
        train_indices=train_indices,
        trait=trait
    )
    print(f"Created {len(activation_samples)} activation samples")

    if len(activation_samples) == 0:
        print(f"ERROR: No activation samples created for {trait}")
        return None, {}

    # Label 분포 확인
    labels = [s.label for s in activation_samples]
    print(f"Labels: {sum(labels)} positive, {len(labels) - sum(labels)} negative")

    # Activation 추출
    print(f"\nExtracting activations...")
    extractor = ActivationExtractor(model, tokenizer, model_name)

    agree_acts, disagree_acts, labels_arr = extract_paired_activations(
        extractor, activation_samples, show_progress=True
    )

    # 유효성 검사
    print("\nValidating activations:")
    print("  Agree activations:")
    stats = validate_activations(agree_acts)
    print(f"    Shape: {agree_acts.shape}")
    print(f"    Finite: {stats['finite_ratio']*100:.1f}%, Mean: {stats['mean']:.4f}")

    print("  Disagree activations:")
    stats = validate_activations(disagree_acts)
    print(f"    Shape: {disagree_acts.shape}")
    print(f"    Finite: {stats['finite_ratio']*100:.1f}%, Mean: {stats['mean']:.4f}")

    # Probe 학습 및 top heads 선택
    print(f"\nTraining probes and selecting top {num_heads} heads...")
    trainer, interventions = train_and_select_heads(
        agree_activations=agree_acts,
        disagree_activations=disagree_acts,
        labels=labels_arr,
        num_layers=extractor.num_layers,
        num_heads=extractor.num_heads,
        head_dim=extractor.head_dim,
        num_to_select=num_heads,
        train_ratio=train_ratio
    )

    return trainer, interventions


def save_trait_results(
    output_dir: str,
    trait: str,
    trainer,
    interventions: dict,
    config: dict,
    timestamp: str = None
):
    """
    Trait별 결과 저장 (파일명에 trait 포함)

    Args:
        output_dir: 출력 디렉토리
        trait: trait 이름
        trainer: ProbeTrainer
        interventions: intervention 딕셔너리
        config: 설정 정보
        timestamp: 타임스탬프 (없으면 자동 생성)
    """
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Interventions 저장 - trait 이름 포함
    interventions_file = os.path.join(
        output_dir, f"interventions_{trait}_{timestamp}.pkl"
    )
    with open(interventions_file, 'wb') as f:
        pickle.dump(interventions, f)
    print(f"Saved interventions to {interventions_file}")

    # Summary 저장
    summary = {
        'timestamp': timestamp,
        'trait': trait,
        'trait_name': TRAIT_NAMES.get(trait, trait),
        'config': config,
        'probe_stats': trainer.get_accuracy_stats() if trainer else {},
        'num_intervention_layers': len(interventions),
        'num_intervention_heads': sum(len(h) for h in interventions.values()),
    }

    summary_file = os.path.join(
        output_dir, f"summary_{trait}_{timestamp}.json"
    )
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved summary to {summary_file}")

    return interventions_file, summary_file


def main():
    parser = argparse.ArgumentParser(description="PAS Activation Steering Training")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS,
                        help="Number of top heads to select for intervention")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="Train/validation split ratio for probes")
    parser.add_argument("--trait", type=str, default=None, choices=TRAITS,
                        help="Specific trait to train (default: all)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("PAS (Personality Activation Steering) Training Pipeline")
    print(f"{'#'*60}")
    print(f"Model: {args.model_name}")
    print(f"Top K heads: {args.num_heads}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Target trait: {args.trait if args.trait else 'all'}")
    print(f"Output dir: {args.output_dir}")

    # 데이터 로드
    print(f"\n{'='*60}")
    print("Loading data")
    print(f"{'='*60}")

    items = load_item_keys()
    print(f"Loaded {len(items)} items from ItemKey")

    samples = load_samples()
    print(f"Loaded {len(samples)} samples (8 synthetic patterns)")

    # 8개 패턴 출력
    print("\n8 Synthetic Patterns:")
    for sample in samples:
        desc = get_sample_description(sample)
        print(f"  {sample.case}: {desc}")

    # Trait별 indices
    trait_indices = get_trait_indices()
    print("\nTrait indices:")
    for trait, indices in trait_indices.items():
        print(f"  {trait.upper()}: {len(indices['train'])} train, {len(indices['test'])} test")

    # 모델 로드
    model, tokenizer = load_model(args.model_name, use_4bit=args.use_4bit)

    # 학습할 trait 결정
    traits_to_train = [args.trait] if args.trait else TRAITS

    # 타임스탬프 (모든 trait에 동일하게 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Config 정보
    config = {
        'model_name': args.model_name,
        'num_heads': args.num_heads,
        'train_ratio': args.train_ratio,
        'num_samples': len(samples),
    }

    # Trait별 학습
    all_results = {}
    for trait in traits_to_train:
        trainer, interventions = train_trait_steering(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            trait=trait,
            items=items,
            samples=samples,
            trait_indices=trait_indices,
            num_heads=args.num_heads,
            train_ratio=args.train_ratio
        )

        if interventions:
            # 결과 저장
            save_trait_results(
                output_dir=args.output_dir,
                trait=trait,
                trainer=trainer,
                interventions=interventions,
                config=config,
                timestamp=timestamp
            )
            all_results[trait] = {
                'interventions': interventions,
                'stats': trainer.get_accuracy_stats() if trainer else {}
            }
        else:
            print(f"WARNING: No interventions found for {trait}")

    # 전체 요약
    print(f"\n{'#'*60}")
    print("Training Complete!")
    print(f"{'#'*60}")

    for trait, result in all_results.items():
        stats = result.get('stats', {})
        n_heads = sum(len(h) for h in result['interventions'].values())
        n_layers = len(result['interventions'])
        print(f"\n{TRAIT_NAMES[trait]} ({trait.upper()}):")
        print(f"  Intervention heads: {n_heads} across {n_layers} layers")
        if stats:
            print(f"  Mean val accuracy: {stats.get('mean_val_acc', 0):.4f}")
            print(f"  Above random (>0.55): {stats.get('above_random', 0)*100:.1f}%")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
