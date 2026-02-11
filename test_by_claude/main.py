"""
main.py
=======
PAS (Personality Activation Steering) 메인 파이프라인

실행:
    python main.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_heads 48
"""

import argparse
import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from config import (
    SCRIPT_DIR, OUTPUT_DIR, TRAITS, TRAIT_NAMES,
    DEFAULT_MODEL, DEFAULT_NUM_HEADS, DEFAULT_TRAIN_RATIO
)
from data_utils import (
    load_item_keys, load_samples, load_train_test_split,
    create_activation_samples, get_trait_samples
)
from activation_extractor import (
    ActivationExtractor, extract_paired_activations, validate_activations
)
from probe_trainer import train_and_select_heads, ProbeTrainer
from steering import ActivationSteering, grid_search_alpha


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


def step1_load_data():
    """Step 1: 데이터 로드"""
    print(f"\n{'='*60}")
    print("Step 1: Loading data")
    print(f"{'='*60}")

    items = load_item_keys()
    print(f"  Loaded {len(items)} items from ItemKey")

    samples = load_samples()
    print(f"  Loaded {len(samples)} samples from Test-set")

    train_idx, test_idx = load_train_test_split()
    print(f"  Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    # Trait별 item 분포
    for trait in TRAITS:
        count = sum(1 for idx in train_idx if idx in items and items[idx].trait == trait)
        print(f"  {trait.upper()}: {count} train items")

    return items, samples, train_idx, test_idx


def step2_create_activation_samples(items, samples, train_idx, trait: str = None):
    """Step 2: Activation 추출용 샘플 생성"""
    print(f"\n{'='*60}")
    print(f"Step 2: Creating activation samples" + (f" for {trait}" if trait else ""))
    print(f"{'='*60}")

    all_act_samples = []

    for sample_idx, sample in enumerate(samples):
        traits_to_use = [trait] if trait else TRAITS
        act_samples = create_activation_samples(
            items=items,
            sample=sample,
            item_indices=train_idx,
            traits=traits_to_use
        )
        all_act_samples.extend(act_samples)
        print(f"  Sample {sample_idx}: {len(act_samples)} activation samples")

    # Label 분포 확인
    labels = [s.label for s in all_act_samples]
    print(f"\n  Total: {len(all_act_samples)} activation samples")
    print(f"  Labels: {sum(labels)} positive, {len(labels) - sum(labels)} negative")

    return all_act_samples


def step3_extract_activations(extractor, activation_samples):
    """Step 3: Activation 추출"""
    print(f"\n{'='*60}")
    print("Step 3: Extracting activations")
    print(f"{'='*60}")

    agree_acts, disagree_acts, labels = extract_paired_activations(
        extractor, activation_samples, show_progress=True
    )

    # 유효성 검사
    print("\n  Agree activations:")
    stats = validate_activations(agree_acts)
    print(f"    Shape: {agree_acts.shape}")
    print(f"    Finite: {stats['finite_ratio']*100:.1f}%, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    print("  Disagree activations:")
    stats = validate_activations(disagree_acts)
    print(f"    Shape: {disagree_acts.shape}")
    print(f"    Finite: {stats['finite_ratio']*100:.1f}%, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    return agree_acts, disagree_acts, labels


def step4_train_probes(agree_acts, disagree_acts, labels, num_layers, num_heads, head_dim, num_to_select):
    """Step 4: Probe 학습 및 direction 탐색"""
    print(f"\n{'='*60}")
    print("Step 4: Training probes and selecting top heads")
    print(f"{'='*60}")

    trainer, interventions = train_and_select_heads(
        agree_activations=agree_acts,
        disagree_activations=disagree_acts,
        labels=labels,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_to_select=num_to_select
    )

    return trainer, interventions


def step5_apply_steering(steering, interventions, alpha):
    """Step 5: Steering 적용"""
    print(f"\n{'='*60}")
    print(f"Step 5: Applying steering with α={alpha}")
    print(f"{'='*60}")

    steering.apply_steering(interventions, alpha=alpha)

    # Intervention 요약
    total_heads = sum(len(heads) for heads in interventions.values())
    print(f"  Applied interventions to {total_heads} heads across {len(interventions)} layers")


def step6_generate_samples(steering, test_prompts, alpha):
    """Step 6: 생성 테스트"""
    print(f"\n{'='*60}")
    print(f"Step 6: Generating sample outputs (α={alpha})")
    print(f"{'='*60}")

    results = []

    for prompt in test_prompts[:3]:  # 처음 3개만
        print(f"\n  Prompt: {prompt[:50]}...")

        # Steering 없이 생성
        steering.reset()
        output_base = steering.generate(prompt, max_new_tokens=30)
        print(f"  [Base] {output_base[:80]}...")

        # Steering 적용 후 생성
        steering.apply_steering(steering.interventions, alpha=alpha) if hasattr(steering, 'interventions') else None
        output_steered = steering.generate(prompt, max_new_tokens=30)
        print(f"  [Steered] {output_steered[:80]}...")

        results.append({
            'prompt': prompt,
            'base': output_base,
            'steered': output_steered
        })

    return results


def save_results(output_dir, trainer, interventions, results, config):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Interventions 저장
    interventions_file = os.path.join(output_dir, f"interventions_{timestamp}.pkl")
    with open(interventions_file, 'wb') as f:
        pickle.dump(interventions, f)
    print(f"  Saved interventions to {interventions_file}")

    # Config 및 결과 저장
    summary = {
        'timestamp': timestamp,
        'config': config,
        'probe_stats': trainer.get_accuracy_stats() if trainer else {},
        'num_intervention_layers': len(interventions),
        'num_intervention_heads': sum(len(h) for h in interventions.values()),
        'sample_results': results if results else []
    }

    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="PAS Activation Steering")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS,
                        help="Number of top heads to select for intervention")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Intervention strength")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="Train/validation split ratio")
    parser.add_argument("--trait", type=str, default=None, choices=TRAITS,
                        help="Specific trait to target (default: all)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation step")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("PAS (Personality Activation Steering) Pipeline")
    print(f"{'#'*60}")
    print(f"Model: {args.model_name}")
    print(f"Top K heads: {args.num_heads}")
    print(f"Alpha: {args.alpha}")
    print(f"Trait: {args.trait if args.trait else 'all'}")

    # Step 1: 데이터 로드
    items, samples, train_idx, test_idx = step1_load_data()

    # 모델 로드
    model, tokenizer = load_model(args.model_name, use_4bit=args.use_4bit)

    # Extractor 초기화
    extractor = ActivationExtractor(model, tokenizer, args.model_name)
    print(f"\nModel config: {extractor.num_layers} layers, {extractor.num_heads} heads, {extractor.head_dim} head_dim")

    # Step 2: Activation 샘플 생성
    activation_samples = step2_create_activation_samples(items, samples, train_idx, args.trait)

    if len(activation_samples) == 0:
        print("\nERROR: No activation samples created. Check your data.")
        return

    # Step 3: Activation 추출
    agree_acts, disagree_acts, labels = step3_extract_activations(extractor, activation_samples)

    # Step 4: Probe 학습
    trainer, interventions = step4_train_probes(
        agree_acts, disagree_acts, labels,
        num_layers=extractor.num_layers,
        num_heads=extractor.num_heads,
        head_dim=extractor.head_dim,
        num_to_select=args.num_heads
    )

    if len(interventions) == 0:
        print("\nWARNING: No interventions found. All probes may have failed.")
        print("Check activation extraction and probe training.")
        return

    # Step 5: Steering 초기화 및 적용
    steering = ActivationSteering(model, tokenizer, args.model_name)
    step5_apply_steering(steering, interventions, args.alpha)

    # Step 6: 생성 테스트 (선택적)
    results = []
    if not args.skip_generation:
        test_prompts = [
            "Tell me about yourself.",
            "How do you feel about mathematics?",
            "Do you enjoy learning new things?",
        ]
        results = step6_generate_samples(steering, test_prompts, args.alpha)

    # 결과 저장
    print(f"\n{'='*60}")
    print("Saving results")
    print(f"{'='*60}")

    config = {
        'model_name': args.model_name,
        'num_heads': args.num_heads,
        'alpha': args.alpha,
        'train_ratio': args.train_ratio,
        'trait': args.trait,
        'num_samples': len(samples),
        'num_activation_samples': len(activation_samples)
    }

    save_results(args.output_dir, trainer, interventions, results, config)

    print(f"\n{'#'*60}")
    print("Pipeline completed!")
    print(f"{'#'*60}")

    # Reset
    steering.reset()


if __name__ == "__main__":
    main()
