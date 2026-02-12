"""
evaluate.py
===========
Base 모델과 Steered 모델의 test set 정답률 비교

실행:
    # 특정 trait 평가 (intervention 파일 필요)
    python evaluate.py --trait se --intervention_file outputs/interventions_se_*.pkl

    # 모든 trait 평가
    python evaluate.py --evaluate_all

    # Alpha 값 조정
    python evaluate.py --trait im --alpha 2.0
"""

import argparse
import os
import json
import pickle
import numpy as np
import torch
from glob import glob
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    OUTPUT_DIR, TRAITS, TRAIT_NAMES, DEFAULT_MODEL, DEFAULT_ALPHA
)
from data_utils import (
    load_item_keys, load_samples, get_trait_indices,
    format_evaluation_prompt, get_sample_description
)
from steering import ActivationSteering


def load_model(model_name: str, use_4bit: bool = False):
    """모델 및 토크나이저 로드"""
    print(f"Loading model: {model_name}")

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
    return model, tokenizer


def get_model_response(
    model,
    tokenizer,
    prompt: str,
    model_name: str
) -> str:
    """
    모델의 Yes/No 응답 추출

    Returns:
        'Yes', 'No', or 'Unknown'
    """
    device = next(model.parameters()).device

    # 프롬프트 포맷팅
    if 'llama-3' in model_name.lower():
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer with only 'Yes' or 'No'."},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    else:
        formatted = f"[INST] <<SYS>>\nAnswer with only 'Yes' or 'No'.\n<</SYS>>\n\n{prompt} [/INST]"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids

    input_ids = input_ids.to(device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 디코딩
    generated_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    # Yes/No 판별
    if 'yes' in response:
        return 'Yes'
    elif 'no' in response:
        return 'No'
    else:
        return 'Unknown'


def evaluate_sample_accuracy(
    model,
    tokenizer,
    model_name: str,
    items: dict,
    sample,
    test_indices: list,
    trait: str
) -> dict:
    """
    특정 샘플에 대해 test items의 정답률 계산

    Returns:
        {'correct': int, 'total': int, 'accuracy': float, 'details': [...]}
    """
    correct = 0
    total = 0
    details = []

    for item_id in test_indices:
        if item_id not in items:
            continue

        statement = items[item_id]
        if statement.trait != trait:
            continue

        # 응답값 확인
        response = sample.responses.get(item_id, 0)
        if response == 0:
            continue

        # 기대 응답 계산
        is_accurate = (response == 2)
        is_positive_keyed = (statement.key == 1)

        # label=1이면 Yes가 정답, label=0이면 No가 정답
        expected_label = 1 if (is_accurate == is_positive_keyed) else 0
        expected_response = 'Yes' if expected_label == 1 else 'No'

        # 모델 응답 획득
        prompt = format_evaluation_prompt(statement.text)
        actual_response = get_model_response(model, tokenizer, prompt, model_name)

        is_correct = (actual_response == expected_response)
        if is_correct:
            correct += 1
        total += 1

        details.append({
            'item_id': item_id,
            'statement': statement.text[:50] + '...',
            'expected': expected_response,
            'actual': actual_response,
            'correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'details': details
    }


def evaluate_trait(
    model,
    tokenizer,
    model_name: str,
    items: dict,
    samples: list,
    trait_indices: dict,
    trait: str,
    interventions: dict = None,
    alpha: float = DEFAULT_ALPHA,
    steering: ActivationSteering = None
) -> dict:
    """
    특정 trait에 대해 모든 샘플의 정답률 평가

    Returns:
        {
            'trait': str,
            'base_accuracy': float,
            'steered_accuracy': float (if interventions provided),
            'sample_results': [...]
        }
    """
    test_indices = trait_indices[trait]['test']
    print(f"\nEvaluating {TRAIT_NAMES[trait]} ({trait.upper()})")
    print(f"  Test items: {len(test_indices)}")

    sample_results = []

    for sample in tqdm(samples, desc=f"Evaluating {trait}"):
        sample_desc = get_sample_description(sample)
        trait_level = sample.get_trait_level(trait)

        result = {
            'case': sample.case,
            'description': sample_desc,
            f'{trait}_level': trait_level,
        }

        # Base 모델 평가
        if steering:
            steering.reset()

        base_result = evaluate_sample_accuracy(
            model, tokenizer, model_name,
            items, sample, test_indices, trait
        )
        result['base_accuracy'] = base_result['accuracy']
        result['base_correct'] = base_result['correct']
        result['base_total'] = base_result['total']

        # Steered 모델 평가
        if interventions and steering:
            steering.apply_steering(interventions, alpha=alpha)

            steered_result = evaluate_sample_accuracy(
                model, tokenizer, model_name,
                items, sample, test_indices, trait
            )
            result['steered_accuracy'] = steered_result['accuracy']
            result['steered_correct'] = steered_result['correct']

            steering.reset()

        sample_results.append(result)

    # 전체 통계 계산
    base_total_correct = sum(r['base_correct'] for r in sample_results)
    base_total_items = sum(r['base_total'] for r in sample_results)
    base_overall_acc = base_total_correct / base_total_items if base_total_items > 0 else 0

    summary = {
        'trait': trait,
        'trait_name': TRAIT_NAMES[trait],
        'alpha': alpha if interventions else None,
        'base_overall_accuracy': base_overall_acc,
        'sample_results': sample_results
    }

    if interventions:
        steered_total_correct = sum(r.get('steered_correct', 0) for r in sample_results)
        steered_overall_acc = steered_total_correct / base_total_items if base_total_items > 0 else 0
        summary['steered_overall_accuracy'] = steered_overall_acc
        summary['improvement'] = steered_overall_acc - base_overall_acc

    return summary


def print_evaluation_results(results: dict):
    """평가 결과 출력"""
    trait = results['trait']
    print(f"\n{'='*70}")
    print(f"{results['trait_name']} ({trait.upper()}) Evaluation Results")
    print(f"{'='*70}")

    if results.get('alpha'):
        print(f"Alpha: {results['alpha']}")

    print(f"\nOverall Accuracy:")
    print(f"  Base model: {results['base_overall_accuracy']*100:.1f}%")

    if 'steered_overall_accuracy' in results:
        print(f"  Steered model: {results['steered_overall_accuracy']*100:.1f}%")
        print(f"  Improvement: {results['improvement']*100:+.1f}%")

    print(f"\nPer-Sample Results (by {trait.upper()} level):")
    print("-" * 70)
    print(f"{'Case':<25} {trait.upper():>8} {'Base':>10} {'Steered':>10} {'Diff':>8}")
    print("-" * 70)

    # High level samples
    for r in results['sample_results']:
        if r[f'{trait}_level'] == 'high':
            base_acc = f"{r['base_accuracy']*100:.1f}%"
            steered_acc = f"{r.get('steered_accuracy', 0)*100:.1f}%" if 'steered_accuracy' in r else 'N/A'
            diff = r.get('steered_accuracy', 0) - r['base_accuracy'] if 'steered_accuracy' in r else 0
            diff_str = f"{diff*100:+.1f}%" if 'steered_accuracy' in r else ''
            print(f"{r['case']:<25} {'HIGH':>8} {base_acc:>10} {steered_acc:>10} {diff_str:>8}")

    # Low level samples
    for r in results['sample_results']:
        if r[f'{trait}_level'] == 'low':
            base_acc = f"{r['base_accuracy']*100:.1f}%"
            steered_acc = f"{r.get('steered_accuracy', 0)*100:.1f}%" if 'steered_accuracy' in r else 'N/A'
            diff = r.get('steered_accuracy', 0) - r['base_accuracy'] if 'steered_accuracy' in r else 0
            diff_str = f"{diff*100:+.1f}%" if 'steered_accuracy' in r else ''
            print(f"{r['case']:<25} {'LOW':>8} {base_acc:>10} {steered_acc:>10} {diff_str:>8}")

    print("-" * 70)


def find_latest_intervention(output_dir: str, trait: str) -> str:
    """최신 intervention 파일 찾기"""
    pattern = os.path.join(output_dir, f"interventions_{trait}_*.pkl")
    files = sorted(glob(pattern))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Steered Model")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--trait", type=str, default=None, choices=TRAITS,
                        help="Trait to evaluate")
    parser.add_argument("--intervention_file", type=str, default=None,
                        help="Path to intervention pickle file")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Intervention strength")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--evaluate_all", action="store_true",
                        help="Evaluate all traits")
    parser.add_argument("--base_only", action="store_true",
                        help="Only evaluate base model (no steering)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for results")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("PAS Evaluation: Base vs Steered Model")
    print(f"{'#'*60}")

    # 데이터 로드
    print("\nLoading data...")
    items = load_item_keys()
    samples = load_samples()
    trait_indices = get_trait_indices()

    print(f"Loaded {len(items)} items, {len(samples)} samples")

    # 모델 로드
    model, tokenizer = load_model(args.model_name, args.use_4bit)

    # Steering 초기화
    steering = ActivationSteering(model, tokenizer, args.model_name)

    # 평가할 trait 결정
    traits_to_eval = TRAITS if args.evaluate_all else ([args.trait] if args.trait else TRAITS)

    all_results = {}

    for trait in traits_to_eval:
        # Intervention 파일 로드
        interventions = None
        if not args.base_only:
            intervention_file = args.intervention_file
            if not intervention_file:
                intervention_file = find_latest_intervention(args.output_dir, trait)

            if intervention_file and os.path.exists(intervention_file):
                print(f"\nLoading interventions from: {intervention_file}")
                with open(intervention_file, 'rb') as f:
                    interventions = pickle.load(f)
                print(f"  Loaded {sum(len(h) for h in interventions.values())} heads")
            else:
                print(f"\nNo intervention file found for {trait}, evaluating base model only")

        # 평가 실행
        results = evaluate_trait(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            items=items,
            samples=samples,
            trait_indices=trait_indices,
            trait=trait,
            interventions=interventions,
            alpha=args.alpha,
            steering=steering
        )

        all_results[trait] = results
        print_evaluation_results(results)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        args.output_dir,
        f"evaluation_results_{timestamp}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
