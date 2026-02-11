"""
main_debug.py
=============
디버깅용 메인 파일 - 데이터셋과 activation 추출 과정을 상세히 출력
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
from data_utils_fixed import (
    load_item_keys, load_samples, load_train_test_split,
    create_activation_samples, get_trait_samples
)
from activation_extractor import (
    ActivationExtractor, extract_paired_activations, validate_activations
)
from probe_trainer import train_and_select_heads, ProbeTrainer
from steering_fixed import ActivationSteering, grid_search_alpha


def debug_tokenization(tokenizer, prompts, model_name):
    """토큰화 결과 디버깅"""
    print(f"\n{'='*60}")
    print("DEBUG: Tokenization")
    print(f"{'='*60}")
    
    for i, prompt in enumerate(prompts[:2]):  # 처음 2개만
        print(f"\n[Prompt {i}]")
        print(f"Text: {prompt}")
        
        # 단순 텍스트 형식
        formatted = f"You are a helpful assistant.\n\n{prompt}"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"Total tokens: {len(tokens)}")
        print(f"Last 5 tokens: {tokens[-5:]}")
        print(f"Last token: '{tokens[-1]}'")
        print(f"Last token ID: {input_ids[-1].item()}")


def debug_activation_samples(activation_samples, num_to_show=5):
    """ActivationSample 데이터 디버깅"""
    print(f"\n{'='*60}")
    print("DEBUG: Activation Samples")
    print(f"{'='*60}")
    
    print(f"\nTotal samples: {len(activation_samples)}")
    
    # Label 분포
    labels = [s.label for s in activation_samples]
    print(f"Label distribution: {sum(labels)} positive (label=1), {len(labels)-sum(labels)} negative (label=0)")
    
    # Trait별 분포
    print("\nBy trait:")
    for trait in TRAITS:
        trait_samples = [s for s in activation_samples if s.statement.trait == trait]
        trait_labels = [s.label for s in trait_samples]
        print(f"  {trait}: {len(trait_samples)} samples, {sum(trait_labels)} label=1, {len(trait_labels)-sum(trait_labels)} label=0")
    
    # Key별 분포
    print("\nBy key:")
    pos_key = [s for s in activation_samples if s.statement.key == 1]
    neg_key = [s for s in activation_samples if s.statement.key == -1]
    print(f"  Positive keyed (+1): {len(pos_key)}")
    print(f"  Negative keyed (-1): {len(neg_key)}")
    
    # 샘플 출력
    print(f"\nFirst {num_to_show} samples:")
    for i, sample in enumerate(activation_samples[:num_to_show]):
        print(f"\n[Sample {i}]")
        print(f"  Statement: {sample.statement.text[:60]}...")
        print(f"  Trait: {sample.statement.trait}, Key: {sample.statement.key}")
        print(f"  Label: {sample.label} ({'Agree is positive' if sample.label == 1 else 'Disagree is positive'})")
        print(f"  Prompt (agree): ...{sample.prompt_agree[-50:]}")
        print(f"  Prompt (disagree): ...{sample.prompt_disagree[-50:]}")


def debug_activations(agree_acts, disagree_acts, labels):
    """Activation 통계 디버깅"""
    print(f"\n{'='*60}")
    print("DEBUG: Activation Statistics")
    print(f"{'='*60}")
    
    # 전체 통계
    print("\nOverall statistics:")
    print(f"  Agree: mean={agree_acts.mean():.6f}, std={agree_acts.std():.6f}")
    print(f"  Disagree: mean={disagree_acts.mean():.6f}, std={disagree_acts.std():.6f}")
    print(f"  Difference (agree-disagree): mean={(agree_acts-disagree_acts).mean():.6f}, std={(agree_acts-disagree_acts).std():.6f}")
    
    # Layer별 통계
    print("\nPer-layer statistics (first 5 layers):")
    for layer in range(min(5, agree_acts.shape[1])):
        agree_layer = agree_acts[:, layer, :, :].flatten()
        disagree_layer = disagree_acts[:, layer, :, :].flatten()
        diff = agree_layer - disagree_layer
        print(f"  Layer {layer}: agree_mean={agree_layer.mean():.6f}, disagree_mean={disagree_layer.mean():.6f}, diff_mean={diff.mean():.6f}")
    
    # Label별 차이 확인
    print("\nBy label:")
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    if len(pos_idx) > 0:
        diff_pos = (agree_acts[pos_idx] - disagree_acts[pos_idx]).flatten()
        print(f"  Label=1 (agree should be positive): diff_mean={diff_pos.mean():.6f}, diff_std={diff_pos.std():.6f}")
    
    if len(neg_idx) > 0:
        diff_neg = (agree_acts[neg_idx] - disagree_acts[neg_idx]).flatten()
        print(f"  Label=0 (disagree should be positive): diff_mean={diff_neg.mean():.6f}, diff_std={diff_neg.std():.6f}")
    
    # 샘플별 차이 확인
    print("\nPer-sample differences (first 10 samples):")
    for i in range(min(10, len(labels))):
        diff = (agree_acts[i] - disagree_acts[i]).flatten()
        print(f"  Sample {i} (label={labels[i]}): diff_mean={diff.mean():.6f}, diff_std={diff.std():.6f}")


def debug_probe_data(X, y):
    """Probe 학습 데이터 디버깅"""
    print(f"\n{'='*60}")
    print("DEBUG: Probe Training Data")
    print(f"{'='*60}")
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y distribution: {(y==1).sum()} positive, {(y==0).sum()} negative")
    
    # 각 head의 데이터 분포 확인 (처음 3개 head만)
    print("\nPer-head statistics (first 3 heads):")
    for layer in range(min(2, X.shape[1])):
        for head in range(min(3, X.shape[2])):
            X_head = X[:, layer, head, :]
            print(f"  Layer {layer}, Head {head}: mean={X_head.mean():.6f}, std={X_head.std():.6f}")
    
    # y=1과 y=0의 activation 차이
    print("\nActivation difference between y=1 and y=0:")
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    
    if len(X_pos) > 0 and len(X_neg) > 0:
        diff = X_pos.mean(axis=0) - X_neg.mean(axis=0)
        print(f"  Overall diff: mean={diff.mean():.6f}, std={diff.std():.6f}")
        print(f"  Max diff: {diff.max():.6f}, Min diff: {diff.min():.6f}")


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
            dtype=torch.float16,
            device_map='auto'
        )

    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="PAS Debug Mode")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("PAS Debug Mode - Detailed Data Inspection")
    print(f"{'#'*60}")

    # Step 1: 데이터 로드
    print(f"\n{'='*60}")
    print("Step 1: Loading data")
    print(f"{'='*60}")
    
    items = load_item_keys()
    samples = load_samples()
    train_idx, test_idx = load_train_test_split()
    
    print(f"  Loaded {len(items)} items")
    print(f"  Loaded {len(samples)} samples")
    print(f"  Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    # Step 2: Activation 샘플 생성
    print(f"\n{'='*60}")
    print("Step 2: Creating activation samples")
    print(f"{'='*60}")
    
    activation_samples = []
    for sample_idx, sample in enumerate(samples):
        act_samples = create_activation_samples(
            items=items,
            sample=sample,
            item_indices=train_idx,
            traits=TRAITS
        )
        activation_samples.extend(act_samples)
        print(f"  Sample {sample_idx}: {len(act_samples)} activation samples")
    
    # 디버깅: Activation 샘플 상세 출력
    debug_activation_samples(activation_samples, num_to_show=5)

    # 모델 로드
    model, tokenizer = load_model(args.model_name, use_4bit=args.use_4bit)
    
    # 디버깅: 토큰화 확인
    test_prompts = [
        activation_samples[0].prompt_agree,
        activation_samples[0].prompt_disagree,
    ]
    debug_tokenization(tokenizer, test_prompts, args.model_name)

    # Step 3: Activation 추출
    print(f"\n{'='*60}")
    print("Step 3: Extracting activations (first 20 samples only for debug)")
    print(f"{'='*60}")
    
    extractor = ActivationExtractor(model, tokenizer, args.model_name)
    
    # 작은 샘플로 테스트
    test_samples = activation_samples[:20]
    agree_acts, disagree_acts, labels = extract_paired_activations(
        extractor, test_samples, show_progress=True
    )
    
    # 디버깅: Activation 통계
    debug_activations(agree_acts, disagree_acts, labels)

    # Step 4: Probe 데이터 준비
    print(f"\n{'='*60}")
    print("Step 4: Preparing probe training data")
    print(f"{'='*60}")
    
    trainer = ProbeTrainer(
        num_layers=extractor.num_layers,
        num_heads=extractor.num_heads,
        head_dim=extractor.head_dim,
        train_ratio=DEFAULT_TRAIN_RATIO
    )
    
    X, y = trainer.prepare_training_data(agree_acts, disagree_acts, labels)
    
    # 디버깅: Probe 데이터
    debug_probe_data(X, y)

    print(f"\n{'#'*60}")
    print("Debug completed! Check the output above.")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
