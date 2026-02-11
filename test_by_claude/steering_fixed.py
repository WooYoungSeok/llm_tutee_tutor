"""
steering.py
===========
Activation Steering (Intervention) 적용

논문 4.2-4.3절:
- 선택된 head의 output에 α * σ * direction 만큼 bias 추가
- Golden section search로 최적 α 탐색
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy
from einops import rearrange

from config import (
    ALPHA_SEARCH_MIN, ALPHA_SEARCH_MAX, ALPHA_SEARCH_TOL, NORM_EPS
)


@dataclass
class SteeringConfig:
    """Steering 설정"""
    alpha: float = 1.0
    interventions: Dict = None  # {layer_name: [(head, direction, sigma), ...]}


class ActivationSteering:
    """
    LLaMA 모델에 activation steering 적용

    o_proj의 bias를 조작하여 특정 head의 output 방향을 제어합니다.
    """

    def __init__(self, model, tokenizer, model_name: str):
        """
        Args:
            model: HuggingFace LLaMA 모델
            tokenizer: 토크나이저
            model_name: 모델 이름
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = next(model.parameters()).device

        # 모델 설정
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        # 원본 bias 캐시 (reset용)
        self._cache_original_biases()

    def _cache_original_biases(self):
        """원본 o_proj bias 저장"""
        self.bias_cache = []

        for layer_idx in range(self.num_layers):
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            if o_proj.bias is not None:
                self.bias_cache.append(deepcopy(o_proj.bias.data))
            else:
                # bias가 없으면 zero tensor (o_proj.weight와 같은 device에 생성)
                self.bias_cache.append(
                    torch.zeros(self.hidden_size, device=o_proj.weight.device, dtype=o_proj.weight.dtype)
                )

    def reset(self):
        """모든 bias를 원본으로 복원"""
        for layer_idx in range(self.num_layers):
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            cached_bias = self.bias_cache[layer_idx].to(o_proj.weight.device)  # 올바른 device로 이동
            
            if o_proj.bias is None:
                o_proj.bias = nn.Parameter(cached_bias.clone())
            else:
                o_proj.bias.data = cached_bias.clone()

    def apply_steering(
        self,
        interventions: Dict[str, List[Tuple[int, np.ndarray, float]]],
        alpha: float = 1.0
    ):
        """
        Intervention을 모델에 적용

        논문 수식:
        x_{l+1} = x_l + Σ_h [Att_h(x_l) + α * σ_h * direction_h]

        실제 구현: o_proj의 bias에 α * σ * W_o @ displacement 추가

        Args:
            interventions: {layer_name: [(head_idx, direction, sigma), ...]}
            alpha: 조정 강도
        """
        # 먼저 reset
        self.reset()

        for layer_name, head_list in interventions.items():
            # layer_name: "model.layers.{layer_idx}.self_attn.head_out"
            layer_idx = int(layer_name.split('.')[2])

            # Head별 displacement 계산
            displacement = np.zeros((self.num_heads, self.head_dim))

            for head_idx, direction, sigma in head_list:
                # direction은 이미 정규화됨, sigma는 std
                displacement[head_idx] = alpha * sigma * direction

            # (num_heads, head_dim) -> (hidden_size,)
            displacement_flat = displacement.reshape(-1)

            # Tensor로 변환 - o_proj가 있는 디바이스 사용 (multi-GPU 대응)
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            displacement_tensor = torch.tensor(
                displacement_flat,
                device=o_proj.weight.device,  # self.device 대신 o_proj의 device 사용
                dtype=o_proj.weight.dtype
            )

            # o_proj를 통과시켜 실제 bias 계산
            # bias = W_o @ displacement
            with torch.no_grad():
                bias_delta = F.linear(displacement_tensor, o_proj.weight)
                
                # Multi-GPU 대응: bias_delta를 o_proj.bias와 같은 device로 이동
                if o_proj.bias is not None:
                    bias_delta = bias_delta.to(o_proj.bias.device)

            # 기존 bias에 추가
            if o_proj.bias is None:
                o_proj.bias = nn.Parameter(bias_delta)
            else:
                o_proj.bias.data = o_proj.bias.data + bias_delta

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Steering이 적용된 상태로 텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            system_prompt: 시스템 프롬프트
            temperature: 샘플링 temperature
            do_sample: 샘플링 사용 여부

        Returns:
            생성된 텍스트
        """
        # 토큰화 - 단순 텍스트 형식 사용
        formatted_prompt = f"{system_prompt}\n\n{prompt}"
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 디코딩 (입력 부분 제외)
        generated_tokens = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text


def golden_section_search(
    objective_fn: Callable[[float], float],
    a: float = ALPHA_SEARCH_MIN,
    b: float = ALPHA_SEARCH_MAX,
    tol: float = ALPHA_SEARCH_TOL
) -> Tuple[float, float]:
    """
    Golden section search로 최적 α 탐색

    논문 4.3절:
    Optimal α = argmin_{α∈[0,10]} f(α)

    Args:
        objective_fn: α -> score (최소화 대상)
        a: 탐색 구간 하한
        b: 탐색 구간 상한
        tol: 수렴 허용 오차

    Returns:
        (optimal_alpha, objective_value)
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    # 초기 점
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = objective_fn(x1)
    f2 = objective_fn(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = objective_fn(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = objective_fn(x2)

    optimal_alpha = (a + b) / 2
    optimal_value = objective_fn(optimal_alpha)

    return optimal_alpha, optimal_value


def search_optimal_alpha(
    steering: ActivationSteering,
    interventions: Dict,
    eval_fn: Callable[[ActivationSteering], float],
    alpha_min: float = ALPHA_SEARCH_MIN,
    alpha_max: float = ALPHA_SEARCH_MAX,
    tol: float = ALPHA_SEARCH_TOL
) -> Tuple[float, float]:
    """
    최적 α 탐색

    Args:
        steering: ActivationSteering 인스턴스
        interventions: intervention 딕셔너리
        eval_fn: steering -> score (낮을수록 좋음)
        alpha_min, alpha_max: 탐색 구간
        tol: 수렴 허용 오차

    Returns:
        (optimal_alpha, score)
    """
    def objective(alpha):
        steering.apply_steering(interventions, alpha=alpha)
        score = eval_fn(steering)
        steering.reset()
        return score

    return golden_section_search(objective, alpha_min, alpha_max, tol)


def grid_search_alpha(
    steering: ActivationSteering,
    interventions: Dict,
    eval_fn: Callable[[ActivationSteering], float],
    alphas: List[float] = None
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Grid search로 α 탐색 (디버깅용)

    Args:
        steering: ActivationSteering 인스턴스
        interventions: intervention 딕셔너리
        eval_fn: steering -> score
        alphas: 테스트할 α 값들

    Returns:
        (best_alpha, best_score, all_results)
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    results = []
    for alpha in alphas:
        steering.apply_steering(interventions, alpha=alpha)
        score = eval_fn(steering)
        steering.reset()
        results.append((alpha, score))
        print(f"  α={alpha:.1f}: score={score:.4f}")

    best_alpha, best_score = min(results, key=lambda x: x[1])

    return best_alpha, best_score, results


if __name__ == "__main__":
    print("ActivationSteering module loaded.")
    print("Usage:")
    print("  steering = ActivationSteering(model, tokenizer, model_name)")
    print("  steering.apply_steering(interventions, alpha=1.0)")
    print("  output = steering.generate(prompt)")
