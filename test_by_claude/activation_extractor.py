"""
activation_extractor.py
=======================
Head-wise activation 추출

각 layer의 attention output (o_proj 이전)에서 head별 activation을 추출합니다.
Forward hook을 사용하여 baukit 의존성 없이 구현합니다.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from config import NORM_EPS


@dataclass
class HeadActivation:
    """단일 head의 activation"""
    layer: int
    head: int
    activation: np.ndarray  # shape: (head_dim,) - 마지막 토큰의 activation


class ActivationExtractor:
    """
    LLaMA 모델에서 head-wise activation 추출

    각 attention layer의 output (reshape 후, o_proj 전)에서
    head별로 분리된 activation을 추출합니다.
    """

    def __init__(self, model, tokenizer, model_name: str):
        """
        Args:
            model: HuggingFace LLaMA 모델
            tokenizer: 토크나이저
            model_name: 모델 이름 (프롬프트 포맷용)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = next(model.parameters()).device

        # 모델 설정 추출
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        # Activation 저장용
        self._activations = {}
        self._hooks = []

    def _get_hook(self, layer_idx: int):
        """특정 layer에 대한 forward hook 생성"""
        def hook(module, input, output):
            # output shape: (batch, seq_len, hidden_size)
            # 마지막 토큰의 activation만 저장
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            # float32로 변환하여 저장 (fp16 overflow 방지)
            last_token_act = out[:, -1, :].detach().cpu().float()
            self._activations[layer_idx] = last_token_act

        return hook

    def _register_hooks(self):
        """모든 attention layer에 hook 등록"""
        self._clear_hooks()

        for layer_idx in range(self.num_layers):
            # LLaMA의 attention output 위치: model.layers[i].self_attn.o_proj의 입력
            # 하지만 o_proj 입력을 직접 얻기 어려우므로,
            # self_attn 전체의 output을 가져온 후 처리

            # 방법: self_attn 모듈의 output hook
            attn_module = self.model.model.layers[layer_idx].self_attn
            hook = attn_module.register_forward_hook(self._get_attn_hook(layer_idx))
            self._hooks.append(hook)

    def _get_attn_hook(self, layer_idx: int):
        """Attention 모듈의 출력에서 o_proj 이전 값 추출"""
        def hook(module, input, output):
            # LlamaAttention.forward returns: (attn_output, attn_weights, past_key_value)
            # attn_output은 이미 o_proj를 통과한 상태

            # 대신 우리는 o_proj 이전의 값이 필요함
            # 이를 위해 입력 hidden_states에서 q,k,v를 직접 계산해야 함
            # 하지만 이는 복잡하므로, 더 간단한 방법 사용:

            # o_proj의 입력을 얻기 위해 o_proj에 별도 hook 설치
            pass

        return hook

    def _register_o_proj_hooks(self):
        """o_proj layer에 입력 hook 등록 (o_proj 이전 activation 추출)"""
        self._clear_hooks()

        for layer_idx in range(self.num_layers):
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj

            def make_hook(idx):
                def hook(module, input, output):
                    # input[0] shape: (batch, seq_len, hidden_size)
                    # 이것이 head들이 concat된 상태의 attention output
                    inp = input[0]
                    last_token = inp[:, -1, :].detach().cpu().float()
                    self._activations[idx] = last_token
                return hook

            hook_handle = o_proj.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(hook_handle)

    def _clear_hooks(self):
        """등록된 모든 hook 제거"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    def extract_activations(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant."
    ) -> np.ndarray:
        """
        단일 프롬프트에 대해 모든 layer의 head-wise activation 추출

        Args:
            prompt: 입력 프롬프트
            system_prompt: 시스템 프롬프트

        Returns:
            shape (num_layers, hidden_size) - 각 layer의 마지막 토큰 activation
        """
        # Hook 등록
        self._register_o_proj_hooks()

        try:
            # 토큰화 - chat template 대신 단순 텍스트 사용
            # Chat template은 EOT 토큰을 추가하여 Yes/No가 마지막 토큰이 아니게 만듦
            # 단순 텍스트 형식을 사용하여 Yes/No가 실제 마지막 토큰이 되도록 함
            formatted_prompt = f"{system_prompt}\n\n{prompt}"
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids
            
            input_ids = input_ids.to(self.device)

            # Forward pass
            with torch.no_grad():
                _ = self.model(input_ids)

            # Activation 수집
            activations = np.zeros((self.num_layers, self.hidden_size), dtype=np.float32)
            for layer_idx in range(self.num_layers):
                if layer_idx in self._activations:
                    act = self._activations[layer_idx].numpy()
                    # batch dimension 제거 (batch=1 가정)
                    if act.ndim == 2:
                        act = act[0]
                    activations[layer_idx] = act

            return activations

        finally:
            self._clear_hooks()

    def extract_head_activations(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant."
    ) -> np.ndarray:
        """
        Head별로 분리된 activation 추출

        Args:
            prompt: 입력 프롬프트
            system_prompt: 시스템 프롬프트

        Returns:
            shape (num_layers, num_heads, head_dim)
        """
        # 전체 activation 추출
        activations = self.extract_activations(prompt, system_prompt)

        # Head별로 reshape
        # (num_layers, hidden_size) -> (num_layers, num_heads, head_dim)
        head_activations = activations.reshape(
            self.num_layers, self.num_heads, self.head_dim
        )

        return head_activations

    def extract_batch(
        self,
        prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        여러 프롬프트에 대해 activation 추출

        Args:
            prompts: 프롬프트 리스트
            system_prompt: 시스템 프롬프트
            show_progress: tqdm 표시 여부

        Returns:
            shape (num_prompts, num_layers, num_heads, head_dim)
        """
        all_activations = []

        iterator = tqdm(prompts, desc="Extracting activations") if show_progress else prompts

        for prompt in iterator:
            act = self.extract_head_activations(prompt, system_prompt)
            all_activations.append(act)

        return np.array(all_activations)


def extract_paired_activations(
    extractor: ActivationExtractor,
    activation_samples,  # List[ActivationSample]
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ActivationSample 리스트에서 agree/disagree 쌍의 activation 추출

    Returns:
        agree_activations: shape (n_samples, num_layers, num_heads, head_dim)
        disagree_activations: shape (n_samples, num_layers, num_heads, head_dim)
        labels: shape (n_samples,) - 1 if agree is positive direction, 0 otherwise
    """
    agree_acts = []
    disagree_acts = []
    labels = []

    iterator = tqdm(activation_samples, desc="Extracting paired activations") if show_progress else activation_samples

    for sample in iterator:
        # Agree 프롬프트 activation
        agree_act = extractor.extract_head_activations(sample.prompt_agree)
        agree_acts.append(agree_act)

        # Disagree 프롬프트 activation
        disagree_act = extractor.extract_head_activations(sample.prompt_disagree)
        disagree_acts.append(disagree_act)

        labels.append(sample.label)

    return (
        np.array(agree_acts),
        np.array(disagree_acts),
        np.array(labels)
    )


def validate_activations(activations: np.ndarray) -> Dict[str, float]:
    """
    Activation 배열의 유효성 검사

    Returns:
        통계 딕셔너리
    """
    total = activations.size
    finite_count = np.isfinite(activations).sum()
    nan_count = np.isnan(activations).sum()
    inf_count = np.isinf(activations).sum()

    stats = {
        'total': total,
        'finite_ratio': finite_count / total,
        'nan_ratio': nan_count / total,
        'inf_ratio': inf_count / total,
        'mean': np.nanmean(activations),
        'std': np.nanstd(activations),
        'min': np.nanmin(activations),
        'max': np.nanmax(activations)
    }

    return stats


if __name__ == "__main__":
    print("ActivationExtractor module loaded.")
    print(f"This module provides:")
    print("  - ActivationExtractor: Head-wise activation extraction from LLaMA models")
    print("  - extract_paired_activations: Extract agree/disagree activation pairs")
    print("  - validate_activations: Check activation validity")
