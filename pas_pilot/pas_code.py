"""
pas_code.py
===========
pas.py의 NaN direction 버그 수정 버전.

수정 사항:
1. get_llama_activations_bau(): float16 → float32 (overflow → NaN 방지)
2. get_com_directions(): 디버깅 print 수정, NaN 전파 방지
3. get_interventions_dict(): direction 정규화 시 epsilon 추가 (0/0 → NaN 방지)
4. get_interventions_dict(): NaN direction 필터링
5. preprocess_activate_dataset(): activation 추출 후 NaN 검증
6. bias_cache 초기화: o_proj.bias가 None인 경우 안전 처리
"""

import json
import random
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from einops import rearrange
import pickle
from functools import partial
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baukit import Trace, TraceDict

from copy import deepcopy

# direction 정규화 시 zero-division 방지를 위한 epsilon
NORM_EPS = 1e-8


def get_model(model_name='meta-llama/Llama-2-7b-chat-hf', use_bit_4=False, adapter=None):
    """
    Loads and sets up the Meta-LLaMA model for inference and activation handling.
    """

    class PASLM:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            if self.config.architectures[0] == 'MistralForCausalLM':
                from modeling_mistral import MistralForCausalLM as ModelForCausalLM
            elif self.config.architectures[0] == 'LlamaForCausalLM':
                from modeling_llama_pas import LlamaForCausalLM as ModelForCausalLM
            else:
                print('PAS not implemented yet for {}.'.format(self.config.architectures[0]))

            if adapter:
                if 'ppo' in adapter:
                    load_model_name = adapter
                    load_adapter = None
                else:
                    load_adapter = adapter
                    load_model_name = model_name
            else:
                load_model_name = model_name
                load_adapter = None
            self.model_file = load_model_name

            if use_bit_4:
                self._load_large_model(load_model_name, ModelForCausalLM)
            else:
                self._load_standard_model(load_model_name, load_adapter, ModelForCausalLM)

            # FIX 6: bias_cache 안전 초기화
            # modeling_llama_pas.py에서 o_proj.bias를 zero tensor로 초기화하므로
            # None이 아닌 zero tensor가 캐시됨
            self.bias_cache = []
            for i, layer in enumerate(self.model.model.layers):
                bias = self.model.model.layers[i].self_attn.o_proj.bias
                if bias is not None:
                    self.bias_cache.append(deepcopy(bias))
                else:
                    # bias가 None인 경우 zero tensor 생성
                    hidden_size = self.model.model.config.hidden_size
                    device = self.model.model.layers[i].self_attn.o_proj.weight.device
                    dtype = self.model.model.layers[i].self_attn.o_proj.weight.dtype
                    self.bias_cache.append(
                        nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
                    )

        def _load_large_model(self, model_name, ModelForCausalLM):
            model = ModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            self.weight_cache = [deepcopy(layer.self_attn.o_proj.weight).cuda() for layer in model.model.layers]
            model = None
            torch.cuda.empty_cache()
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            self.model = ModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, quantization_config=quantization_config, torch_dtype=torch.bfloat16)

        def _load_standard_model(self, model_name, adapter, ModelForCausalLM):
            if 'ppo' in model_name:
                self.model = ModelForCausalLM.from_pretrained(model_name, device_map='cuda')
            else:
                self.model = ModelForCausalLM.from_pretrained(model_name).half().cuda()
            if adapter:
                self.model.load_adapter(adapter)
            self.model.eval()
            self.device = self.model.device

        def generate(model, text, max_length=512, max_new_tokens=None):
            tokenizer = model.tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            stop_id = tokenizer.sep_token_id
            pad_id = tokenizer.pad_token_id

            device = model.device
            input_ids = [t for t in text]
            min_prompt_len = min(len(t) for t in input_ids)
            max_prompt_len = max(len(t) for t in input_ids)

            if max_new_tokens:
                max_length = max_prompt_len + max_new_tokens
            tokens = torch.full((len(input_ids), max_length), pad_id, dtype=torch.long).to(device)
            for k, t in enumerate(input_ids):
                tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            prev_pos = 0
            cur_pos = min_prompt_len - 1
            input_text_mask = tokens != pad_id
            eos_reached = torch.tensor([False] * len(input_ids), device=device)
            past_key_values = None

            with torch.no_grad():
                for cur_pos_add in range(max_length):
                    cur_pos += 1
                    if prev_pos != 0:
                        prev_pos = cur_pos - 1
                    if tokens.shape[1] == cur_pos:
                        break
                    torch.cuda.empty_cache()

                    logits = model.model(tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=past_key_values)
                    next_token = torch.topk(logits['logits'][:, -1], 1, dim=-1)[1][:, -1]
                    next_token = next_token.reshape(-1)
                    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                    tokens[:, cur_pos] = next_token
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == model.tokenizer.eos_token_id)

                    if all(eos_reached):
                        break
                    prev_pos = cur_pos
                    past_key_values = logits["past_key_values"]
            return tokens

        def __call__(self, input_ids):
            with torch.no_grad():
                logits = self.model(input_ids)
                return logits

        def get_last_activations(self, layer):
            return self.model.model.layers[layer].activations

        def reset_all(self):
            for i, layer in enumerate(self.model.model.layers):
                self.model.model.layers[i].self_attn.o_proj.bias = deepcopy(self.bias_cache[i])

        def get_activations(self, all_head_wise_activations, labels, num_to_intervene=48):
            def get_top_heads(separated_activations, separated_labels, num_layers, num_heads, num_to_intervene):
                probes, all_head_accs_np = train_probes(separated_activations,
                                                        separated_labels, num_layers=num_layers, num_heads=num_heads)
                all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads, 2)
                all_head_accs_np = all_head_accs_np.mean(2)
                top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
                top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

                return top_heads, probes

            def train_probes(separated_head_wise_activations, separated_labels,
                             num_layers, num_heads):
                all_head_accs = []
                probes = []

                train_idxs = np.arange(len(separated_labels))

                train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * (1 - 0.4)),
                                                  replace=False)
                val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

                all_X_train = np.array([separated_head_wise_activations[i] for i in train_set_idxs])
                all_X_val = np.array([separated_head_wise_activations[i] for i in val_set_idxs])
                y_train = np.array([separated_labels[i] for i in train_set_idxs])
                y_val = np.array([separated_labels[i] for i in val_set_idxs])

                for layer in tqdm(range(num_layers), desc="Training probes"):
                    for head in range(num_heads):
                        X_train = all_X_train[:, layer, head, :]
                        X_val = all_X_val[:, layer, head, :]

                        # FIX: NaN 입력 검증
                        if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
                            print(f"  WARNING: NaN/inf in activations at layer {layer}, head {head}")
                            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

                        clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
                        y_pred = clf.predict(X_train)
                        y_val_pred = clf.predict(X_val)
                        all_head_accs.append([accuracy_score(y_val, y_val_pred), accuracy_score(y_train, y_pred)])
                        probes.append(clf)

                all_head_accs_np = np.array(all_head_accs)

                return probes, all_head_accs_np

            def flattened_idx_to_layer_head(flattened_idx, num_heads):
                return flattened_idx // num_heads, flattened_idx % num_heads

            def layer_head_to_flattened_idx(layer, head, num_heads):
                return layer * num_heads + head

            def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, com_directions=None):
                interventions = {}
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.self_attn.head_out"] = []
                for layer, head in top_heads:
                    if com_directions is not None:
                        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
                    else:
                        direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_

                    # FIX 3: epsilon 추가로 zero-division → NaN 방지
                    norm = np.linalg.norm(direction)
                    if norm < NORM_EPS:
                        print(f"  WARNING: Near-zero norm at layer {layer}, head {head} (norm={norm:.2e}), skipping")
                        continue
                    direction = direction / norm

                    # FIX 4: NaN direction 필터링
                    if not np.isfinite(direction).all():
                        print(f"  WARNING: NaN/inf direction at layer {layer}, head {head}, skipping")
                        continue

                    activations = tuning_activations[:, layer, head, :]  # batch x head_dim
                    proj_vals = activations @ direction.T
                    proj_val_std = np.std(proj_vals)

                    if not np.isfinite(proj_val_std):
                        print(f"  WARNING: NaN/inf proj_val_std at layer {layer}, head {head}, skipping")
                        continue

                    interventions[f"model.layers.{layer}.self_attn.head_out"].append(
                        (head, direction.squeeze(), proj_val_std))

                for layer, head in top_heads:
                    key = f"model.layers.{layer}.self_attn.head_out"
                    if key in interventions:
                        interventions[key] = sorted(interventions[key], key=lambda x: x[0])

                # 빈 레이어 제거
                interventions = {k: v for k, v in interventions.items() if len(v) > 0}

                return interventions

            def get_com_directions(num_layers, num_heads, usable_head_wise_activations,
                                   usable_labels):
                com_directions = []
                usable_labels = np.array(usable_labels)

                # label 분포 확인
                n_pos = np.sum(usable_labels == 1)
                n_neg = np.sum(usable_labels == 0)
                print(f"  COM directions: {n_pos} positive, {n_neg} negative samples")

                if n_pos == 0 or n_neg == 0:
                    print("  WARNING: One class has 0 samples! Directions will be zero.")

                for layer in range(num_layers):
                    for head in range(num_heads):
                        head_wise_activations = usable_head_wise_activations[:, layer, head, :]

                        # FIX: NaN 검증 후 안전한 mean 계산
                        pos_mask = usable_labels == 1
                        neg_mask = usable_labels == 0

                        if pos_mask.sum() > 0:
                            true_mass_mean = np.nanmean(head_wise_activations[pos_mask], axis=0)
                        else:
                            true_mass_mean = np.zeros(head_wise_activations.shape[1])

                        if neg_mask.sum() > 0:
                            false_mass_mean = np.nanmean(head_wise_activations[neg_mask], axis=0)
                        else:
                            false_mass_mean = np.zeros(head_wise_activations.shape[1])

                        diff = true_mass_mean - false_mass_mean
                        # NaN을 0으로 대체
                        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                        com_directions.append(diff)

                com_directions = np.array(com_directions)

                # FIX 4: 디버깅 print 수정
                norms = np.linalg.norm(com_directions, axis=1)
                n_zero = np.sum(norms < NORM_EPS)
                n_nan = np.sum(~np.isfinite(norms))
                print(f"  Direction norms: {n_zero}/{len(norms)} near-zero, {n_nan}/{len(norms)} NaN/inf")
                if len(norms) > 0:
                    valid_norms = norms[np.isfinite(norms) & (norms > NORM_EPS)]
                    if len(valid_norms) > 0:
                        print(f"  Valid norm range: [{valid_norms.min():.4f}, {valid_norms.max():.4f}]")

                return com_directions

            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            # NaN 검증 후 rearrange
            head_wise_activations = np.array(deepcopy(all_head_wise_activations))
            nan_ratio = 1 - np.isfinite(head_wise_activations).mean()
            if nan_ratio > 0:
                print(f"  WARNING: {nan_ratio*100:.1f}% of activation values are NaN/inf before rearrange")
                head_wise_activations = np.nan_to_num(head_wise_activations, nan=0.0, posinf=0.0, neginf=0.0)

            head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)
            tuning_activations = deepcopy(head_wise_activations)

            top_heads, probes = get_top_heads(head_wise_activations, labels, num_layers, num_heads, num_to_intervene)

            com_directions = get_com_directions(num_layers, num_heads, head_wise_activations, labels)

            interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, com_directions)

            return interventions

        def preprocess_activate_dataset(self, dataset, system_prompt="You are a helpful, honest and concise assistant."):
            self.system_prompt = system_prompt

            def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
                if 'llama-3' in self.model_file.lower():
                    if model_output:
                        con = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": model_output}
                        ]
                        return torch.tensor(tokenizer.apply_chat_template(con)[:-5]).unsqueeze(0)
                    else:
                        con = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                        ]
                        return torch.tensor(tokenizer.apply_chat_template(con)).unsqueeze(0)
                else:
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
                    dialog_tokens = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                    return torch.tensor(tokenizer(dialog_tokens).input_ids).unsqueeze(0)

            def data_preprocess(dataset):
                all_prompts = []
                for i in range(len(dataset)):
                    question = dataset[i]['question']

                    pos_answer = dataset[i]['answer_matching_behavior']
                    pos_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, question, pos_answer)
                    all_prompts.append(pos_tokens)

                    neg_answer = dataset[i]['answer_not_matching_behavior']
                    neg_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, question, neg_answer)
                    all_prompts.append(neg_tokens)

                return all_prompts

            def get_llama_activations_bau(model, prompt):
                HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
                MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

                with torch.no_grad():
                    prompt = prompt.to(model.device)
                    with TraceDict(model, HEADS + MLPS) as ret:
                        output = model(prompt, output_hidden_states=True)
                    hidden_states = output.hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze().detach().cpu()

                    # FIX 1: float32 유지 (float16 overflow → NaN 방지)
                    head_wise_hidden_states = [
                        ret[head].output.squeeze().detach().cpu().float()
                        for head in HEADS
                    ]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()

                    mlp_wise_hidden_states = [
                        ret[mlp].output.squeeze().detach().cpu().float()
                        for mlp in MLPS
                    ]
                    mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

                return hidden_states.float().numpy(), head_wise_hidden_states, mlp_wise_hidden_states

            prompts = data_preprocess(dataset)

            all_layer_wise_activations = []
            all_head_wise_activations = []

            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
                layer_wise_activation, head_wise_activation, _ = get_llama_activations_bau(self.model, prompt)
                all_layer_wise_activations.append(layer_wise_activation[:, -1, :])

                # FIX 5: activation 추출 직후 NaN 검증
                act = head_wise_activation[:, -1, :]
                finite_ratio = np.isfinite(act).mean()
                if finite_ratio < 1.0:
                    print(f"  Prompt {prompt_idx}: {finite_ratio*100:.1f}% finite values (replacing NaN/inf with 0)")
                    act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

                all_head_wise_activations.append(act)

            return all_head_wise_activations

        def set_activate(self, interventions, alpha):
            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            for head_out_name, list_int_vec in interventions.items():
                layer_no = int(head_out_name.split('.')[2])
                displacement = np.zeros((num_heads, int(self.model.model.config.hidden_size / num_heads)))
                for head_no, head_vec, std in list_int_vec:
                    displacement[head_no] = alpha * std * head_vec
                device = self.model.model.layers[layer_no].self_attn.o_proj.weight.device.index
                displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                if '70B' in self.model_file:
                    bias_tobe = F.linear(displacement.to(torch.bfloat16), self.weight_cache[layer_no]).to(device)
                else:
                    bias_tobe = F.linear(displacement.to(torch.float16), self.model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
                self.model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    model = PASLM()
    model.reset_all()
    return model, model.tokenizer
