"""
main_by_claude_code.py
======================
main_by_claude.py의 NaN direction 버그 수정 버전.

수정 사항:
1. process_pas(): personal_data에 'question' 필드 추가 (KeyError 수정)
2. process_pas(): activation-label 정렬 보장 (data[0] 기준 인덱싱 고정)
3. process_pas(): activation 추출 후 NaN/inf 검증 추가
4. print_and_save_results(): se/im/as 3개 trait 대응 (Big Five 5개 하드코딩 제거)
5. getItems 경로: 실행 위치 무관하게 동작하도록 수정
"""

import json
from huggingface_hub import login
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint
from pas_code import get_model
from copy import deepcopy
from baseline_utils import process_answers, process_few_shot, calc_mean_and_var, process_personality_prompt

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
global_result = {}
global_cnt = {}

# Define constants
SCORES = {
    "A": 2, "B": 1
}

SCORES_BACK = {
    2: 'Accurate',
    1: 'Inaccurate',
    0: 'Unknown'
}

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

# Templates for personality assessment
TEMPLATE = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Accurate
(B). Inaccurate
"""

# 실행 파일 기준 디렉토리
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output, model_file):
    """Convert prompts to tokens based on the model type."""
    if 'llama-3' in model_file.lower():
        if model_output:
            con = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": model_output}
            ]
            return tokenizer.apply_chat_template(con)[:-5]
        else:
            con = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ]
            return tokenizer.apply_chat_template(con)
    else:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        dialog_tokens = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
        return tokenizer(dialog_tokens).input_ids


def getItems(filename):
    """Load data from files."""
    with open(os.path.join(filename, 'Test-set.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(os.path.join(filename, 'traintest_split_balanced_80_20.json'), encoding='utf-8') as f:
        split_data = json.load(f)
    return data, pd.read_excel(os.path.join(filename, 'all_data_ItemKey.xlsx')), split_data['train_index'], split_data['test_index']


def generateAnswer(tokenizer, model, dataset, template, scores=SCORES, system_prompt=SYSTEM_PROMPT, model_file=None):
    """Generate answers using the model."""
    batch_size = 3 if '70B' in model_file else 10
    questions = [item["text"].lower() for item in dataset]
    answers = []

    for batch in range(0, len(questions), batch_size):
        with torch.no_grad():
            outputs = model.generate(
                [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option', model_file)
                 for prompt in questions[batch:batch + batch_size]],
                max_new_tokens=15,
            )
            output_text = tokenizer.batch_decode(outputs)
            if 'llama-3' in model_file.lower():
                answer = [text.split("<|end_header_id|>")[3] for text in output_text]
            else:
                answer = [text.split("[/INST]")[-1] for text in output_text]
            answers.extend(answer)

    return answers


def calc_mean_and_var(result):
    """Calculate mean and variance of results."""
    mean = {key: np.mean(np.array(item)) for key, item in result.items()}
    std = {key: np.std(np.array(item)) for key, item in result.items()}
    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }


def from_index_to_data(train_index, test_index, text_file, dataset, dataset_set):
    """Convert indexed data to structured format."""
    data = []
    for i in tqdm(dataset):
        d_train = []
        d_test = []
        for t_i in train_index:
            t = text_file[text_file['Full#'] == t_i].iloc[0].to_list()
            item = {
                'label_raw': t[4],
                'text': t[5],
                'label_trait': t[3],
                'key': {'+': 1, '-': -1}[t[2][0]]
            }
            # exec 대신 직접 접근
            item['value'] = i[f'i{t_i}']
            item['case'] = i['case']
            d_train.append(item)
        for t_i in test_index:
            t = text_file[text_file['Full#'] == t_i].iloc[0].to_list()
            item = {
                'label_raw': t[4],
                'text': t[5],
                'label_trait': t[3],
                'key': {'+': 1, '-': -1}[t[2][0]]
            }
            item['value'] = i[f'i{t_i}']
            item['case'] = i['case']
            d_test.append(item)
        data.append({'train': d_train, 'test': d_test})
    return data


def get_activate_layer(layer_num, activate_name):
    lower_bound = layer_num // 4
    upper_bound = layer_num - lower_bound
    step = (upper_bound - lower_bound + 1) // 5

    if 'se' in activate_name:
        value = lower_bound
    elif 'im' in activate_name:
        value = lower_bound + 1 * step
    elif 'as' in activate_name:
        value = lower_bound + 2 * step
    return value


def lmean(l):
    """Calculate mean of a list."""
    if len(l) == 0:
        return 0.0
    return sum(l) / len(l)


def process_pas(data, model, tokenizer, model_file):
    """
    Process data using Persona Activation Steering (PAS) method.

    FIX 1: personal_data에 'question' 필드 추가 (preprocess_activate_dataset 호환)
    FIX 2: data[0] 기준으로 valid item 인덱스를 고정하여 activation-label 정렬 보장
    FIX 3: activation 추출 후 NaN/inf 검증
    """
    # --- Step 1: data[0] 기준으로 valid item 목록 및 personal_data 구성 ---
    # valid_item_indices: (personal, train_idx) 튜플 리스트 → 모든 sample에서 동일한 순서 보장
    valid_item_indices = []
    personal_data = []

    for personal in ['se', 'im', 'as']:
        for train_idx, item in enumerate(data[0]['train']):
            if item['label_trait'] == personal and item['value'] in [1, 2]:
                valid_item_indices.append((personal, train_idx))

                # FIX 1: 'question' 필드 추가 (pas.py의 preprocess_activate_dataset 필요)
                is_accurate_positive = (item['value'] == 2 and item['key'] == 1) or \
                                       (item['value'] == 1 and item['key'] == -1)

                question_text = TEMPLATE.format(item['text'].lower())

                if is_accurate_positive:
                    personal_data.append({
                        'question': question_text,
                        'answer_matching_behavior': 'A',
                        'answer_not_matching_behavior': 'B'
                    })
                else:
                    personal_data.append({
                        'question': question_text,
                        'answer_matching_behavior': 'B',
                        'answer_not_matching_behavior': 'A'
                    })

    print(f"Total valid items (from data[0]): {len(valid_item_indices)}")
    print(f"Total personal_data for activation extraction: {len(personal_data)}")

    # --- Step 2: Activation 추출 ---
    print("Preprocessing activation dataset...")
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)
    print(f"Total activations computed: {len(all_head_wise_activations)}")

    # FIX 3: NaN/inf 검증
    nan_count = 0
    for idx, act in enumerate(all_head_wise_activations):
        if not np.isfinite(act).all():
            nan_count += 1
            # NaN/inf를 0으로 대체
            all_head_wise_activations[idx] = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
    if nan_count > 0:
        print(f"WARNING: {nan_count}/{len(all_head_wise_activations)} activations contained NaN/inf (replaced with 0)")

    # --- Step 3: 각 sample에 대해 activation 수집 및 개입 계산 ---
    activations_list = []
    for index, sample in enumerate(tqdm(data, desc="Processing samples")):
        model.reset_all()

        system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them;' + \
                             ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" for it in sample['train']])

        labels = []
        head_wise_activations = []

        # FIX 2: valid_item_indices 기반으로 고정된 순서 사용
        for act_idx, (personal, train_idx) in enumerate(valid_item_indices):
            item = sample['train'][train_idx]

            # 이 sample에서도 해당 item이 valid한지 확인
            if item['value'] not in [1, 2]:
                # data[0]에서는 valid했지만 이 sample에서는 invalid → skip하되 activation index 증가
                continue

            is_accurate_positive = (item['value'] == 2 and item['key'] == 1) or \
                                   (item['value'] == 1 and item['key'] == -1)

            if is_accurate_positive:
                labels.extend([1, 0])
            else:
                labels.extend([0, 1])

            # activation index: 각 valid item은 2개 activation (pos, neg)
            head_wise_activations.extend([
                deepcopy(all_head_wise_activations[act_idx * 2]),
                deepcopy(all_head_wise_activations[act_idx * 2 + 1])
            ])

        print(f"Sample {index + 1}: {len(labels)} labels, {len(head_wise_activations)} activations")

        if len(labels) == 0 or len(head_wise_activations) == 0:
            print(f"WARNING: Sample {index + 1} has no valid data, skipping...")
            continue

        # label 균형 확인
        n_pos = sum(1 for l in labels if l == 1)
        n_neg = sum(1 for l in labels if l == 0)
        print(f"  Labels: {n_pos} positive, {n_neg} negative")

        print(f"Computing activations for sample {index + 1}/{len(data)}...")
        activate = model.get_activations(deepcopy(head_wise_activations), labels, num_to_intervene=24)

        activations_list.append({
            'case': sample['train'][0]['case'],
            'activations': activate,
            'system_prompt': system_prompt_text
        })

    return activations_list


def print_and_save_results(results, mode, model_file, dataset_set):
    """
    Print and save the final results of the personality assessment.
    FIX: se/im/as 3개 trait 대응 (Big Five 5개 하드코딩 제거)
    """
    print('*******Finally:******')

    # 동적으로 trait 수 감지
    trait_names = ['as', 'im', 'se']  # sorted alphabetically (calc_mean_and_var sorts by key)
    num_traits = len(trait_names)

    mean = [i['mean_ver']['mean'] for i in results]
    std = [i['mean_ver']['std'] for i in results]
    mean_abs = [i['mean_ver_abs']['mean'] for i in results]
    std_abs = [i['mean_ver_abs']['std'] for i in results]

    log_score = {}
    log_mean = {}
    log_std = {}

    for j, trait in enumerate(trait_names):
        trait_mean = [i[j][1] for i in mean]
        trait_std = [i[j][1] for i in std]
        trait_mean_abs = [i[j][1] for i in mean_abs]
        trait_std_abs = [i[j][1] for i in std_abs]

        log_score[f'mean_{trait}'] = lmean(trait_mean)
        log_score[f'std_{trait}'] = lmean(trait_std)
        log_score[f'mean_{trait}_abs'] = lmean(trait_mean_abs)
        log_score[f'std_{trait}_abs'] = lmean(trait_std_abs)

        log_mean[trait] = trait_mean
        log_std[trait] = trait_std

    log = {
        'score': log_score,
        'mean': log_mean,
        'std': log_std,
    }

    pprint(log)

    os.makedirs(os.path.join(SCRIPT_DIR, 'log'), exist_ok=True)
    log_filename = os.path.join(SCRIPT_DIR, 'log', f'{mode}_{model_file.split("/")[-1]}_{dataset_set}.json')
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {log_filename}")


def main(mode=None, model_file='', model=None, tokenizer=None, dataset_set='OOD'):
    """Main function to run personality assessment."""
    dataset, text_file, train_index, test_index = getItems(SCRIPT_DIR)
    print("-" * 40)
    print(f"Current Prompt: {TEMPLATE}")
    results = []
    data = from_index_to_data(train_index, test_index, text_file, dataset, dataset_set)

    if mode == 'NO_CHANGE':
        model.reset_all()
        answers = generateAnswer(tokenizer, model, data[0]['test'], TEMPLATE, model_file=model_file)
        results = process_answers(data, answers)

    elif mode == 'few-shot':
        results = process_few_shot(data, model, tokenizer, model_file)

    elif mode == 'personality_prompt':
        results = process_personality_prompt(data, model, tokenizer, model_file)

    elif mode == 'PAS':
        print("Computing activations for intervention...")
        activations_list = process_pas(data, model, tokenizer, model_file)

        os.makedirs(os.path.join(SCRIPT_DIR, 'activations'), exist_ok=True)
        output_file = os.path.join(SCRIPT_DIR, 'activations', f'PAS_{model_file.split("/")[-1]}_{dataset_set}.csv')

        csv_data = []
        for item in activations_list:
            row = {
                'case': item['case'],
                'system_prompt': item['system_prompt']
            }

            for layer_name, layer_data in item['activations'].items():
                for head_idx, head_data in enumerate(layer_data):
                    if hasattr(head_data, 'tolist'):
                        head_values = head_data.tolist()
                    else:
                        head_values = head_data

                    if isinstance(head_values, (list, tuple)):
                        for i, val in enumerate(head_values):
                            row[f'{layer_name}_head{head_idx}_dim{i}'] = val
                    else:
                        row[f'{layer_name}_head{head_idx}'] = head_values

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)

        print(f"\nActivations saved to: {output_file}")
        print(f"Total samples processed: {len(activations_list)}")
        print(f"CSV shape: {df.shape}")

        if activations_list:
            first_activation = activations_list[0]['activations']
            print(f"\nActivation summary:")
            print(f"  - Number of layers with interventions: {len(first_activation)}")
            for layer_name in list(first_activation.keys())[:3]:
                num_heads = len(first_activation[layer_name])
                print(f"  - {layer_name}: {num_heads} heads")

            # NaN 검증
            nan_directions = 0
            total_directions = 0
            for layer_name, interventions in first_activation.items():
                for head_no, direction, std in interventions:
                    total_directions += 1
                    if not np.isfinite(direction).all() or not np.isfinite(std):
                        nan_directions += 1
            print(f"  - Direction NaN check: {nan_directions}/{total_directions} contain NaN/inf")

        return

    if results:
        print_and_save_results(results, mode, model_file, dataset_set)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PAS for SE/IM/AS educational constructs")
    parser.add_argument("--modes", default='PAS', help="Mode: NO_CHANGE, few-shot, personality_prompt, PAS")
    parser.add_argument("--model_file", default='meta-llama/Meta-Llama-3-8B-Instruct', help="Model name or path")
    args = parser.parse_args()

    model_file = args.model_file
    modes = [args.modes]

    model, tokenizer = get_model(model_file)
    if 'llama-3' in model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    for mode in modes:
        main(mode=mode, model_file=model_file, model=model, tokenizer=tokenizer, dataset_set='OOD')
