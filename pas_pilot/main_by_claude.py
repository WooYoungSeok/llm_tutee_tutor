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
from pas import get_model
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


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output, model_file):
    """
    Convert prompts to tokens based on the model type.
    """
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
    """
    Load data from files.

    Test-set.json: 300 samples for Test, if you need to load more, you can download from huggingface (https://huggingface.co/datasets/WestlakeNLP/PAPI-300K)
    mpi_300_split.json: The original IPIP-NEO-300 questionnaire includes the IPIP-NEO-120 and the remaining 180 questions.
        Existing psychological research indicates that the IPIP-NEO-120 is sufficient to represent a person's personality traits.
        Therefore, we first use the IPIP-NEO-120 to concretize a person's characteristics,
        and then divide the remaining 180 questions into a section that we ask the model to predict.
    IPIP-NEO-ItemKey.xls:  This is where we store the question text corresponding to each question ID (the original data only contains the ID part).
    """
    with open(filename + 'Test-set.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(filename + 'traintest_split_balanced_80_20.json', encoding='utf-8') as f:
        split_data = json.load(f)
    return data, pd.read_excel(filename + 'all_data_ItemKey.xlsx'), split_data['train_index'], split_data['test_index']


def generateAnswer(tokenizer, model, dataset, template, scores=SCORES, system_prompt=SYSTEM_PROMPT, model_file=None):
    """
    Generate answers using the model.
    """
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
    """
    Calculate mean and variance of results.
    """
    mean = {key: np.mean(np.array(item)) for key, item in result.items()}
    std = {key: np.std(np.array(item)) for key, item in result.items()}
    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }


def from_index_to_data(train_index, test_index, text_file, dataset, dataset_set):
    """
    Convert indexed data to structured format.
    """
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
            exec(f"item['value'] = i['i{t_i}']")
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
            exec(f"item['value'] = i['i{t_i}']")
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
    """
    Calculate mean of a list.
    """
    return sum(l) / len(l)


def process_pas(data, model, tokenizer, model_file):
    """
    Process data using Persona Activation Steering (PAS) method.
    
    FIXED: Correctly handles key field to determine positive/negative personality
    """
    # Prepare personal data for activation (only for value 1 or 2, excluding 0: Unknown)
    personal_data = []
    for personal in ['se', 'im', 'as']:
        for item in data[0]['train']:
            if item['label_trait'] == personal and item['value'] in [1, 2]:
                # All questions use standard A/B format
                # A = Accurate, B = Inaccurate
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',  # A = Accurate
                    'answer_not_matching_behavior': 'B'  # B = Inaccurate
                })
    
    print(f"Total personal_data items (valid only): {len(personal_data)}")

    # Preprocess activation dataset
    print("Preprocessing activation dataset...")
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)
    print(f"Total activations computed: {len(all_head_wise_activations)}")

    activations_list = []
    for index, sample in enumerate(tqdm(data, desc="Processing samples")):
        model.reset_all()

        # Generate system prompt from training data
        system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them;' + \
                             ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" for it in sample['train']])

        # Prepare labels and activations for each personality trait
        labels = []
        head_wise_activations = []
        personal_number = 0
        
        for personal in ['se', 'im', 'as']:
            for item in sample['train']:
                if item['label_trait'] == personal:
                    if item['value'] in [1, 2]:  # Only process valid values (excluding 0: Unknown)
                        # FIXED: Use key field to determine positive/negative
                        # key == 1 ('+' question): Accurate (value=2) = Positive
                        # key == -1 ('-' question): Accurate (value=2) = Negative
                        
                        # Check if "Accurate" choice represents positive personality
                        is_accurate_positive = (item['value'] == 2 and item['key'] == 1) or \
                                              (item['value'] == 1 and item['key'] == -1)
                        
                        if is_accurate_positive:
                            labels.extend([1, 0])  # A (Accurate) is positive
                        else:
                            labels.extend([0, 1])  # B (Inaccurate) is positive
                        
                        head_wise_activations.extend([
                            deepcopy(all_head_wise_activations[personal_number]),
                            deepcopy(all_head_wise_activations[personal_number + 1])
                        ])
                        personal_number += 2  # Only increment for valid items

        print(f"Sample {index + 1}: {len(labels)} labels, {len(head_wise_activations)} activations")
        
        # Skip if no valid data
        if len(labels) == 0 or len(head_wise_activations) == 0:
            print(f"Warning: Sample {index + 1} has no valid data, skipping...")
            continue

        # Get activations for intervention
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

    Args:
    results (list): List of result dictionaries from the assessment
    mode (str): The mode of assessment used
    model_file (str): The name of the model file used
    dataset_set (str): The dataset set used (e.g., 'OOD')
    """
    print('*******Finally:******')

    # Extract mean and standard deviation values
    mean = [i['mean_ver']['mean'] for i in results]
    mean_A, mean_C, mean_E, mean_N, mean_O = ([i[j][1] for i in mean] for j in range(5))

    std = [i['mean_ver']['std'] for i in results]
    std_A, std_C, std_E, std_N, std_O = ([i[j][1] for i in std] for j in range(5))

    mean_abs = [i['mean_ver_abs']['mean'] for i in results]
    mean_A_abs, mean_C_abs, mean_E_abs, mean_N_abs, mean_O_abs = ([i[j][1] for i in mean_abs] for j in range(5))

    std_abs = [i['mean_ver_abs']['std'] for i in results]
    std_A_abs, std_C_abs, std_E_abs, std_N_abs, std_O_abs = ([i[j][1] for i in std_abs] for j in range(5))

    # Prepare log dictionary
    log = {
        'score': {
            'mean_A': lmean(mean_A), 'mean_C': lmean(mean_C), 'mean_E': lmean(mean_E),
            'mean_N': lmean(mean_N), 'mean_O': lmean(mean_O),
            'std_A': lmean(std_A), 'std_C': lmean(std_C), 'std_E': lmean(std_E),
            'std_N': lmean(std_N), 'std_O': lmean(std_O),
            'mean_A_abs': lmean(mean_A_abs), 'mean_C_abs': lmean(mean_C_abs),
            'mean_E_abs': lmean(mean_E_abs), 'mean_N_abs': lmean(mean_N_abs),
            'mean_O_abs': lmean(mean_O_abs),
            'std_A_abs': lmean(std_A_abs), 'std_C_abs': lmean(std_C_abs),
            'std_E_abs': lmean(std_E_abs), 'std_N_abs': lmean(std_N_abs),
            'std_O_abs': lmean(std_O_abs)
        }
    }

    # Print the log
    pprint(log)

    # Add more details to the log
    log['mean'] = {'A': mean_A, 'C': mean_C, 'E': mean_E, 'N': mean_N, 'O': mean_O}
    log['std'] = {'A': std_A, 'C': std_C, 'E': std_E, 'N': std_N, 'O': std_O}

    # Save the log to a file
    log_filename = f'./log/{mode}_{model_file.split("/")[-1]}_{dataset_set}.json'
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {log_filename}")


def main(mode=None, model_file='', model=None, tokenizer=None, dataset_set='OOD'):
    """
    Main function to run personality assessment.
    """
    dataset, text_file, train_index, test_index = getItems('./pas_pilot/')
    print("-" * 40)
    print(f"Current Prompt: {TEMPLATE}")
    results = []
    data = from_index_to_data(train_index, test_index, text_file, dataset, dataset_set)

    if mode == 'NO_CHANGE':
        # Process data without any changes
        model.reset_all()
        answers = generateAnswer(tokenizer, model, data[0]['test'], TEMPLATE, model_file=model_file)
        results = process_answers(data, answers)

    elif mode == 'few-shot':
        # Process data using few-shot learning
        results = process_few_shot(data, model, tokenizer, model_file)

    elif mode == 'personality_prompt':
        # Process data using personality prompts
        results = process_personality_prompt(data, model, tokenizer, model_file)

    elif mode == 'PAS':
        # Process data using PAS (Personality Assessment System)
        print("Computing activations for intervention...")
        activations_list = process_pas(data, model, tokenizer, model_file)
        # Save activations to file
        os.makedirs('./activations', exist_ok=True)
        output_file = f'./activations/PAS_{model_file.split("/")[-1]}_{dataset_set}.csv'
        
        # Convert activations to CSV format
        csv_data = []
        for item in activations_list:
            row = {
            'case': item['case'],
            'system_prompt': item['system_prompt']
            }
            
            # Flatten activation data
            for layer_name, layer_data in item['activations'].items():
                for head_idx, head_data in enumerate(layer_data):
                    # Convert tensor to list if needed
                    if hasattr(head_data, 'tolist'):
                        head_values = head_data.tolist()
                    else:
                        head_values = head_data
                    
                    # Add flattened values as separate columns
                    if isinstance(head_values, list):
                        for i, val in enumerate(head_values):
                            row[f'{layer_name}_head{head_idx}_dim{i}'] = val
                    else:
                        row[f'{layer_name}_head{head_idx}'] = head_values
            
            csv_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        
        print(f"\nActivations saved to: {output_file}")
        print(f"Total samples processed: {len(activations_list)}")
        print(f"CSV shape: {df.shape}")
                
        # Print summary
        if activations_list:
            first_activation = activations_list[0]['activations']
            print(f"\nActivation summary:")
            print(f"  - Number of layers with interventions: {len(first_activation)}")
            for layer_name in list(first_activation.keys())[:3]:  # Show first 3 layers
                num_heads = len(first_activation[layer_name])
                print(f"  - {layer_name}: {num_heads} heads")
        
        return  # Early return for PAS mode

    # Print and save final results (for non-PAS modes)
    if results:
        print_and_save_results(results, mode, model_file, dataset_set)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="mode")
    parser.add_argument("--modes", default='PAS', help="Name of the user to greet")
    parser.add_argument("--model_file", default='meta-llama/Meta-Llama-3-8B-Instruct', help="Name of the user to greet")
    args = parser.parse_args()

    model_file = args.model_file
    modes = [args.modes]

    model, tokenizer = get_model(model_file)
    if 'llama-3' in model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    for mode in modes:
        main(mode=mode, model_file=model_file, model=model, tokenizer=tokenizer, dataset_set='OOD')