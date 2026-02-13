"""
chat.py
=======
Steered 모델과 대화하며 Alpha 값 조정

각 trait별로 alpha 값을 실시간으로 조정하며 대화할 수 있습니다.

실행:
    python chat.py

명령어:
    /alpha se 2.0    - SE의 alpha를 2.0으로 설정
    /alpha im -1.0   - IM의 alpha를 -1.0으로 설정 (반대 방향)
    /alpha as 0      - AS steering 끄기
    /reset           - 모든 alpha를 0으로 리셋
    /status          - 현재 alpha 값 확인
    /load se         - SE intervention 파일 로드
    /help            - 도움말 출력
    /quit            - 종료
"""

import argparse
import os
import pickle
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    OUTPUT_DIR, TRAITS, TRAIT_NAMES, DEFAULT_MODEL, DEFAULT_ALPHA
)
from steering import ActivationSteering


class InteractiveChat:
    """
    Trait별 Alpha 조정이 가능한 대화 인터페이스
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        output_dir: str = OUTPUT_DIR
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = next(model.parameters()).device

        # Steering 초기화
        self.steering = ActivationSteering(model, tokenizer, model_name)

        # Trait별 interventions 및 alpha
        self.interventions = {}  # {trait: intervention_dict}
        self.alphas = {trait: 0.0 for trait in TRAITS}  # 초기값 0 (steering 없음)

        # 채팅 히스토리
        self.history = []

    def load_intervention(self, trait: str, filepath: str = None):
        """Intervention 파일 로드"""
        if filepath is None:
            # 최신 파일 찾기
            pattern = os.path.join(self.output_dir, f"interventions_{trait}_*.pkl")
            files = sorted(glob(pattern))
            if not files:
                print(f"No intervention file found for {trait}")
                return False
            filepath = files[-1]

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False

        with open(filepath, 'rb') as f:
            self.interventions[trait] = pickle.load(f)

        n_heads = sum(len(h) for h in self.interventions[trait].values())
        print(f"Loaded {trait.upper()} intervention: {n_heads} heads from {filepath}")
        return True

    def load_all_interventions(self):
        """모든 trait의 intervention 로드 시도"""
        for trait in TRAITS:
            self.load_intervention(trait)

    def apply_current_steering(self):
        """현재 alpha 설정으로 steering 적용"""
        # 먼저 리셋
        self.steering.reset()

        # 각 trait의 intervention을 현재 alpha로 적용
        for trait in TRAITS:
            if trait not in self.interventions:
                continue

            alpha = self.alphas[trait]
            if abs(alpha) < 1e-6:  # alpha가 0이면 skip
                continue

            interventions = self.interventions[trait]
            self.steering.apply_steering(interventions, alpha=alpha)

    def generate_response(self, user_input: str, max_new_tokens: int = 512) -> str:
        """현재 steering 설정으로 응답 생성"""
        # Steering 적용
        self.apply_current_steering()

        # 프롬프트 생성
        if 'llama-3' in self.model_name.lower():
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            # 히스토리 추가
            for h in self.history[-4:]:  # 최근 4턴만 유지
                messages.append({"role": "user", "content": h['user']})
                messages.append({"role": "assistant", "content": h['assistant']})
            messages.append({"role": "user", "content": user_input})

            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
        else:
            # Llama-2 형식
            prompt = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{user_input} [/INST]"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 디코딩
        generated_tokens = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 히스토리 저장
        self.history.append({'user': user_input, 'assistant': response})

        return response

    def print_status(self):
        """현재 상태 출력"""
        print("\n" + "="*50)
        print("Current Steering Status")
        print("="*50)

        for trait in TRAITS:
            loaded = trait in self.interventions
            alpha = self.alphas[trait]
            status = "LOADED" if loaded else "NOT LOADED"
            active = "ACTIVE" if loaded and abs(alpha) > 1e-6 else "INACTIVE"

            print(f"  {TRAIT_NAMES[trait]} ({trait.upper()}):")
            print(f"    Status: {status}")
            print(f"    Alpha: {alpha:+.2f}")
            print(f"    Steering: {active}")

        print("="*50 + "\n")

    def print_help(self):
        """도움말 출력"""
        print("""
========================================
Interactive Chat Commands
========================================

Alpha Adjustment:
  /alpha <trait> <value>  - Set alpha for a trait
                            Example: /alpha se 2.0
                            Negative values steer in opposite direction
                            Example: /alpha im -1.5

  /reset                  - Reset all alphas to 0

Intervention Management:
  /load <trait>           - Load intervention file for a trait
  /load all               - Load all available interventions

Status:
  /status                 - Show current alpha values and status

Other:
  /clear                  - Clear chat history
  /help                   - Show this help message
  /quit, /exit, /q        - Exit the chat

Traits:
  se  - Self-Efficacy
  im  - Intrinsic Motivation
  as  - Academic Stress

Example Session:
  /load all
  /alpha se 2.0
  /alpha im 1.5
  /status
  Tell me about your approach to learning mathematics.
  /alpha se -1.0
  Tell me about your approach to learning mathematics.
========================================
""")

    def process_command(self, cmd: str) -> bool:
        """
        명령어 처리

        Returns:
            True if should continue, False if should exit
        """
        parts = cmd.strip().split()
        if not parts:
            return True

        command = parts[0].lower()

        if command in ['/quit', '/exit', '/q']:
            print("Goodbye!")
            return False

        elif command == '/help':
            self.print_help()

        elif command == '/status':
            self.print_status()

        elif command == '/reset':
            for trait in TRAITS:
                self.alphas[trait] = 0.0
            print("All alphas reset to 0")
            self.print_status()

        elif command == '/clear':
            self.history = []
            print("Chat history cleared")

        elif command == '/load':
            if len(parts) < 2:
                print("Usage: /load <trait> or /load all")
            elif parts[1].lower() == 'all':
                self.load_all_interventions()
            elif parts[1].lower() in TRAITS:
                self.load_intervention(parts[1].lower())
            else:
                print(f"Unknown trait: {parts[1]}")
                print(f"Available traits: {', '.join(TRAITS)}")

        elif command == '/alpha':
            if len(parts) < 3:
                print("Usage: /alpha <trait> <value>")
                print("Example: /alpha se 2.0")
            else:
                trait = parts[1].lower()
                if trait not in TRAITS:
                    print(f"Unknown trait: {trait}")
                    print(f"Available traits: {', '.join(TRAITS)}")
                else:
                    try:
                        value = float(parts[2])
                        self.alphas[trait] = value
                        print(f"Set {trait.upper()} alpha to {value:+.2f}")

                        if trait not in self.interventions:
                            print(f"Warning: No intervention loaded for {trait}. Use /load {trait}")
                    except ValueError:
                        print(f"Invalid alpha value: {parts[2]}")

        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands")

        return True

    def run(self):
        """대화 루프 실행"""
        print("\n" + "#"*50)
        print("Interactive Chat with Steered LLM")
        print("#"*50)
        print("\nType /help for commands, /quit to exit")
        print("Loading available interventions...\n")

        self.load_all_interventions()
        self.print_status()

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    if not self.process_command(user_input):
                        break
                else:
                    print("\nAssistant: ", end="", flush=True)
                    response = self.generate_response(user_input)
                    print(response)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\nError: {e}")


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


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with Steered LLM")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory containing intervention files")

    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_model(args.model_name, args.use_4bit)

    # 대화 시작
    chat = InteractiveChat(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        output_dir=args.output_dir
    )

    chat.run()


if __name__ == "__main__":
    main()
