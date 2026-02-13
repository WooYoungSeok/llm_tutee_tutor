"""
chat.py
=======
Steered 모델과 대화하며 Alpha 값 조정

각 trait별로 alpha 값 및 personality level을 실시간으로 조정하며 대화할 수 있습니다.

실행:
    # 기본 (system prompt 없음)
    python chat.py

    # Trait별 level 지정 (system prompt 설정)
    python chat.py --se high --im low --as high

    # 특정 trait만 지정
    python chat.py --trait se --level high

명령어:
    /alpha se 2.0    - SE의 alpha를 2.0으로 설정
    /alpha im -1.0   - IM의 alpha를 -1.0으로 설정 (반대 방향)
    /alpha as 0      - AS steering 끄기
    /level se high   - SE level을 high로 변경 (system prompt 업데이트)
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
    OUTPUT_DIR, TRAITS, TRAIT_NAMES, DEFAULT_MODEL, DEFAULT_ALPHA, SYSTEM_PROMPTS
)
from steering import ActivationSteering
from data_utils import get_system_prompt_for_levels


class InteractiveChat:
    """
    Trait별 Alpha 조정이 가능한 대화 인터페이스
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        output_dir: str = OUTPUT_DIR,
        trait_levels: dict = None
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

        # Trait별 level (system prompt용): {'se': 'high', 'im': 'low', 'as': 'high'}
        self.trait_levels = trait_levels or {}

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

    def get_system_prompt(self) -> str:
        """현재 trait_levels에서 system prompt 생성"""
        if self.trait_levels:
            return get_system_prompt_for_levels(self.trait_levels)
        return "You are a helpful assistant."

    def generate_response(self, user_input: str, max_new_tokens: int = 512) -> str:
        """현재 steering 설정으로 응답 생성"""
        # Steering 적용
        self.apply_current_steering()

        # system prompt 결정 (trait_levels 기반)
        system_prompt = self.get_system_prompt()

        # 프롬프트 생성 (chat template 적용)
        if 'llama-3' in self.model_name.lower():
            messages = [{"role": "system", "content": system_prompt}]
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
            prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"
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
            level = self.trait_levels.get(trait, "not set")
            status = "LOADED" if loaded else "NOT LOADED"
            active = "ACTIVE" if loaded and abs(alpha) > 1e-6 else "INACTIVE"

            print(f"  {TRAIT_NAMES[trait]} ({trait.upper()}):")
            print(f"    Level: {level}")
            print(f"    Status: {status}")
            print(f"    Alpha: {alpha:+.2f}")
            print(f"    Steering: {active}")

        print("\n  System Prompt Preview:")
        sys_prompt = self.get_system_prompt()
        preview = sys_prompt[:120].replace('\n', ' ')
        print(f"    {preview}...")
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

Personality Level (System Prompt):
  /level <trait> <high|low|none>  - Set trait level for system prompt
                                    Example: /level se high
                                    Example: /level im low
                                    Example: /level as none  (remove)
                                    Note: clears chat history

Intervention Management:
  /load <trait>           - Load intervention file for a trait
  /load all               - Load all available interventions

Status:
  /status                 - Show current alpha values, levels, and status

Other:
  /clear                  - Clear chat history
  /help                   - Show this help message
  /quit, /exit, /q        - Exit the chat

Traits:
  se  - Self-Efficacy (Academic Self-Efficacy)
  im  - Intrinsic Motivation
  as  - Academic Stress

Example Session:
  # Start with SE-high, IM-low, AS-high persona
  python chat.py --se high --im low --as high

  /load all
  /alpha se 2.0
  /alpha im 1.5
  /status
  Tell me about your approach to learning mathematics.
  /level se low           (switch to SE-low persona)
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

        elif command == '/level':
            if len(parts) < 3:
                print("Usage: /level <trait> <high|low|none>")
                print("Example: /level se high")
            else:
                trait = parts[1].lower()
                level = parts[2].lower()
                if trait not in TRAITS:
                    print(f"Unknown trait: {trait}")
                    print(f"Available traits: {', '.join(TRAITS)}")
                elif level == 'none':
                    self.trait_levels.pop(trait, None)
                    print(f"Cleared {trait.upper()} level (removed from system prompt)")
                elif level in ('high', 'low'):
                    self.trait_levels[trait] = level
                    print(f"Set {trait.upper()} level to {level}")
                    self.history = []  # system prompt 변경 시 히스토리 초기화
                    print("Chat history cleared due to system prompt change.")
                else:
                    print(f"Invalid level: {level}. Use 'high', 'low', or 'none'")

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
    parser = argparse.ArgumentParser(
        description="Interactive Chat with Steered LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # No system prompt (default)
  python chat.py

  # Set all trait levels
  python chat.py --se high --im low --as high

  # Set single trait (shorthand: --trait + --level)
  python chat.py --trait se --level high

  # Mix: some traits set, others not
  python chat.py --se high --im high
"""
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory containing intervention files")

    # Per-trait level args
    parser.add_argument("--se", type=str, choices=['high', 'low'],
                        help="Self-Efficacy level (high/low)")
    parser.add_argument("--im", type=str, choices=['high', 'low'],
                        help="Intrinsic Motivation level (high/low)")
    parser.add_argument("--as", dest="as_level", type=str, choices=['high', 'low'],
                        help="Academic Stress level (high/low)")

    # Shorthand: --trait se --level high (sets that single trait)
    parser.add_argument("--trait", type=str, choices=TRAITS,
                        help="Single trait to configure (use with --level)")
    parser.add_argument("--level", type=str, choices=['high', 'low'],
                        help="Level for --trait (high/low)")

    args = parser.parse_args()

    # trait_levels 딕셔너리 구성
    trait_levels = {}
    if args.se:
        trait_levels['se'] = args.se
    if args.im:
        trait_levels['im'] = args.im
    if args.as_level:
        trait_levels['as'] = args.as_level
    # --trait + --level shorthand
    if args.trait and args.level:
        trait_levels[args.trait] = args.level

    if trait_levels:
        print("\nPersonality profile:")
        for t, lv in trait_levels.items():
            print(f"  {TRAIT_NAMES[t]} ({t.upper()}): {lv}")

    # 모델 로드
    model, tokenizer = load_model(args.model_name, args.use_4bit)

    # 대화 시작
    chat = InteractiveChat(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        output_dir=args.output_dir,
        trait_levels=trait_levels
    )

    chat.run()


if __name__ == "__main__":
    main()
