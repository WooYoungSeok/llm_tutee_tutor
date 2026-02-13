"""
chat.py
=======
Steered 모델과 대화하며 Alpha 값 조정 (단일 trait 모드)

특정 trait에 대해 persona prompt를 적용하고 alpha 값을 조정하며 대화합니다.

실행:
    python chat.py --trait se           # SE trait으로 대화
    python chat.py --trait im --alpha 2.0  # IM trait, alpha=2.0
    python chat.py --trait as --level low  # AS-low persona로 대화

명령어:
    /alpha <value>   - Alpha 값 설정 (예: /alpha 2.0)
    /level high      - High level persona로 전환
    /level low       - Low level persona로 전환
    /reset           - Alpha를 0으로 리셋
    /status          - 현재 상태 확인
    /clear           - 대화 히스토리 초기화
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
    OUTPUT_DIR, TRAITS, TRAIT_NAMES, DEFAULT_MODEL, DEFAULT_ALPHA,
    get_persona_prompt
)
from steering import ActivationSteering


class SingleTraitChat:
    """
    단일 Trait에 대한 대화 인터페이스
    Persona prompt + Activation Steering 적용
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        trait: str,
        level: str = 'high',
        alpha: float = DEFAULT_ALPHA,
        output_dir: str = OUTPUT_DIR
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.trait = trait
        self.level = level
        self.alpha = alpha
        self.output_dir = output_dir

        # Multi-GPU 대응
        self.device = model.model.layers[0].self_attn.o_proj.weight.device

        # Steering 초기화
        self.steering = ActivationSteering(model, tokenizer, model_name)

        # Intervention 로드
        self.interventions = None
        self.load_intervention()

        # 채팅 히스토리
        self.history = []

    def load_intervention(self, filepath: str = None):
        """Intervention 파일 로드"""
        if filepath is None:
            pattern = os.path.join(self.output_dir, f"interventions_{self.trait}_*.pkl")
            files = sorted(glob(pattern))
            if not files:
                print(f"WARNING: No intervention file found for {self.trait}")
                return False
            filepath = files[-1]

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False

        with open(filepath, 'rb') as f:
            self.interventions = pickle.load(f)

        n_heads = sum(len(h) for h in self.interventions.values())
        print(f"Loaded {self.trait.upper()} intervention: {n_heads} heads from {filepath}")
        return True

    def get_current_persona_prompt(self) -> str:
        """현재 level에 맞는 persona prompt 반환"""
        return get_persona_prompt(self.trait, self.level)

    def apply_steering(self):
        """현재 alpha로 steering 적용"""
        self.steering.reset()

        if self.interventions and abs(self.alpha) > 1e-6:
            self.steering.apply_steering(self.interventions, alpha=self.alpha)

    def generate_response(self, user_input: str, max_new_tokens: int = 150) -> str:
        """현재 설정으로 응답 생성"""
        # Steering 적용
        self.apply_steering()

        # Persona prompt 가져오기
        persona_prompt = self.get_current_persona_prompt()

        # 프롬프트 생성
        if 'llama-3' in self.model_name.lower():
            messages = [
                {"role": "system", "content": persona_prompt}
            ]
            # 히스토리 추가
            for h in self.history[-4:]:
                messages.append({"role": "user", "content": h['user']})
                messages.append({"role": "assistant", "content": h['assistant']})
            messages.append({"role": "user", "content": user_input})

            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
        else:
            prompt = f"[INST] <<SYS>>\n{persona_prompt}\n<</SYS>>\n\n{user_input} [/INST]"
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
        print(f"Current Status: {TRAIT_NAMES[self.trait]} ({self.trait.upper()})")
        print("="*50)
        print(f"  Trait Level: {self.level.upper()}")
        print(f"  Alpha: {self.alpha:+.2f}")
        print(f"  Intervention: {'LOADED' if self.interventions else 'NOT LOADED'}")
        print(f"  Steering: {'ACTIVE' if self.interventions and abs(self.alpha) > 1e-6 else 'INACTIVE'}")
        print("\nPersona Prompt:")
        print(f"  {self.get_current_persona_prompt()[:100]}...")
        print("="*50 + "\n")

    def print_help(self):
        """도움말 출력"""
        print(f"""
========================================
Interactive Chat - {TRAIT_NAMES[self.trait]} ({self.trait.upper()})
========================================

Alpha Adjustment:
  /alpha <value>  - Set alpha value
                    Example: /alpha 2.0
                    Positive: steer towards HIGH {self.trait.upper()}
                    Negative: steer towards LOW {self.trait.upper()}

  /reset          - Reset alpha to 0 (no steering)

Persona Level:
  /level high     - Use HIGH {self.trait.upper()} persona prompt
  /level low      - Use LOW {self.trait.upper()} persona prompt

Status:
  /status         - Show current settings

Other:
  /clear          - Clear chat history
  /help           - Show this help message
  /quit           - Exit the chat

Example Session:
  /level high
  /alpha 2.0
  /status
  Tell me about your approach to learning.
  /alpha -2.0
  /level low
  Tell me about your approach to learning.
========================================
""")

    def process_command(self, cmd: str) -> bool:
        """명령어 처리"""
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
            self.alpha = 0.0
            print("Alpha reset to 0")
            self.print_status()

        elif command == '/clear':
            self.history = []
            print("Chat history cleared")

        elif command == '/alpha':
            if len(parts) < 2:
                print("Usage: /alpha <value>")
                print(f"Current alpha: {self.alpha:+.2f}")
            else:
                try:
                    value = float(parts[1])
                    self.alpha = value
                    print(f"Set alpha to {value:+.2f}")
                except ValueError:
                    print(f"Invalid alpha value: {parts[1]}")

        elif command == '/level':
            if len(parts) < 2:
                print("Usage: /level <high|low>")
                print(f"Current level: {self.level}")
            elif parts[1].lower() in ['high', 'low']:
                self.level = parts[1].lower()
                print(f"Set level to {self.level.upper()}")
                print(f"Persona: {self.get_current_persona_prompt()[:80]}...")
            else:
                print(f"Invalid level: {parts[1]}")
                print("Available levels: high, low")

        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands")

        return True

    def run(self):
        """대화 루프 실행"""
        print("\n" + "#"*50)
        print(f"Interactive Chat: {TRAIT_NAMES[self.trait]} ({self.trait.upper()})")
        print("#"*50)
        print("\nType /help for commands, /quit to exit\n")

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
                import traceback
                traceback.print_exc()


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
    parser = argparse.ArgumentParser(description="Interactive Chat with Steered LLM (Single Trait)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--trait", type=str, required=True, choices=TRAITS,
                        help="Trait to use for steering (se, im, as)")
    parser.add_argument("--level", type=str, default='high', choices=['high', 'low'],
                        help="Initial persona level (high or low)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Initial alpha value for steering")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory containing intervention files")

    args = parser.parse_args()

    print(f"\n{'#'*50}")
    print(f"Starting chat for {TRAIT_NAMES[args.trait]} ({args.trait.upper()})")
    print(f"  Level: {args.level.upper()}")
    print(f"  Alpha: {args.alpha:+.2f}")
    print(f"{'#'*50}\n")

    # 모델 로드
    model, tokenizer = load_model(args.model_name, args.use_4bit)

    # 대화 시작
    chat = SingleTraitChat(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        trait=args.trait,
        level=args.level,
        alpha=args.alpha,
        output_dir=args.output_dir
    )

    chat.run()


if __name__ == "__main__":
    main()
