# llm_backend/transformers_backend.py
from collections.abc import Generator
from typing import Dict, List, Union

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBackend
from .utils import truncate_messages


class PEFTTransformersBackend(LLMBackend):
    def __init__(self, base_model: str, peft_path: str):
        config = PeftConfig.from_pretrained(peft_path)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(base, peft_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_prompt(self, messages: List[Dict]) -> str:
        prompt = "You are a helpful assistant.\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prefix = "User" if role == "user" else "Assistant"
            prompt += f"{prefix}: {content}\n"
        prompt += "Assistant:"
        return prompt

    def _count_tokens(self, messages: List[Dict]) -> int:
        total = 0
        for msg in messages:
            tokens = self.tokenizer.encode(msg["content"], add_special_tokens=False)
            total += len(tokens)
        return total

    def chat(
        self, messages: List[Dict], stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        messages = truncate_messages(messages, self)
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_new_tokens = 256
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Extract only the assistant's reply after last "Assistant:" token
        reply = output.split("Assistant:")[-1].strip()
        return reply

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Embedding not implemented for Transformers backend.")

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def count_tokens(self, messages: List[Dict]) -> int:
        return sum(len(self.tokenize(msg["content"])) for msg in messages)

    def reset(self) -> None:
        pass


class TransformersBackend(PEFTTransformersBackend):
    def __init__(self, model_name: str):
        super().__init__(base_model=model_name, peft_path=None)
