"""
Token Usage and Cost Tracking Utility for different Models

This module provides a TokenCostTracker class to monitor token usage and estimate API costs
for language models during text generation or processing tasks.

Features:
- Supports multiple models with configurable cost rates per million tokens.
- Counts input and output tokens using the tiktoken tokenizer.
- Tracks cumulative and per-entry token usage.
- Calculates estimated costs based on token usage and model-specific pricing.
- Optionally logs usage statistics to a JSON file with timestamps.
- Handles single strings or lists of strings for input/output texts.
- Provides safe fallbacks and warnings for tokenizer loading and encoding errors.
"""


import tiktoken
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other):
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens
        )


class TokenCostTracker:
    # Define cost rates per 1M tokens
    COST_RATES = {
        "openai/gpt-4o": {"input": 2.50, "output": 10.00},
        "openai/gpt-4o-mini": {"input": 0.17, "output": 0.66},
        "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        # Add as many models as needed
    }

    def __init__(self, model_name: str, log_file: Optional[str] = None):
        """
        Initialize the token cost tracker.

        Args:
            model_name: Name of the model being used (can include API provider prefix)
            log_file: Optional path to save usage logs
        """
        if model_name not in self.COST_RATES:
            raise ValueError(f"No cost rates defined for model {model_name}")

        self.model_name = model_name
        self.tiktoken_model_name = model_name.split('/')[-1]  # Remove provider prefix for tiktoken
        self.log_file = log_file or f"token_usage_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.total_usage = TokenUsage()
        self.entries_processed = 0

        # Initialize tiktoken encoder
        try:
            self.encoding = tiktoken.encoding_for_model(self.tiktoken_model_name)
        except Exception as e:
            print(f"Warning: Could not load tiktoken for {self.tiktoken_model_name}. Using cl100k_base instead.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not text:
            return 0

            # Ensure text is a string
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                print(f"Warning: Could not convert input to string: {e}")
                return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print(f"Warning: Error encoding text: {e}")
            return 0

    def add_usage(self, input_text: Union[str, List[str]], output_text: Union[str, List[str]]) -> TokenUsage:
        """
        Track token usage for input and output text.

        Args:
            input_text: Input text or list of texts
            output_text: Output text or list of texts

        Returns:
            TokenUsage object with token counts
        """
        # Convert to lists if single strings
        input_texts = [input_text] if isinstance(input_text, str) else input_text
        output_texts = [output_text] if isinstance(output_text, str) else output_text

        # Count tokens
        usage = TokenUsage(
            sum(self.count_tokens(text) for text in input_texts),
            sum(self.count_tokens(text) for text in output_texts)
        )

        # Update total usage
        self.total_usage += usage
        self.entries_processed += 1

        return usage

    def calculate_cost(self, usage: TokenUsage) -> Dict:
        """Calculate costs based on token usage."""
        rates = self.COST_RATES[self.model_name]

        input_cost = (usage.input_tokens / 1_000_000) * rates["input"]
        output_cost = (usage.output_tokens / 1_000_000) * rates["output"]
        total_cost = input_cost + output_cost

        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }

    def get_current_usage(self) -> Dict:
        """Get current usage statistics."""
        total_stats = self.calculate_cost(self.total_usage)

        if self.entries_processed > 0:
            avg_usage = TokenUsage(
                self.total_usage.input_tokens // self.entries_processed,
                self.total_usage.output_tokens // self.entries_processed
            )
            avg_stats = self.calculate_cost(avg_usage)
        else:
            avg_stats = self.calculate_cost(TokenUsage())

        return {
            "total_stats": total_stats,
            "average_per_entry": avg_stats,
            "entries_processed": self.entries_processed
        }

    def save_usage_log(self):
        """Save usage statistics to a log file."""
        usage_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "usage_stats": self.get_current_usage()
        }

        try:
            with open(self.log_file, 'w') as f:
                json.dump(usage_data, f, indent=4)
        except Exception as e:
            print(f"Error saving usage log: {e}")

