"""
Abstract base classes for instruction datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class InstructionDataset(ABC):
    """
    Abstract base class for instruction datasets.
    
    Subclasses must implement methods to load and format instructions
    for use in refusal direction extraction.
    """
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset from source."""
    
    @abstractmethod
    def get_instructions(self) -> List[str]:
        """
        Get raw instruction strings.
        
        Returns:
            List of instruction strings
        """
    
    @abstractmethod
    def get_formatted_prompts(self, tokenizer: Any) -> List[str]:
        """
        Get instructions formatted with the model's chat template.
        
        Args:
            tokenizer: HuggingFace tokenizer with chat template
            
        Returns:
            List of formatted prompt strings ready for tokenization
        """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of instructions."""
    
    def as_chat_messages(self, instruction: str) -> List[Dict[str, str]]:
        """
        Convert a single instruction to chat message format.
        
        Args:
            instruction: Raw instruction string
            
        Returns:
            List with single user message dict
        """
        return [{"role": "user", "content": instruction}]
