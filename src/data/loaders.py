"""
Concrete implementations of instruction dataset loaders.
"""

from typing import List, Any, Optional
from datasets import load_dataset  # type: ignore[import-untyped]

from .base import InstructionDataset


class HarmfulInstructionDataset(InstructionDataset):
    """
    Dataset of harmful instructions that typically trigger refusals.
    
    Default source: mlabonne/harmful_behaviors
    """
    
    def __init__(
        self,
        dataset_name: str = "mlabonne/harmful_behaviors",
        instruction_column: str = "text",
        split: str = "train",
        n_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.instruction_column = instruction_column
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self._instructions: List[str] = []
        self._loaded = False
    
    def load(self) -> None:
        """Load harmful instructions from HuggingFace Hub."""
        if self._loaded:
            return
            
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        # Sample if requested
        if self.n_samples is not None and self.n_samples < len(dataset):
            dataset = dataset.shuffle(seed=self.seed).select(range(self.n_samples))
        
        self._instructions = dataset[self.instruction_column]
        self._loaded = True
    
    def get_instructions(self) -> List[str]:
        """Get raw harmful instruction strings."""
        if not self._loaded:
            self.load()
        return self._instructions
    
    def get_formatted_prompts(self, tokenizer: Any) -> List[str]:
        """Format instructions with chat template."""
        if not self._loaded:
            self.load()
            
        formatted = []
        for instruction in self._instructions:
            messages = self.as_chat_messages(instruction)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted.append(prompt)
        return formatted
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._instructions)


class HarmlessInstructionDataset(InstructionDataset):
    """
    Dataset of harmless instructions for baseline comparison.
    
    Default source: tatsu-lab/alpaca
    """
    
    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        instruction_column: str = "instruction",
        split: str = "train",
        n_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.instruction_column = instruction_column
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self._instructions: List[str] = []
        self._loaded = False
    
    def load(self) -> None:
        """Load harmless instructions from HuggingFace Hub."""
        if self._loaded:
            return
            
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        # Filter out empty instructions
        dataset = dataset.filter(
            lambda x: x[self.instruction_column] and len(x[self.instruction_column].strip()) > 0
        )
        
        # Sample if requested
        if self.n_samples is not None and self.n_samples < len(dataset):
            dataset = dataset.shuffle(seed=self.seed).select(range(self.n_samples))
        
        self._instructions = dataset[self.instruction_column]
        self._loaded = True
    
    def get_instructions(self) -> List[str]:
        """Get raw harmless instruction strings."""
        if not self._loaded:
            self.load()
        return self._instructions
    
    def get_formatted_prompts(self, tokenizer: Any) -> List[str]:
        """Format instructions with chat template."""
        if not self._loaded:
            self.load()
            
        formatted = []
        for instruction in self._instructions:
            messages = self.as_chat_messages(instruction)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted.append(prompt)
        return formatted
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._instructions)
