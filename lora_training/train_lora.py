"""
LoRA Fine-Tuning for Braille-Native Cognition

Uses Unsloth for efficient LoRA training on consumer hardware.
Trains the model to truly think in braille, not just translate.

Requirements:
    pip install unsloth transformers datasets peft accelerate bitsandbytes
"""

import os
import json
import torch
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    try:
        import unsloth
    except ImportError:
        missing.append("unsloth")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    
    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def load_dataset():
    """Load the braille training dataset"""
    dataset_path = Path(__file__).parent / "braille_alpaca.json"
    
    if not dataset_path.exists():
        print("Dataset not found. Run prepare_dataset.py first.")
        return None
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    return data


def format_prompt(example):
    """Format example for training"""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    return prompt


def train():
    """Main training function"""
    
    print("=" * 60)
    print("LoRA Fine-Tuning for Braille-Native Cognition")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    # Load dataset
    data = load_dataset()
    if data is None:
        return
    
    # Model configuration
    max_seq_length = 2048
    model_name = "unsloth/llama-3.2-1b-instruct-bnb-4bit"  # Small, fast model
    
    print(f"\nLoading base model: {model_name}")
    
    # Load model with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Prepare dataset
    formatted_data = [{"text": format_prompt(ex)} for ex in data]
    dataset = Dataset.from_list(formatted_data)
    
    print(f"Training on {len(dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./braille_lora_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Short training for demo
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save LoRA adapters
    output_path = Path(__file__).parent / "braille_lora_adapters"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nLoRA adapters saved to {output_path}")
    
    # Save merged model for Ollama
    print("\nMerging LoRA adapters with base model...")
    merged_path = Path(__file__).parent / "braille_lora_merged"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {merged_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nTo use with Ollama:")
    print("  1. Convert to GGUF: python -m llama.cpp.convert ...")
    print("  2. Create Modelfile pointing to the GGUF")
    print("  3. ollama create braille-lora -f Modelfile")


def train_simple():
    """Simplified training without Unsloth (uses transformers directly)"""
    
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("=" * 60)
    print("LoRA Training for Braille-Native Cognition")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers peft datasets")
        return
    
    # Load dataset
    data = load_dataset()
    if data is None:
        return
    
    # Use a very small model for M1 Pro 16GB
    model_name = "distilgpt2"  # 82M params, fits easily in memory
    
    print(f"Loading model: {model_name} (82M params)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    
    # Move to MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # LoRA config - GPT2 uses different attention module names
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],  # GPT2 attention layer
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset with labels
    def tokenize(example):
        prompt = format_prompt(example)
        encodings = tokenizer(prompt, truncation=True, max_length=512, padding="max_length", return_tensors=None)
        # For causal LM, labels = input_ids (model predicts next token)
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings
    
    dataset = Dataset.from_list(data)
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    
    # Training
    training_args = TrainingArguments(
        output_dir="./braille_lora_simple",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        use_cpu=False,  # Use MPS on Mac
        dataloader_pin_memory=False,  # Disable for MPS
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,  # Updated from deprecated 'tokenizer'
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save
    output_path = Path(__file__).parent / "braille_lora_simple_output"
    model.save_pretrained(output_path)
    print(f"\nLoRA adapters saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        train_simple()
    else:
        train()
