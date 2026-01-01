"""
Test the fine-tuned LoRA model for braille understanding
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def test_model():
    print("=" * 60)
    print("Testing Braille-Native LoRA Model")
    print("=" * 60)
    
    # Load base model
    model_name = "distilgpt2"
    adapter_path = Path(__file__).parent / "braille_lora_simple_output"
    
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}\n")
    
    # Test prompts
    test_prompts = [
        "### Instruction:\nWhat is the braille pattern for the letter 'h'?\n\n### Response:",
        "### Instruction:\nWhat file type does this braille header indicate: ⡐⡋⠃⠄\n\n### Response:",
        "### Instruction:\nDecode this braille: ⠓⠑⠇⠇⠕\n\n### Response:",
        "### Instruction:\nHow can braille represent any binary file?\n\n### Response:",
    ]
    
    print("Testing prompts:\n")
    for prompt in test_prompts:
        print("-" * 40)
        print(f"Prompt: {prompt.split('Instruction:')[1].split('Response:')[0].strip()[:50]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"Response: {response[:200]}...")
        print()


if __name__ == "__main__":
    test_model()
