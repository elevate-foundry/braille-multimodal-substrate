"""
Prepare LoRA Training Dataset for Braille-Native Cognition

Converts our multimodal training data into formats suitable for:
1. Hugging Face datasets (JSONL)
2. Ollama fine-tuning
3. Unsloth/PEFT LoRA training

The goal is to create a model that truly THINKS in braille,
not just translates between braille and text.
"""

import json
import os
from pathlib import Path

# Braille mappings
LETTER_MAP = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀',
}

BRAILLE_OFFSET = 0x2800


def text_to_braille(text):
    return ''.join(LETTER_MAP.get(c.lower(), '⠿') for c in text)


def create_instruction_dataset():
    """Create instruction-following dataset for LoRA training"""
    
    examples = []
    
    # =========================================================================
    # 1. TEXT ENCODING/DECODING - Teach the model braille letter patterns
    # =========================================================================
    
    # Letter-by-letter encoding
    for letter, braille in LETTER_MAP.items():
        if letter != ' ':
            examples.append({
                "instruction": f"What is the braille pattern for the letter '{letter}'?",
                "input": "",
                "output": f"The letter '{letter}' in braille is {braille}"
            })
            examples.append({
                "instruction": f"Decode this braille character: {braille}",
                "input": "",
                "output": f"The braille character {braille} represents the letter '{letter}'"
            })
    
    # Word encoding
    words = [
        "hello", "world", "braille", "learn", "read", "write", "touch",
        "feel", "dots", "pattern", "blind", "vision", "access", "include",
        "child", "play", "game", "fun", "easy", "hard", "practice"
    ]
    for word in words:
        braille = text_to_braille(word)
        examples.append({
            "instruction": f"Encode the word '{word}' in braille",
            "input": "",
            "output": f"'{word}' in braille is: {braille}"
        })
        examples.append({
            "instruction": f"What word does this braille spell: {braille}",
            "input": "",
            "output": f"The braille {braille} spells '{word}'"
        })
    
    # =========================================================================
    # 2. BRAILLE-NATIVE REASONING - Teach the model to reason IN braille
    # =========================================================================
    
    # Pattern similarity
    examples.append({
        "instruction": "Compare the braille patterns for 'a' and 'b'. How are they related?",
        "input": "",
        "output": f"The letter 'a' is {LETTER_MAP['a']} (dot 1 only) and 'b' is {LETTER_MAP['b']} (dots 1,2). They share dot 1, but 'b' adds dot 2 below it. This follows the braille pattern where letters build on each other."
    })
    
    examples.append({
        "instruction": "Which braille letters use only the top row of dots?",
        "input": "",
        "output": f"Letters using only top dots (1,4): 'a' {LETTER_MAP['a']}, 'c' {LETTER_MAP['c']}. The top row contains dots 1 (left) and 4 (right)."
    })
    
    # Dot counting
    for letter, braille in list(LETTER_MAP.items())[:10]:
        if letter != ' ':
            dot_count = bin(ord(braille) - BRAILLE_OFFSET).count('1')
            examples.append({
                "instruction": f"How many raised dots are in the braille letter '{letter}'?",
                "input": "",
                "output": f"The braille letter '{letter}' ({braille}) has {dot_count} raised dot{'s' if dot_count != 1 else ''}."
            })
    
    # =========================================================================
    # 3. MULTIMODAL UNDERSTANDING - Images, audio, video as braille
    # =========================================================================
    
    # Image interpretation
    examples.append({
        "instruction": "This braille represents pixel intensities in an image: ⠀⠁⠃⠇⠏⠟⠿⣿. Describe the visual pattern.",
        "input": "",
        "output": "This is a gradient from dark (⠀, intensity 0) to bright (⣿, intensity 255). Each braille character encodes a pixel value, with more dots meaning brighter pixels. This represents a smooth left-to-right brightness transition."
    })
    
    examples.append({
        "instruction": "What kind of image would produce this braille pattern: ⣿⣿⣿⣿⠀⠀⠀⠀⣿⣿⣿⣿",
        "input": "",
        "output": "This pattern shows bright (⣿) regions on the left and right with a dark (⠀) gap in the middle. This could represent two bright objects separated by darkness, like two lights in a dark room or a split image."
    })
    
    # Audio interpretation
    examples.append({
        "instruction": "This braille represents audio MFCC features: ⡀⡀⣿⣿⡀⡀⣿⣿. What does it sound like?",
        "input": "",
        "output": "The alternating pattern of low (⡀) and high (⣿) values suggests a rhythmic sound with regular beats. The 4-character cycle indicates a tempo. This could be a drum beat or rhythmic music."
    })
    
    # Video/motion
    examples.append({
        "instruction": "These braille frames show motion. Frame 1: ⣿⠀⠀⠀, Frame 2: ⠀⣿⠀⠀, Frame 3: ⠀⠀⣿⠀, Frame 4: ⠀⠀⠀⣿. Describe the motion.",
        "input": "",
        "output": "A bright object (⣿) is moving from left to right across 4 frames. The motion is linear and constant-speed. This represents horizontal movement like a ball rolling or a light moving across a screen."
    })
    
    # =========================================================================
    # 4. SEMANTIC ANALYSIS - Understanding braille at a deeper level
    # =========================================================================
    
    examples.append({
        "instruction": "What is the Hamming distance between ⠁ and ⠃?",
        "input": "",
        "output": "⠁ (letter 'a') has pattern 00000001 and ⠃ (letter 'b') has pattern 00000011. They differ by 1 bit (dot 2), so the Hamming distance is 1. This means they are very similar patterns."
    })
    
    examples.append({
        "instruction": "Explain why braille is ideal for encoding multimodal data.",
        "input": "",
        "output": "8-dot braille provides 256 unique patterns (U+2800-U+28FF), mapping perfectly to byte values (0-255). This makes it ideal for: 1) Text (letter mappings), 2) Images (pixel intensities), 3) Audio (spectral features), 4) Video (frame sequences). The tactile nature also enables haptic output."
    })
    
    # =========================================================================
    # 5. TUTORING SCENARIOS - For BrailleBuddy integration
    # =========================================================================
    
    examples.append({
        "instruction": "A child is learning the letter 'h'. Give a memorable hint about its braille pattern.",
        "input": "",
        "output": f"The letter 'h' in braille is {LETTER_MAP['h']}. Think of it like a little house! It has dots 1, 2, and 5 - two dots stacked on the left side (like a wall) and one dot on the right (like a window). H is for House!"
    })
    
    examples.append({
        "instruction": "A student got 3 correct and 7 wrong while learning braille. Give encouraging feedback.",
        "input": "",
        "output": "You're making progress! 3 correct answers means you're starting to recognize patterns. Braille takes practice - even Louis Braille took time to perfect his system. Focus on the letters you got right and build from there. Every expert was once a beginner!"
    })
    
    examples.append({
        "instruction": "Generate an easy braille practice exercise for a beginner.",
        "input": "",
        "output": f"Try this: What letter is {LETTER_MAP['e']}? Hint: It's a vowel and it's the most common letter in English! (Answer: 'e')"
    })
    
    return examples


def save_datasets(examples, output_dir):
    """Save datasets in multiple formats"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. JSONL format (Hugging Face compatible)
    jsonl_path = os.path.join(output_dir, "braille_lora_train.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Saved {len(examples)} examples to {jsonl_path}")
    
    # 2. Alpaca format (for many LoRA trainers)
    alpaca_path = os.path.join(output_dir, "braille_alpaca.json")
    with open(alpaca_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved Alpaca format to {alpaca_path}")
    
    # 3. Chat format (for chat-tuned models)
    chat_examples = []
    for ex in examples:
        chat_examples.append({
            "messages": [
                {"role": "user", "content": ex["instruction"] + (" " + ex["input"] if ex["input"] else "")},
                {"role": "assistant", "content": ex["output"]}
            ]
        })
    
    chat_path = os.path.join(output_dir, "braille_chat.jsonl")
    with open(chat_path, 'w', encoding='utf-8') as f:
        for ex in chat_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Saved chat format to {chat_path}")
    
    # 4. Statistics
    stats = {
        "total_examples": len(examples),
        "avg_instruction_length": sum(len(ex["instruction"]) for ex in examples) / len(examples),
        "avg_output_length": sum(len(ex["output"]) for ex in examples) / len(examples),
        "categories": {
            "encoding_decoding": sum(1 for ex in examples if "encode" in ex["instruction"].lower() or "decode" in ex["instruction"].lower()),
            "pattern_analysis": sum(1 for ex in examples if "pattern" in ex["instruction"].lower() or "dot" in ex["instruction"].lower()),
            "multimodal": sum(1 for ex in examples if "image" in ex["instruction"].lower() or "audio" in ex["instruction"].lower() or "video" in ex["instruction"].lower()),
            "tutoring": sum(1 for ex in examples if "learn" in ex["instruction"].lower() or "hint" in ex["instruction"].lower() or "practice" in ex["instruction"].lower())
        }
    }
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics: {stats}")
    
    return stats


def main():
    print("=" * 60)
    print("Preparing LoRA Training Dataset for Braille-Native Cognition")
    print("=" * 60)
    
    # Generate examples
    examples = create_instruction_dataset()
    
    # Save in multiple formats
    output_dir = Path(__file__).parent
    stats = save_datasets(examples, output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset ready for LoRA fine-tuning!")
    print("=" * 60)
    print(f"\nTotal examples: {stats['total_examples']}")
    print("\nTo fine-tune with Unsloth:")
    print("  pip install unsloth")
    print("  python train_lora.py")
    print("\nTo push to Hugging Face:")
    print("  huggingface-cli login")
    print("  python push_to_hub.py")


if __name__ == "__main__":
    main()
