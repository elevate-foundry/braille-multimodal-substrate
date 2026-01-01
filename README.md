# Braille Multimodal Substrate

A system for training Ollama models to natively understand 8-dot braille as a universal encoding for **text, images, audio, and video**.

## Overview

This project demonstrates **Braille-Native Cognition** — using 8-dot braille (Unicode U+2800-U+28FF, 256 patterns) as a multimodal substrate where AI can reason directly in braille space without decoding to traditional representations.

### Key Innovation

Instead of treating braille as just a text encoding, this system uses braille as a **universal byte-level representation** for:

| Modality | Encoding Method |
|----------|-----------------|
| **Text** | Standard letter mapping (a=⠁, b=⠃, etc.) |
| **Images** | Pixel intensity (0-255 → ⠀ to ⣿) |
| **Audio** | MFCC spectral features as braille patterns |
| **Video** | Temporal sequences of braille-encoded frames |

## Quick Start

### Prerequisites

```bash
# Install Ollama
brew install ollama

# Install Python dependencies
python3 -m venv braille_env
source braille_env/bin/activate
pip install opencv-python numpy librosa soundfile pillow
```

### Create the Multimodal Braille Model

```bash
# Generate training data
python3 multimodal_braille_training.py

# Create the Ollama model
ollama create multimodal-braille -f braille_training/Modelfile.multimodal

# Test it
ollama run multimodal-braille
```

### Example Queries

```bash
# Text decoding
echo "Decode this braille: ⠓⠑⠇⠇⠕ ⠺⠕⠗⠇⠙" | ollama run multimodal-braille

# Image interpretation
echo "This braille is an 8x8 image: ⠀⠁⠃⠇⠏⠟⠿⣿ - describe the pattern" | ollama run multimodal-braille

# Audio interpretation
echo "This braille represents MFCC features: ⡀⡀⣿⣿⡀⡀⣿⣿ - what sound?" | ollama run multimodal-braille

# Video/motion detection
echo "Frame 1: ⠀⣿⠀⠀ Frame 2: ⠀⠀⣿⠀ Frame 3: ⠀⠀⠀⣿ - describe motion" | ollama run multimodal-braille
```

## Project Structure

```
├── braille_converter.py           # Core multimodal-to-braille conversion
├── multimodal_braille_training.py # Training data generator (800 examples)
├── braille_native_ops.py          # Braille-native operations (Hamming, morphology)
├── braille_reasoning_engine.py    # Reasoning engine in braille space
├── braille_semantic_graph.py      # Semantic graph for braille patterns
├── braille_training_objectives.py # Progressive training curriculum
├── evolution_demonstration.py     # Full system demonstration
├── test_braille_system.py         # Comprehensive test suite
├── braille_ollama_setup.py        # Ollama integration
└── braille_training/              # Generated training data
    ├── Modelfile.multimodal       # Ollama model configuration
    ├── multimodal_training_data.json
    └── multimodal_corpus.txt
```

## The Evolution of Braille-Native Cognition

### Stage 1: Carrier Format
Braille as a universal byte-level interchange (like a ZIP file)

### Stage 2: Native Operations
Braille-native operations (Hamming distance, morphological transforms)

### Stage 3: Constrained Reasoning
Reasoning engine operates entirely in braille space — no decode step

### Stage 4: Progressive Training
Token-level → Pattern-level → Semantic-level curriculum

## Braille Encoding Reference

### 8-Dot Braille Cell
```
1 4
2 5
3 6
7 8
```

- **Unicode Range**: U+2800 to U+28FF (256 patterns)
- **Empty cell**: ⠀ (U+2800)
- **Full cell**: ⣿ (U+28FF)

### Letter Mapping
```
a=⠁ b=⠃ c=⠉ d=⠙ e=⠑ f=⠋ g=⠛ h=⠓ i=⠊ j=⠚
k=⠅ l=⠇ m=⠍ n=⠝ o=⠕ p=⠏ q=⠟ r=⠗ s=⠎ t=⠞
u=⠥ v=⠧ w=⠺ x=⠭ y=⠽ z=⠵ space=⠀
```

## Documentation

- [Quick Start Guide](Braille-Native%20Cognition:%20Quick%20Start%20Guide.md)
- [Architectural Design](Braille%20Semantic%20Graph:%20Architectural%20Design.md)
- [Implementation Details](Braille-Native%20Substrate:%20Implementation-Level%20Clarifications.md)
- [Evolution of Braille-Native Cognition](The%20Evolution%20of%20Braille-Native%20Cognition.md)

## License

MIT

## Related Projects

- [native-braille-ai](https://github.com/elevate-foundry/native-braille-ai) - Native braille processing
- [sal-llm](https://github.com/elevate-foundry/sal-llm) - SAL LLM with braille consciousness
- [sal-voice](https://github.com/elevate-foundry/sal-voice) - Multimodal interface with braille core
