# Braille-Native Cognition: Quick Start Guide

This guide shows you exactly how to run the braille cognition evolution system on your machine.

## Prerequisites

You need Python 3.8+ and the following packages:

```bash
pip install numpy pandas librosa soundfile pillow
```

If you want to use the provided virtual environment setup (recommended), you can use the same approach we used in the sandbox.

## Setup (One-Time)

### Option 1: Using a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv braille_env

# Activate it
source braille_env/bin/activate  # On Windows: braille_env\Scripts\activate

# Install dependencies
pip install numpy pandas librosa soundfile pillow
```

### Option 2: System-Wide Installation

```bash
pip install numpy pandas librosa soundfile pillow
```

## Running the System

### Quick Start: Run the Complete Demonstration

This is the easiest way to see the entire evolution in action:

```bash
# Activate the virtual environment (if you created one)
source braille_env/bin/activate

# Run the demonstration
python3 evolution_demonstration.py
```

This will output a comprehensive walkthrough of all four stages, showing:
- Stage 1: Braille as a carrier format
- Stage 2: Braille-native operations
- Stage 3: Braille-constrained reasoning engine
- Stage 4: Progressive training objectives
- Synthesis and key insights

### Running Individual Stages

If you want to run each stage independently:

#### Stage 1: Braille Converter (Carrier Format)

```bash
python3 braille_converter.py
```

**What it does:**
- Converts text to braille
- Creates a multimodal dataset
- Demonstrates braille as a unified encoding format

**Output:**
- Text-to-braille conversions
- Dataset statistics
- Sample braille corpus

#### Stage 2: Braille-Native Operations

```bash
python3 braille_native_ops.py
```

**What it does:**
- Demonstrates Hamming distance and similarity metrics
- Shows morphological operations (dilation, erosion)
- Performs sequence-level operations
- Computes braille-native metrics

**Output:**
- Hamming distance calculations
- Nearest neighbor searches
- Morphological transformations
- Sequence similarity scores

#### Stage 3: Braille-Constrained Reasoning Engine

```bash
python3 braille_reasoning_engine.py
```

**What it does:**
- Initializes a reasoning engine that operates entirely in braille space
- Stores patterns in braille-native memory
- Performs braille-native attention
- Demonstrates pattern transformations
- Shows that no decode step occurs during reasoning

**Output:**
- Structural analysis of patterns
- Memory retrieval results
- Morphological transformations
- Attention fusion outputs
- Engine statistics

#### Stage 4: Progressive Training Objectives

```bash
python3 braille_training_objectives.py
```

**What it does:**
- Generates token-level training examples
- Generates pattern-level training examples
- Generates semantic-level training examples
- Defines a progressive curriculum
- Validates objective integrity

**Output:**
- Training examples at each level
- Curriculum schedule
- Objective validation reports
- Training statistics

## Understanding the Output

### Stage 1 Output Example

```
Input text: Hello Braille
Braille: ⠓⠑⠇⠇⠕ ⠠⠃⠗⠁⠊⠇⠇⠑
Braille length: 13 characters
Status: Braille is a passive carrier; no reasoning occurs in braille space
```

### Stage 2 Output Example

```
Character 1: 00000000
Character 2: 11111111
Hamming distance: 8
Hamming similarity: 0.000
Status: Similarity computed entirely in braille space
```

### Stage 3 Output Example

```
Query: ⠓⠑⠇⠇⠕
Operation: structural_analysis
  popcount: 14
  density: 0.35
Operation: memory_retrieval
  similar_patterns: [('⠓⠑⠇⠇⠕', 1.0), ('⠓⠑⠇⠏⠕', 0.9)]
Status: All reasoning occurs in braille space
```

### Stage 4 Output Example

```
Training curriculum (1000 total steps):
  token_level           500 steps ( 50.0%)
  pattern_level         300 steps ( 30.0%)
  semantic_level        200 steps ( 20.0%)

TOKEN:
  Braille-Native: False
  Reasoning: Token-level loss can be computed in braille space, but model internals can still use float space.

PATTERN:
  Braille-Native: True
  Reasoning: Pattern-level loss is defined only on braille-native properties. Model cannot use float space without violating loss.

SEMANTIC:
  Braille-Native: True
  Reasoning: Semantic-level loss requires reasoning about meaning in braille space.
```

## File Structure

```
.
├── QUICKSTART.md                           # This file
├── BRAILLE_COGNITION_EVOLUTION.md          # Main documentation
├── EVOLUTION_SUMMARY.txt                   # Plain-text summary
├── BRAILLE_EVOLUTION_ARCHITECTURE.md       # Design document
├── evolution_demonstration.py              # Master demonstration script
├── braille_converter.py                    # Stage 1 implementation
├── braille_native_ops.py                   # Stage 2 implementation
├── braille_reasoning_engine.py             # Stage 3 implementation
├── braille_training_objectives.py          # Stage 4 implementation
└── braille_training/                       # Training data and config
    ├── braille_corpus.txt                  # Sample training corpus
    ├── Modelfile                           # Ollama model configuration
    ├── inference_config.json               # Inference settings
    └── prompt_templates.json               # Prompt templates
```

## Customization

### Modifying Stage 1: Text Encoding

Edit `braille_converter.py` to add custom text samples:

```python
converter = BrailleConverter()
text = "Your custom text here"
braille = converter.text_to_braille(text)
print(f"Input: {text}")
print(f"Braille: {braille}")
```

### Modifying Stage 3: Memory Patterns

Edit `braille_reasoning_engine.py` to add custom patterns:

```python
reasoner = BrailleReasoner()
reasoner.memory.store("⠓⠑⠇⠇⠕")  # Store a pattern
reasoner.memory.store("⠠⠃⠗⠁⠊⠇⠇⠑")  # Store another
```

### Modifying Stage 4: Training Curriculum

Edit `braille_training_objectives.py` to adjust the curriculum:

```python
schedule = ProgressiveTrainingSchedule(sequences)
curriculum = schedule.get_curriculum(total_steps=2000)  # Increase total steps
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, make sure you've installed the dependencies:

```bash
pip install numpy pandas librosa soundfile pillow
```

### Virtual Environment Issues

If the virtual environment isn't activating, try:

```bash
# On Linux/Mac
source braille_env/bin/activate

# On Windows
braille_env\Scripts\activate

# Or use Python directly
python3 -m venv braille_env
```

### Unicode/Braille Display Issues

If braille characters aren't displaying correctly, ensure your terminal supports UTF-8:

```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

## Next Steps

After running the demonstrations, you can:

1. **Extend Stage 1**: Add image, audio, or video encoding to the multimodal converter.
2. **Enhance Stage 2**: Implement additional morphological operators or custom metrics.
3. **Build Stage 3**: Create a full neural network that operates in braille space.
4. **Implement Stage 4**: Train an actual model using the progressive curriculum.

## Questions?

Refer to the main documentation (`BRAILLE_COGNITION_EVOLUTION.md`) for detailed explanations of each stage and the underlying architecture.
