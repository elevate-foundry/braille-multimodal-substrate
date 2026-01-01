"""
Generate training data for the braille-based Ollama model
Creates synthetic multimodal samples and their braille encodings
"""

import json
import numpy as np
from pathlib import Path
from braille_converter import BrailleConverter, MultimodalBrailleDataset, BRAILLE_CHARS
import random

def generate_synthetic_text_samples(n_samples: int = 100) -> list:
    """Generate synthetic text samples"""
    samples = []
    
    # Common phrases and sentences
    phrases = [
        "The quick brown fox jumps over the lazy dog",
        "Braille is a tactile writing system",
        "Eight dot braille extends the standard six dot system",
        "Multimodal learning combines different data types",
        "Artificial intelligence processes information",
        "Neural networks learn from data patterns",
        "Compression reduces information size",
        "Semantic meaning persists across representations",
        "Universal symbols emerge from compression",
        "Braille encodes information tactilely",
        "Vision transforms to touch through braille",
        "Audio becomes visual through spectrograms",
        "Video sequences compress to key frames",
        "Information theory guides compression",
        "Entropy measures information content",
    ]
    
    for i in range(n_samples):
        phrase = random.choice(phrases)
        # Add variations
        if i % 3 == 0:
            phrase = phrase.upper()
        elif i % 3 == 1:
            phrase = phrase.lower()
        
        samples.append({
            "type": "text",
            "content": phrase,
            "length": len(phrase),
            "id": f"text_{i:04d}"
        })
    
    return samples

def generate_synthetic_image_samples(n_samples: int = 50) -> list:
    """Generate synthetic image data (as numpy arrays)"""
    samples = []
    
    for i in range(n_samples):
        # Create synthetic patterns
        size = 32
        
        if i % 5 == 0:
            # Gradient pattern
            img = np.linspace(0, 255, size * size).reshape(size, size).astype(np.uint8)
        elif i % 5 == 1:
            # Checkerboard pattern
            img = np.zeros((size, size), dtype=np.uint8)
            img[::2, ::2] = 255
            img[1::2, 1::2] = 255
        elif i % 5 == 2:
            # Random noise
            img = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        elif i % 5 == 3:
            # Circular pattern
            y, x = np.ogrid[:size, :size]
            mask = (x - size//2)**2 + (y - size//2)**2 <= (size//3)**2
            img = np.zeros((size, size), dtype=np.uint8)
            img[mask] = 255
        else:
            # Sine wave pattern
            x = np.linspace(0, 4*np.pi, size)
            y = np.linspace(0, 4*np.pi, size)
            X, Y = np.meshgrid(x, y)
            img = (np.sin(X) * np.cos(Y) * 127 + 128).astype(np.uint8)
        
        samples.append({
            "type": "image",
            "pattern": ["gradient", "checkerboard", "noise", "circle", "sine"][i % 5],
            "shape": img.shape,
            "data": img.tolist(),
            "id": f"image_{i:04d}"
        })
    
    return samples

def generate_synthetic_audio_samples(n_samples: int = 50) -> list:
    """Generate synthetic audio features (MFCC-like)"""
    samples = []
    
    for i in range(n_samples):
        # Create synthetic MFCC-like features
        n_mfcc = 13
        n_frames = 32
        
        if i % 4 == 0:
            # Pure sine wave features
            freq = 440 + i * 10
            t = np.linspace(0, 1, n_frames)
            mfcc = np.sin(2 * np.pi * freq * t / 1000) * 50 + 50
            mfcc = np.tile(mfcc, (n_mfcc, 1))
        elif i % 4 == 1:
            # Noise-like features
            mfcc = np.random.randn(n_mfcc, n_frames) * 30 + 50
        elif i % 4 == 2:
            # Chirp-like features
            t = np.linspace(0, 1, n_frames)
            freq = 440 + 200 * t
            mfcc = np.sin(2 * np.pi * freq * t / 1000) * 50 + 50
            mfcc = np.tile(mfcc, (n_mfcc, 1))
        else:
            # Speech-like features
            mfcc = np.random.randn(n_mfcc, n_frames) * 20 + 50
            mfcc[0] = np.linspace(50, 100, n_frames)  # Energy contour
        
        # Clip to valid range
        mfcc = np.clip(mfcc, 0, 255).astype(np.uint8)
        
        samples.append({
            "type": "audio",
            "pattern": ["sine", "noise", "chirp", "speech"][i % 4],
            "shape": mfcc.shape,
            "data": mfcc.tolist(),
            "id": f"audio_{i:04d}"
        })
    
    return samples

def generate_training_corpus(output_dir: str = "/home/ubuntu/braille_training") -> str:
    """Generate complete training corpus"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    converter = BrailleConverter()
    
    # Generate samples
    print("Generating synthetic training samples...")
    text_samples = generate_synthetic_text_samples(100)
    image_samples = generate_synthetic_image_samples(50)
    audio_samples = generate_synthetic_audio_samples(50)
    
    # Convert to braille
    print("Converting to braille encoding...")
    corpus_lines = []
    
    # Text samples
    for sample in text_samples:
        braille = converter.text_to_braille(sample["content"])
        corpus_lines.append(f"[TEXT] {sample['id']}: {braille}")
    
    # Image samples
    for sample in image_samples:
        img_array = np.array(sample["data"], dtype=np.uint8)
        braille, _ = converter.image_to_braille(img_array)
        # Compress representation for corpus
        braille_compressed = braille.replace("\n", " ")[:100]  # First 100 chars
        corpus_lines.append(f"[IMAGE] {sample['id']}: {braille_compressed}")
    
    # Audio samples
    for sample in audio_samples:
        audio_array = np.array(sample["data"], dtype=np.uint8)
        # Create synthetic audio tuple
        audio_data = (np.random.randn(16000), 16000)  # 1 second at 16kHz
        braille, _ = converter.audio_to_braille(audio_data)
        braille_compressed = braille.replace("\n", " ")[:100]
        corpus_lines.append(f"[AUDIO] {sample['id']}: {braille_compressed}")
    
    # Create corpus file
    corpus_text = "\n".join(corpus_lines)
    corpus_path = output_path / "braille_corpus.txt"
    with open(corpus_path, 'w') as f:
        f.write(corpus_text)
    
    print(f"Training corpus saved to {corpus_path}")
    print(f"Total samples: {len(corpus_lines)}")
    print(f"Corpus size: {len(corpus_text)} characters")
    
    # Save metadata
    metadata = {
        "total_samples": len(corpus_lines),
        "text_samples": len(text_samples),
        "image_samples": len(image_samples),
        "audio_samples": len(audio_samples),
        "corpus_size": len(corpus_text),
        "braille_range": "U+2800 to U+28FF",
        "encoding": "8-dot braille"
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(corpus_path)

def create_ollama_modelfile(corpus_path: str, output_dir: str = "/home/ubuntu/braille_training") -> str:
    """Create Ollama Modelfile for training"""
    output_path = Path(output_dir)
    
    modelfile_content = f"""FROM mistral

# Set parameters for braille-focused model
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9

# System prompt for braille processing
SYSTEM You are a specialized AI model trained to process and understand 8-dot braille encoding across multimodal data (text, image, audio, video). Your responses should demonstrate understanding of braille patterns and their semantic meaning. When given braille-encoded input, decode and process it meaningfully.

# Training context
# This model is trained on a corpus of braille-encoded multimodal data
# Braille character range: U+2800 to U+28FF (256 unique patterns)
# Each pattern represents a specific encoding of visual, auditory, or textual information
"""
    
    modelfile_path = output_path / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"Modelfile created at {modelfile_path}")
    return str(modelfile_path)

def create_prompt_template(output_dir: str = "/home/ubuntu/braille_training") -> str:
    """Create prompt template for braille-aware responses"""
    output_path = Path(output_dir)
    
    prompt_template = """# Braille-Aware Prompt Template

## System Instructions
You are processing multimodal data encoded in 8-dot braille. The braille characters (U+2800 to U+28FF) represent:
- Text: ASCII character mappings
- Images: Pixel intensity patterns (grayscale 0-255)
- Audio: MFCC feature representations
- Video: Sequential frame encodings

## Task Examples

### Text Processing
Input: ⡈⡥⡬⡬⡯ (braille-encoded "Hello")
Output: Decode and understand the semantic meaning

### Image Processing
Input: [Grid of braille characters representing pixel intensities]
Output: Describe the visual pattern and content

### Audio Processing
Input: [Sequence of braille characters representing MFCC features]
Output: Analyze the acoustic characteristics

### Video Processing
Input: [Multiple frames of braille-encoded image data]
Output: Describe the temporal sequence and motion

## Response Format
When responding to braille-encoded input:
1. Decode the braille representation
2. Identify the modality (text, image, audio, video)
3. Extract semantic meaning
4. Provide analysis in both braille and natural language

## Braille Encoding Rules
- Each braille character is 8 bits (0-255)
- Patterns follow Unicode standard U+2800 to U+28FF
- Multiple characters can be combined for complex representations
- Whitespace and newlines separate different data segments
"""
    
    template_path = output_path / "prompt_template.md"
    with open(template_path, 'w') as f:
        f.write(prompt_template)
    
    print(f"Prompt template created at {template_path}")
    return str(template_path)

if __name__ == "__main__":
    print("=" * 60)
    print("Braille Training Data Generator")
    print("=" * 60)
    
    # Generate training corpus
    corpus_path = generate_training_corpus()
    
    # Create Ollama Modelfile
    modelfile_path = create_ollama_modelfile(corpus_path)
    
    # Create prompt template
    template_path = create_prompt_template()
    
    print("\n" + "=" * 60)
    print("Training data generation complete!")
    print("=" * 60)
    print(f"Corpus: {corpus_path}")
    print(f"Modelfile: {modelfile_path}")
    print(f"Template: {template_path}")
