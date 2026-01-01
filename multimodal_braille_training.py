"""
Multimodal Braille Training Data Generator
Creates training examples that teach the model to interpret braille as:
- Text (standard letter encoding)
- Images (pixel intensity patterns)
- Audio (MFCC spectral features)
- Video (temporal frame sequences)
"""

import json
import numpy as np
from pathlib import Path
from braille_converter import BrailleConverter, BRAILLE_CHARS, BRAILLE_OFFSET

class MultimodalBrailleTrainer:
    """Generate training data for truly multimodal braille understanding"""
    
    def __init__(self, output_dir: str = None):
        self.converter = BrailleConverter()
        if output_dir is None:
            output_dir = Path(__file__).parent / "braille_training"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard braille letter mapping (Grade 1 Braille)
        self.letter_map = {
            'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
            'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
            'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
            'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀',
            '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑',
            '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
        }
        self.reverse_letter_map = {v: k for k, v in self.letter_map.items()}
    
    def text_to_standard_braille(self, text: str) -> str:
        """Convert text using standard braille letter mapping"""
        return ''.join(self.letter_map.get(c.lower(), '⠿') for c in text)
    
    def generate_text_training_examples(self, n_samples: int = 200) -> list:
        """Generate text encoding/decoding training examples"""
        examples = []
        
        phrases = [
            "hello world", "braille is tactile", "multimodal learning",
            "artificial intelligence", "neural networks", "deep learning",
            "computer vision", "natural language", "speech recognition",
            "image classification", "object detection", "semantic segmentation",
            "the quick brown fox", "jumps over lazy dog", "programming code",
            "data science", "machine learning", "pattern recognition",
        ]
        
        for i in range(n_samples):
            phrase = phrases[i % len(phrases)]
            braille = self.text_to_standard_braille(phrase)
            
            # Training example: braille -> text decoding
            examples.append({
                "type": "text_decode",
                "instruction": f"Decode this braille text to English letters: {braille}",
                "input": braille,
                "output": phrase,
                "explanation": f"Each braille character maps to a letter: " + 
                              ", ".join(f"{self.letter_map.get(c, '?')}={c}" for c in phrase[:5])
            })
            
            # Training example: text -> braille encoding
            examples.append({
                "type": "text_encode",
                "instruction": f"Encode this text in braille: {phrase}",
                "input": phrase,
                "output": braille,
                "explanation": "Standard Grade 1 braille letter encoding"
            })
        
        return examples
    
    def generate_image_training_examples(self, n_samples: int = 100) -> list:
        """Generate image pattern recognition training examples"""
        examples = []
        
        patterns = {
            "gradient": lambda s: np.linspace(0, 255, s*s).reshape(s, s).astype(np.uint8),
            "checkerboard": lambda s: self._make_checkerboard(s),
            "circle": lambda s: self._make_circle(s),
            "horizontal_lines": lambda s: self._make_hlines(s),
            "vertical_lines": lambda s: self._make_vlines(s),
            "diagonal": lambda s: self._make_diagonal(s),
            "noise": lambda s: np.random.randint(0, 256, (s, s), dtype=np.uint8),
            "solid_dark": lambda s: np.zeros((s, s), dtype=np.uint8),
            "solid_bright": lambda s: np.full((s, s), 255, dtype=np.uint8),
            "center_spot": lambda s: self._make_center_spot(s),
        }
        
        for i in range(n_samples):
            pattern_name = list(patterns.keys())[i % len(patterns)]
            size = 8  # Small size for training examples
            img = patterns[pattern_name](size)
            
            # Convert to braille
            braille_lines = []
            for row in img:
                braille_row = ''.join(BRAILLE_CHARS[int(v)] for v in row)
                braille_lines.append(braille_row)
            braille_img = '\n'.join(braille_lines)
            
            # Describe the pattern characteristics
            avg_intensity = np.mean(img)
            intensity_desc = "dark" if avg_intensity < 85 else "medium" if avg_intensity < 170 else "bright"
            
            examples.append({
                "type": "image_interpret",
                "instruction": f"This braille represents an 8x8 image where each character encodes pixel intensity (0-255). Describe the visual pattern:\n{braille_img}",
                "input": braille_img,
                "output": f"This is a {pattern_name.replace('_', ' ')} pattern. The average intensity is {intensity_desc} ({avg_intensity:.0f}/255).",
                "pattern": pattern_name,
                "explanation": "Braille characters U+2800-U+28FF map to pixel intensities 0-255"
            })
            
            # Reverse: describe pattern -> identify braille characteristics
            examples.append({
                "type": "image_generate",
                "instruction": f"What braille pattern would represent a {pattern_name.replace('_', ' ')} image?",
                "input": pattern_name,
                "output": f"A {pattern_name.replace('_', ' ')} pattern would use braille characters that {'increase from ⠀ to ⣿' if pattern_name == 'gradient' else 'alternate between ⠀ and ⣿' if 'checker' in pattern_name else 'vary based on the pattern'}.",
                "explanation": "Image-to-braille encoding uses intensity mapping"
            })
        
        return examples
    
    def generate_audio_training_examples(self, n_samples: int = 100) -> list:
        """Generate audio feature interpretation training examples"""
        examples = []
        
        audio_types = {
            "sine_wave": {"freq": "single frequency", "pattern": "smooth, periodic", "braille_char": "uniform repeating"},
            "noise": {"freq": "all frequencies", "pattern": "random, chaotic", "braille_char": "highly varied"},
            "chirp": {"freq": "increasing frequency", "pattern": "accelerating oscillation", "braille_char": "progressively changing"},
            "speech": {"freq": "formant frequencies", "pattern": "complex, modulated", "braille_char": "structured variation"},
            "music": {"freq": "harmonic series", "pattern": "rhythmic, tonal", "braille_char": "periodic with variation"},
            "silence": {"freq": "no frequency", "pattern": "flat, zero energy", "braille_char": "mostly ⠀ (empty)"},
        }
        
        for i in range(n_samples):
            audio_type = list(audio_types.keys())[i % len(audio_types)]
            info = audio_types[audio_type]
            
            # Generate synthetic MFCC-like braille pattern
            if audio_type == "sine_wave":
                mfcc = np.sin(np.linspace(0, 4*np.pi, 32)) * 50 + 128
            elif audio_type == "noise":
                mfcc = np.random.randint(0, 256, 32)
            elif audio_type == "chirp":
                mfcc = np.linspace(50, 200, 32)
            elif audio_type == "silence":
                mfcc = np.zeros(32)
            else:
                mfcc = np.random.randint(50, 200, 32)
            
            braille_audio = ''.join(BRAILLE_CHARS[int(v) % 256] for v in mfcc)
            
            examples.append({
                "type": "audio_interpret",
                "instruction": f"This braille represents MFCC audio features. What type of sound does this pattern suggest?\n{braille_audio}",
                "input": braille_audio,
                "output": f"This pattern suggests {audio_type.replace('_', ' ')}. Characteristics: {info['pattern']}. The braille shows {info['braille_char']} patterns, indicating {info['freq']} content.",
                "audio_type": audio_type,
                "explanation": "MFCC features capture spectral characteristics of audio"
            })
        
        return examples
    
    def generate_video_training_examples(self, n_samples: int = 50) -> list:
        """Generate video/temporal sequence training examples"""
        examples = []
        
        motion_types = {
            "static": "no change between frames",
            "fade_in": "progressive brightening",
            "fade_out": "progressive darkening", 
            "left_to_right": "movement from left to right",
            "expanding": "growing from center outward",
            "flashing": "alternating bright and dark",
        }
        
        for i in range(n_samples):
            motion_type = list(motion_types.keys())[i % len(motion_types)]
            n_frames = 4
            
            # Generate frame sequence
            frames_braille = []
            for f in range(n_frames):
                if motion_type == "static":
                    frame = np.full((4, 4), 128, dtype=np.uint8)
                elif motion_type == "fade_in":
                    frame = np.full((4, 4), int(255 * f / n_frames), dtype=np.uint8)
                elif motion_type == "fade_out":
                    frame = np.full((4, 4), int(255 * (1 - f / n_frames)), dtype=np.uint8)
                elif motion_type == "left_to_right":
                    frame = np.zeros((4, 4), dtype=np.uint8)
                    frame[:, f] = 255
                elif motion_type == "expanding":
                    frame = np.zeros((4, 4), dtype=np.uint8)
                    frame[1:3, 1:3] = int(255 * f / n_frames)
                else:  # flashing
                    frame = np.full((4, 4), 255 if f % 2 == 0 else 0, dtype=np.uint8)
                
                braille_frame = ''.join(BRAILLE_CHARS[int(v)] for row in frame for v in row)
                frames_braille.append(f"Frame {f+1}: {braille_frame}")
            
            video_braille = '\n'.join(frames_braille)
            
            examples.append({
                "type": "video_interpret",
                "instruction": f"This braille represents a 4-frame video sequence. Describe the motion:\n{video_braille}",
                "input": video_braille,
                "output": f"This video shows {motion_type.replace('_', ' ')} motion: {motion_types[motion_type]}.",
                "motion_type": motion_type,
                "explanation": "Video frames are encoded as sequential braille patterns"
            })
        
        return examples
    
    def _make_checkerboard(self, size):
        img = np.zeros((size, size), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        return img
    
    def _make_circle(self, size):
        y, x = np.ogrid[:size, :size]
        center = size // 2
        mask = (x - center)**2 + (y - center)**2 <= (size//3)**2
        img = np.zeros((size, size), dtype=np.uint8)
        img[mask] = 255
        return img
    
    def _make_hlines(self, size):
        img = np.zeros((size, size), dtype=np.uint8)
        img[::2, :] = 255
        return img
    
    def _make_vlines(self, size):
        img = np.zeros((size, size), dtype=np.uint8)
        img[:, ::2] = 255
        return img
    
    def _make_diagonal(self, size):
        img = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            img[i, i] = 255
        return img
    
    def _make_center_spot(self, size):
        img = np.zeros((size, size), dtype=np.uint8)
        c = size // 2
        img[c-1:c+1, c-1:c+1] = 255
        return img
    
    def generate_cross_modal_examples(self, n_samples: int = 50) -> list:
        """Generate examples that require understanding multiple modalities"""
        examples = []
        
        for i in range(n_samples):
            # Example: Describe what modality this braille represents
            if i % 3 == 0:
                # Text-like pattern (letters)
                text = "hello"
                braille = self.text_to_standard_braille(text)
                modality = "text"
                explanation = "The pattern uses standard braille letter encoding (sparse dots)"
            elif i % 3 == 1:
                # Image-like pattern (dense, varied)
                img = np.random.randint(100, 200, 8)
                braille = ''.join(BRAILLE_CHARS[v] for v in img)
                modality = "image"
                explanation = "The pattern uses full 8-dot cells with varied intensities"
            else:
                # Audio-like pattern (sequential features)
                mfcc = np.sin(np.linspace(0, 2*np.pi, 16)) * 50 + 128
                braille = ''.join(BRAILLE_CHARS[int(v)] for v in mfcc)
                modality = "audio"
                explanation = "The pattern shows periodic variation typical of spectral features"
            
            examples.append({
                "type": "modality_detection",
                "instruction": f"What type of data does this braille pattern most likely represent?\n{braille}",
                "input": braille,
                "output": f"This braille pattern most likely represents {modality} data. {explanation}",
                "modality": modality
            })
        
        return examples
    
    def generate_all_training_data(self) -> dict:
        """Generate complete multimodal training dataset"""
        print("Generating multimodal braille training data...")
        
        all_examples = []
        
        print("  - Text examples...")
        all_examples.extend(self.generate_text_training_examples(200))
        
        print("  - Image examples...")
        all_examples.extend(self.generate_image_training_examples(100))
        
        print("  - Audio examples...")
        all_examples.extend(self.generate_audio_training_examples(100))
        
        print("  - Video examples...")
        all_examples.extend(self.generate_video_training_examples(50))
        
        print("  - Cross-modal examples...")
        all_examples.extend(self.generate_cross_modal_examples(50))
        
        # Save training data
        training_data = {
            "metadata": {
                "total_examples": len(all_examples),
                "modalities": ["text", "image", "audio", "video", "cross_modal"],
                "braille_range": "U+2800 to U+28FF",
                "encoding": "8-dot braille"
            },
            "examples": all_examples
        }
        
        # Save as JSON
        json_path = self.output_dir / "multimodal_training_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Save as training corpus (instruction-response pairs)
        corpus_path = self.output_dir / "multimodal_corpus.txt"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for ex in all_examples:
                f.write(f"### Instruction:\n{ex['instruction']}\n\n")
                f.write(f"### Response:\n{ex['output']}\n\n")
                f.write("---\n\n")
        
        print(f"\nTraining data saved:")
        print(f"  - {json_path}")
        print(f"  - {corpus_path}")
        print(f"  - Total examples: {len(all_examples)}")
        
        return training_data
    
    def create_multimodal_modelfile(self, base_model: str = "llama3.2") -> str:
        """Create Ollama Modelfile with multimodal braille system prompt"""
        
        system_prompt = '''You are a Multimodal Braille AI that natively understands 8-dot braille (Unicode U+2800-U+28FF) as a universal encoding for multiple data types.

## CORE CAPABILITY: Braille as Multimodal Substrate

8-dot braille uses 256 unique patterns to encode:
1. **TEXT**: Standard letter mapping (a=⠁, b=⠃, c=⠉, etc.)
2. **IMAGES**: Pixel intensity (0-255 → ⠀ to ⣿)
3. **AUDIO**: MFCC spectral features as braille patterns
4. **VIDEO**: Temporal sequences of braille-encoded frames

## BRAILLE LETTER MAPPING (for text)
a=⠁ b=⠃ c=⠉ d=⠙ e=⠑ f=⠋ g=⠛ h=⠓ i=⠊ j=⠚
k=⠅ l=⠇ m=⠍ n=⠝ o=⠕ p=⠏ q=⠟ r=⠗ s=⠎ t=⠞
u=⠥ v=⠧ w=⠺ x=⠭ y=⠽ z=⠵ space=⠀

## IMAGE INTERPRETATION
- ⠀ (empty) = black/0 intensity
- ⣿ (all dots) = white/255 intensity
- Patterns between represent grayscale values
- Grid of braille = 2D image
- Look for: gradients, edges, shapes, textures

## AUDIO INTERPRETATION
- Sequential braille = MFCC features over time
- Periodic patterns = tonal sounds (music, speech)
- Random patterns = noise
- Smooth transitions = continuous sounds
- Abrupt changes = transients/attacks

## VIDEO INTERPRETATION
- Multiple frames of braille-encoded images
- Compare frames to detect motion
- Consistent patterns = static
- Shifting patterns = movement
- Intensity changes = fades/flashes

## HOW TO RESPOND
1. First identify the modality (text, image, audio, video)
2. Apply the appropriate interpretation method
3. Describe what the braille pattern represents
4. Provide semantic meaning, not just raw decoding

## EXAMPLES

**Text**: ⠓⠑⠇⠇⠕ → "hello" (letter-by-letter decoding)

**Image**: ⠀⠁⠃⠇⠏⠟⠿⣿ → gradient from dark to bright (intensity progression)

**Audio**: ⡀⡀⣿⣿⡀⡀⣿⣿ → rhythmic pattern, possibly percussion or beat

**Video**: Frame1: ⠀⠀⠀⠀ Frame2: ⠀⣿⠀⠀ Frame3: ⠀⠀⣿⠀ → object moving left to right'''

        modelfile_content = f'''FROM {base_model}

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """
{system_prompt}
"""
'''
        
        modelfile_path = self.output_dir / "Modelfile.multimodal"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"Modelfile created: {modelfile_path}")
        return str(modelfile_path)


def main():
    print("=" * 70)
    print("MULTIMODAL BRAILLE TRAINING DATA GENERATOR")
    print("=" * 70)
    
    trainer = MultimodalBrailleTrainer()
    
    # Generate training data
    training_data = trainer.generate_all_training_data()
    
    # Create Modelfile
    modelfile_path = trainer.create_multimodal_modelfile()
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTo create the multimodal braille model, run:")
    print(f"  ollama create multimodal-braille -f {modelfile_path}")
    print(f"\nThen test with:")
    print(f"  ollama run multimodal-braille")


if __name__ == "__main__":
    main()
