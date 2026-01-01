"""
Multimodal-to-Braille Conversion System
Converts audio, video, text, and images to 8-dot braille representations
"""

import numpy as np
import cv2
import librosa
import soundfile as sf
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import hashlib

# 8-dot Braille Unicode range: U+2800 to U+28FF (256 characters)
# Each braille character represents 8 dots arranged in 2 columns of 4 dots
BRAILLE_OFFSET = 0x2800
BRAILLE_CHARS = [chr(BRAILLE_OFFSET + i) for i in range(256)]

class BrailleConverter:
    """Core converter for multimodal data to 8-dot braille"""
    
    def __init__(self, resolution: int = 32):
        """
        Initialize the converter
        
        Args:
            resolution: Grid resolution for visual encoding (default 32x32)
        """
        self.resolution = resolution
        self.braille_map = {}
        
    def _value_to_braille_pattern(self, value: float, max_value: float = 1.0) -> int:
        """
        Convert a normalized value [0, 1] to an 8-dot braille pattern (0-255)
        
        Each dot represents a bit:
        Dot positions:
        1 4
        2 5
        3 6
        7 8
        
        Args:
            value: Value between 0 and 1
            max_value: Maximum value for normalization
            
        Returns:
            Integer 0-255 representing braille dot pattern
        """
        normalized = min(1.0, max(0.0, value / max_value))
        # Map value to 8-bit pattern (0-255)
        pattern = int(normalized * 255)
        return pattern
    
    def text_to_braille(self, text: str) -> str:
        """
        Convert text to braille representation
        Uses character encoding mapped to braille patterns
        
        Args:
            text: Input text string
            
        Returns:
            Braille-encoded string
        """
        braille_text = ""
        for char in text:
            # Use ASCII value modulo 256 to map to braille
            ascii_val = ord(char)
            braille_pattern = ascii_val % 256
            braille_text += BRAILLE_CHARS[braille_pattern]
        return braille_text
    
    def image_to_braille(self, image_path: Union[str, np.ndarray], 
                        downscale: bool = True) -> Tuple[str, Dict]:
        """
        Convert image to braille representation
        Downscales image and maps pixel intensities to braille patterns
        
        Args:
            image_path: Path to image file or numpy array
            downscale: Whether to downscale to resolution x resolution
            
        Returns:
            Tuple of (braille_string, metadata)
        """
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = image_path
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0-255
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        if downscale:
            img = cv2.resize(img, (self.resolution, self.resolution))
        
        # Convert to braille
        braille_rows = []
        for row in img:
            braille_row = ""
            for pixel in row:
                pattern = self._value_to_braille_pattern(pixel, 255)
                braille_row += BRAILLE_CHARS[pattern]
            braille_rows.append(braille_row)
        
        braille_string = "\n".join(braille_rows)
        
        metadata = {
            "type": "image",
            "original_shape": img.shape,
            "resolution": self.resolution,
            "hash": hashlib.md5(img.tobytes()).hexdigest()
        }
        
        return braille_string, metadata
    
    def audio_to_braille(self, audio_path: Union[str, Tuple[np.ndarray, int]],
                        n_mfcc: int = 13, hop_length: int = 512) -> Tuple[str, Dict]:
        """
        Convert audio to braille representation
        Uses MFCC (Mel-frequency cepstral coefficients) for audio feature extraction
        
        Args:
            audio_path: Path to audio file or (audio_data, sr) tuple
            n_mfcc: Number of MFCC coefficients
            hop_length: Number of samples between frames
            
        Returns:
            Tuple of (braille_string, metadata)
        """
        if isinstance(audio_path, str):
            y, sr = librosa.load(audio_path, sr=None)
        else:
            y, sr = audio_path
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        
        # Normalize MFCC
        mfcc_norm = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
        
        # Downsample time dimension to fit resolution
        if mfcc_norm.shape[1] > self.resolution:
            indices = np.linspace(0, mfcc_norm.shape[1] - 1, self.resolution, dtype=int)
            mfcc_norm = mfcc_norm[:, indices]
        elif mfcc_norm.shape[1] < self.resolution:
            # Pad with zeros
            pad_width = ((0, 0), (0, self.resolution - mfcc_norm.shape[1]))
            mfcc_norm = np.pad(mfcc_norm, pad_width, mode='constant')
        
        # Pad or truncate frequency dimension
        if mfcc_norm.shape[0] < self.resolution:
            pad_width = ((0, self.resolution - mfcc_norm.shape[0]), (0, 0))
            mfcc_norm = np.pad(mfcc_norm, pad_width, mode='constant')
        else:
            mfcc_norm = mfcc_norm[:self.resolution, :]
        
        # Convert to braille
        braille_rows = []
        for row in mfcc_norm:
            braille_row = ""
            for value in row:
                pattern = self._value_to_braille_pattern(value, 1.0)
                braille_row += BRAILLE_CHARS[pattern]
            braille_rows.append(braille_row)
        
        braille_string = "\n".join(braille_rows)
        
        metadata = {
            "type": "audio",
            "sample_rate": sr,
            "duration": len(y) / sr,
            "n_mfcc": n_mfcc,
            "shape": mfcc_norm.shape,
            "hash": hashlib.md5(mfcc_norm.tobytes()).hexdigest()
        }
        
        return braille_string, metadata
    
    def video_to_braille(self, video_path: str, n_frames: int = 5) -> Tuple[List[str], Dict]:
        """
        Convert video to braille representation
        Extracts key frames and converts each to braille
        
        Args:
            video_path: Path to video file
            n_frames: Number of frames to extract
            
        Returns:
            Tuple of (list of braille_strings, metadata)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        braille_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert frame to grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                braille_frame, _ = self.image_to_braille(frame_gray, downscale=True)
                braille_frames.append(braille_frame)
        
        cap.release()
        
        metadata = {
            "type": "video",
            "total_frames": total_frames,
            "fps": fps,
            "duration": total_frames / fps if fps > 0 else 0,
            "sampled_frames": n_frames,
            "resolution": self.resolution
        }
        
        return braille_frames, metadata
    
    def braille_to_vector(self, braille_string: str) -> np.ndarray:
        """
        Convert braille string to numerical vector for ML processing
        
        Args:
            braille_string: Braille-encoded string
            
        Returns:
            Numpy array of braille pattern values
        """
        vector = []
        for char in braille_string:
            if ord(char) >= BRAILLE_OFFSET:
                pattern = ord(char) - BRAILLE_OFFSET
                vector.append(pattern)
        return np.array(vector, dtype=np.uint8)
    
    def vector_to_braille(self, vector: np.ndarray) -> str:
        """
        Convert numerical vector back to braille string
        
        Args:
            vector: Numpy array of values 0-255
            
        Returns:
            Braille-encoded string
        """
        braille_string = ""
        for val in vector:
            pattern = int(val) % 256
            braille_string += BRAILLE_CHARS[pattern]
        return braille_string


class MultimodalBrailleDataset:
    """Dataset manager for multimodal braille-encoded data"""
    
    def __init__(self, output_dir: str = "/tmp/braille_dataset"):
        """
        Initialize dataset manager
        
        Args:
            output_dir: Directory to store encoded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = BrailleConverter()
        self.manifest = []
    
    def add_text_sample(self, text: str, label: str = "text") -> Dict:
        """Add text sample to dataset"""
        braille_text = self.converter.text_to_braille(text)
        sample = {
            "type": "text",
            "label": label,
            "original": text,
            "braille": braille_text,
            "braille_vector": self.converter.braille_to_vector(braille_text).tolist()
        }
        self.manifest.append(sample)
        return sample
    
    def add_image_sample(self, image_path: str, label: str = "image") -> Dict:
        """Add image sample to dataset"""
        braille_image, metadata = self.converter.image_to_braille(image_path)
        sample = {
            "type": "image",
            "label": label,
            "path": str(image_path),
            "braille": braille_image,
            "braille_vector": self.converter.braille_to_vector(braille_image).tolist(),
            "metadata": metadata
        }
        self.manifest.append(sample)
        return sample
    
    def add_audio_sample(self, audio_path: str, label: str = "audio") -> Dict:
        """Add audio sample to dataset"""
        braille_audio, metadata = self.converter.audio_to_braille(audio_path)
        sample = {
            "type": "audio",
            "label": label,
            "path": str(audio_path),
            "braille": braille_audio,
            "braille_vector": self.converter.braille_to_vector(braille_audio).tolist(),
            "metadata": metadata
        }
        self.manifest.append(sample)
        return sample
    
    def add_video_sample(self, video_path: str, label: str = "video", n_frames: int = 5) -> Dict:
        """Add video sample to dataset"""
        braille_frames, metadata = self.converter.video_to_braille(video_path, n_frames)
        sample = {
            "type": "video",
            "label": label,
            "path": str(video_path),
            "braille_frames": braille_frames,
            "braille_vectors": [self.converter.braille_to_vector(f).tolist() for f in braille_frames],
            "metadata": metadata
        }
        self.manifest.append(sample)
        return sample
    
    def save_manifest(self, filename: str = "manifest.json"):
        """Save dataset manifest to file"""
        manifest_path = self.output_dir / filename
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        return str(manifest_path)
    
    def get_braille_corpus(self) -> str:
        """Get concatenated braille corpus for training"""
        corpus = []
        for sample in self.manifest:
            if sample["type"] in ["text", "image", "audio"]:
                corpus.append(sample["braille"])
            elif sample["type"] == "video":
                corpus.extend(sample["braille_frames"])
        return "\n---\n".join(corpus)


if __name__ == "__main__":
    print("Multimodal-to-Braille Conversion System initialized")
    print(f"Braille character range: U+2800 to U+28FF ({len(BRAILLE_CHARS)} characters)")
    
    # Quick test
    converter = BrailleConverter()
    
    # Test text conversion
    test_text = "Hello Braille!"
    braille_text = converter.text_to_braille(test_text)
    print(f"\nText: {test_text}")
    print(f"Braille: {braille_text}")
    
    # Test vector conversion
    vector = converter.braille_to_vector(braille_text)
    print(f"Vector shape: {vector.shape}")
    print(f"Vector sample: {vector[:10]}")
