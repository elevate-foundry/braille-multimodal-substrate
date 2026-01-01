"""
Comprehensive Test Suite for Multimodal Braille System
Tests all components: conversion, encoding, inference, and model configuration
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict
from braille_converter import BrailleConverter, MultimodalBrailleDataset, BRAILLE_CHARS
from braille_ollama_setup import BrailleConstrainedInference, BrailleInferenceEngine
import sys

class BrailleSystemTester:
    """Comprehensive test suite for braille system"""
    
    def __init__(self):
        """Initialize tester"""
        self.converter = BrailleConverter()
        self.inference = BrailleConstrainedInference()
        self.engine = BrailleInferenceEngine()
        self.results = []
        
    def test_text_conversion(self) -> Dict:
        """Test text to braille conversion"""
        print("\n" + "="*70)
        print("TEST 1: TEXT CONVERSION")
        print("="*70)
        
        test_cases = [
            "Hello World",
            "Braille",
            "Multimodal AI",
            "8-dot encoding",
            "Semantic compression"
        ]
        
        results = []
        for text in test_cases:
            braille = self.converter.text_to_braille(text)
            vector = self.converter.braille_to_vector(braille)
            reconstructed = self.converter.vector_to_braille(vector)
            
            result = {
                "text": text,
                "braille": braille,
                "braille_length": len(braille),
                "vector_shape": vector.shape,
                "reconstruction_match": reconstructed == braille
            }
            results.append(result)
            
            print(f"\nText: {text}")
            print(f"Braille: {braille}")
            print(f"Vector shape: {vector.shape}")
            print(f"Reconstruction OK: {result['reconstruction_match']}")
        
        return {"test": "text_conversion", "results": results, "passed": all(r["reconstruction_match"] for r in results)}
    
    def test_image_conversion(self) -> Dict:
        """Test image to braille conversion"""
        print("\n" + "="*70)
        print("TEST 2: IMAGE CONVERSION")
        print("="*70)
        
        # Create synthetic images
        test_images = {
            "gradient": np.linspace(0, 255, 32*32).reshape(32, 32).astype(np.uint8),
            "checkerboard": np.zeros((32, 32), dtype=np.uint8),
            "random": np.random.randint(0, 256, (32, 32), dtype=np.uint8),
            "circle": self._create_circle_pattern()
        }
        
        results = []
        for name, img in test_images.items():
            braille, metadata = self.converter.image_to_braille(img)
            
            result = {
                "pattern": name,
                "image_shape": img.shape,
                "braille_lines": len(braille.split("\n")),
                "metadata": metadata,
                "sample_braille": braille.split("\n")[0][:50]
            }
            results.append(result)
            
            print(f"\nPattern: {name}")
            print(f"Image shape: {img.shape}")
            print(f"Braille lines: {result['braille_lines']}")
            print(f"Sample: {result['sample_braille']}")
        
        return {"test": "image_conversion", "results": results, "passed": len(results) == 4}
    
    def test_audio_conversion(self) -> Dict:
        """Test audio to braille conversion"""
        print("\n" + "="*70)
        print("TEST 3: AUDIO CONVERSION")
        print("="*70)
        
        # Create synthetic audio
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        test_signals = {
            "sine_440Hz": np.sin(2 * np.pi * 440 * t),
            "sine_880Hz": np.sin(2 * np.pi * 880 * t),
            "chirp": np.sin(2 * np.pi * (440 + 440*t) * t),
            "noise": np.random.randn(len(t))
        }
        
        results = []
        for name, signal in test_signals.items():
            audio_data = (signal, sr)
            braille, metadata = self.converter.audio_to_braille(audio_data)
            
            result = {
                "signal": name,
                "duration": metadata["duration"],
                "sample_rate": metadata["sample_rate"],
                "braille_lines": len(braille.split("\n")),
                "metadata": metadata,
                "sample_braille": braille.split("\n")[0][:50]
            }
            results.append(result)
            
            print(f"\nSignal: {name}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Sample rate: {result['sample_rate']} Hz")
            print(f"Braille lines: {result['braille_lines']}")
            print(f"Sample: {result['sample_braille']}")
        
        return {"test": "audio_conversion", "results": results, "passed": len(results) == 4}
    
    def test_braille_vocabulary(self) -> Dict:
        """Test braille character vocabulary"""
        print("\n" + "="*70)
        print("TEST 4: BRAILLE VOCABULARY")
        print("="*70)
        
        # Check all braille characters
        vocab_size = len(BRAILLE_CHARS)
        print(f"Total braille characters: {vocab_size}")
        print(f"Range: U+2800 to U+{2800 + vocab_size - 1:04X}")
        
        # Sample characters
        samples = [BRAILLE_CHARS[i] for i in [0, 64, 128, 192, 255]]
        print(f"Sample characters: {' '.join(samples)}")
        
        # Test vector conversion
        test_vector = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        braille_str = self.converter.vector_to_braille(test_vector)
        recovered_vector = self.converter.braille_to_vector(braille_str)
        
        vectors_match = np.array_equal(test_vector, recovered_vector)
        print(f"Vector conversion round-trip: {'OK' if vectors_match else 'FAILED'}")
        
        return {
            "test": "braille_vocabulary",
            "vocab_size": vocab_size,
            "vector_round_trip": vectors_match,
            "passed": vocab_size == 256 and vectors_match
        }
    
    def test_multimodal_dataset(self) -> Dict:
        """Test multimodal dataset creation"""
        print("\n" + "="*70)
        print("TEST 5: MULTIMODAL DATASET")
        print("="*70)
        
        dataset = MultimodalBrailleDataset()
        
        # Add text sample
        text_sample = dataset.add_text_sample("Multimodal learning", "example")
        print(f"Text sample added: {len(text_sample['braille'])} braille chars")
        
        # Add image sample
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        image_sample = dataset.add_image_sample(img, "example")
        print(f"Image sample added: {len(image_sample['braille'])} braille chars")
        
        # Add audio sample
        audio_data = (np.random.randn(16000), 16000)
        audio_sample = dataset.add_audio_sample(audio_data, "example")
        print(f"Audio sample added: {len(audio_sample['braille'])} braille chars")
        
        # Get corpus
        corpus = dataset.get_braille_corpus()
        print(f"Total corpus size: {len(corpus)} characters")
        
        return {
            "test": "multimodal_dataset",
            "samples_added": len(dataset.manifest),
            "corpus_size": len(corpus),
            "passed": len(dataset.manifest) == 3
        }
    
    def test_inference_prompts(self) -> Dict:
        """Test inference prompt generation"""
        print("\n" + "="*70)
        print("TEST 6: INFERENCE PROMPTS")
        print("="*70)
        
        test_queries = [
            "What is braille?",
            "Explain multimodal encoding",
            "How does compression work?"
        ]
        
        results = []
        for query in test_queries:
            enhanced_prompt = self.inference.encode_prompt_for_braille(query)
            constrained = self.inference.constrain_output_to_braille(query)
            
            result = {
                "query": query,
                "enhanced_prompt_length": len(enhanced_prompt),
                "constrained_output": constrained,
                "all_braille": all(ord(c) >= 0x2800 for c in constrained if ord(c) >= 0x2800 or c in [' ', '\n'])
            }
            results.append(result)
            
            print(f"\nQuery: {query}")
            print(f"Enhanced prompt length: {result['enhanced_prompt_length']}")
            print(f"Constrained: {result['constrained_output']}")
        
        return {"test": "inference_prompts", "results": results, "passed": len(results) == 3}
    
    def test_braille_model_config(self) -> Dict:
        """Test model configuration"""
        print("\n" + "="*70)
        print("TEST 7: MODEL CONFIGURATION")
        print("="*70)
        
        config_path = Path("/home/ubuntu/braille_training/inference_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Model name: {config['model_name']}")
            print(f"Base model: {config['base_model']}")
            print(f"Encoding: {config['encoding']}")
            print(f"Braille range: {config['braille_range']}")
            print(f"Multimodal support: {config['multimodal_support']}")
            print(f"Training samples: {config['training_data']['total_samples']}")
            
            return {
                "test": "model_config",
                "config": config,
                "passed": all([
                    config['encoding'] == '8-dot-braille',
                    config['multimodal_support']['text'],
                    config['multimodal_support']['image'],
                    config['multimodal_support']['audio'],
                    config['multimodal_support']['video']
                ])
            }
        else:
            return {"test": "model_config", "passed": False, "error": "Config file not found"}
    
    def test_end_to_end_pipeline(self) -> Dict:
        """Test complete end-to-end pipeline"""
        print("\n" + "="*70)
        print("TEST 8: END-TO-END PIPELINE")
        print("="*70)
        
        # Create sample data
        text = "Braille is a tactile writing system"
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)), 16000)
        
        # Convert all modalities
        text_braille = self.converter.text_to_braille(text)
        image_braille, _ = self.converter.image_to_braille(img)
        audio_braille, _ = self.converter.audio_to_braille(audio)
        
        # Query model
        query_result = self.engine.query_braille_model(text, "text")
        
        print(f"Text → Braille: {len(text)} → {len(text_braille)} chars")
        print(f"Image → Braille: {img.size} → {len(image_braille)} chars")
        print(f"Audio → Braille: 16000 → {len(audio_braille)} chars")
        print(f"Query result: {query_result['braille_encoded']}")
        
        return {
            "test": "end_to_end",
            "text_conversion": len(text_braille) > 0,
            "image_conversion": len(image_braille) > 0,
            "audio_conversion": len(audio_braille) > 0,
            "query_result": query_result['braille_encoded'],
            "passed": all([
                len(text_braille) > 0,
                len(image_braille) > 0,
                len(audio_braille) > 0
            ])
        }
    
    def _create_circle_pattern(self) -> np.ndarray:
        """Create circular pattern"""
        size = 32
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= (size//3)**2
        img = np.zeros((size, size), dtype=np.uint8)
        img[mask] = 255
        return img
    
    def run_all_tests(self) -> Dict:
        """Run all tests"""
        print("\n" + "="*70)
        print("MULTIMODAL BRAILLE SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_text_conversion,
            self.test_image_conversion,
            self.test_audio_conversion,
            self.test_braille_vocabulary,
            self.test_multimodal_dataset,
            self.test_inference_prompts,
            self.test_braille_model_config,
            self.test_end_to_end_pipeline
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                print(f"ERROR in {test_func.__name__}: {str(e)}")
                results.append({
                    "test": test_func.__name__,
                    "passed": False,
                    "error": str(e)
                })
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in results if r.get("passed", False))
        total = len(results)
        
        for result in results:
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            print(f"{status}: {result['test']}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "results": results,
            "success_rate": passed / total if total > 0 else 0
        }


if __name__ == "__main__":
    tester = BrailleSystemTester()
    summary = tester.run_all_tests()
    
    print("\n" + "="*70)
    print("SYSTEM STATUS")
    print("="*70)
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"Status: {'✓ READY' if summary['success_rate'] == 1.0 else '⚠ PARTIAL'}")
    
    # Save results
    results_path = Path("/home/ubuntu/braille_training/test_results.json")
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
