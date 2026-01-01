"""
Parlant Braille Agent - Multimodal Braille-Native Conversational AI

This agent uses Parlant's guideline-based control to create a conversational
AI that natively understands 8-dot braille as a multimodal encoding substrate.
"""

import parlant.sdk as p
from braille_converter import BrailleConverter, BRAILLE_CHARS, BRAILLE_OFFSET
import numpy as np
import subprocess
import json

# Initialize braille converter
converter = BrailleConverter()

# Standard braille letter mapping
LETTER_MAP = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀',
}
REVERSE_MAP = {v: k for k, v in LETTER_MAP.items()}


# ============================================================================
# BRAILLE TOOLS
# ============================================================================

@p.tool
async def encode_text_to_braille(context: p.ToolContext, text: str) -> p.ToolResult:
    """
    Encode plain text to standard braille letters.
    
    Args:
        text: The text to encode in braille
    
    Returns:
        Braille-encoded text using standard letter mapping
    """
    braille = ''.join(LETTER_MAP.get(c.lower(), '⠿') for c in text)
    return p.ToolResult(json.dumps({
        "original": text,
        "braille": braille,
        "length": len(braille),
        "encoding": "standard-letters"
    }))


@p.tool
async def decode_braille_to_text(context: p.ToolContext, braille: str) -> p.ToolResult:
    """
    Decode braille letters back to plain text.
    
    Args:
        braille: The braille text to decode
    
    Returns:
        Decoded plain text
    """
    text = ''.join(REVERSE_MAP.get(c, '?') for c in braille)
    return p.ToolResult(json.dumps({
        "braille": braille,
        "decoded": text,
        "encoding": "standard-letters"
    }))


@p.tool
async def encode_image_to_braille(
    context: p.ToolContext, 
    pattern: str = "gradient",
    size: int = 8
) -> p.ToolResult:
    """
    Generate a braille-encoded image pattern.
    
    Args:
        pattern: Type of pattern (gradient, checkerboard, circle, noise, solid_dark, solid_bright)
        size: Size of the image grid (default 8x8)
    
    Returns:
        Braille-encoded image representation
    """
    if pattern == "gradient":
        img = np.linspace(0, 255, size * size).reshape(size, size).astype(np.uint8)
    elif pattern == "checkerboard":
        img = np.zeros((size, size), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
    elif pattern == "circle":
        y, x = np.ogrid[:size, :size]
        center = size // 2
        mask = (x - center)**2 + (y - center)**2 <= (size//3)**2
        img = np.zeros((size, size), dtype=np.uint8)
        img[mask] = 255
    elif pattern == "noise":
        img = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    elif pattern == "solid_dark":
        img = np.zeros((size, size), dtype=np.uint8)
    elif pattern == "solid_bright":
        img = np.full((size, size), 255, dtype=np.uint8)
    else:
        img = np.linspace(0, 255, size * size).reshape(size, size).astype(np.uint8)
    
    # Convert to braille
    braille_lines = []
    for row in img:
        braille_row = ''.join(BRAILLE_CHARS[int(v)] for v in row)
        braille_lines.append(braille_row)
    
    braille_img = '\n'.join(braille_lines)
    
    return p.ToolResult(json.dumps({
        "pattern": pattern,
        "size": f"{size}x{size}",
        "braille": braille_img,
        "encoding": "pixel-intensity",
        "description": f"Each braille character represents pixel intensity 0-255"
    }))


@p.tool
async def interpret_braille_pattern(
    context: p.ToolContext,
    braille: str
) -> p.ToolResult:
    """
    Analyze a braille pattern and determine its modality and meaning.
    
    Args:
        braille: The braille pattern to interpret
    
    Returns:
        Analysis of the braille pattern including modality detection
    """
    # Analyze pattern characteristics
    chars = [c for c in braille if c in BRAILLE_CHARS or ord(c) >= BRAILLE_OFFSET]
    
    if not chars:
        return p.ToolResult(json.dumps({"error": "No valid braille characters found"}))
    
    # Check if it looks like text (uses sparse letter patterns)
    text_chars = set(LETTER_MAP.values())
    text_ratio = sum(1 for c in chars if c in text_chars) / len(chars)
    
    # Check intensity distribution
    intensities = [ord(c) - BRAILLE_OFFSET for c in chars if ord(c) >= BRAILLE_OFFSET]
    avg_intensity = np.mean(intensities) if intensities else 0
    intensity_std = np.std(intensities) if intensities else 0
    
    # Determine modality
    if text_ratio > 0.7:
        modality = "text"
        decoded = ''.join(REVERSE_MAP.get(c, '?') for c in chars)
        interpretation = f"Text content: '{decoded}'"
    elif '\n' in braille:
        modality = "image"
        lines = braille.strip().split('\n')
        interpretation = f"Image grid: {len(lines)} rows, appears to be a visual pattern"
    elif intensity_std > 50:
        modality = "audio"
        interpretation = "High variation suggests audio MFCC features"
    else:
        modality = "unknown"
        interpretation = "Pattern type unclear"
    
    return p.ToolResult(json.dumps({
        "modality": modality,
        "interpretation": interpretation,
        "stats": {
            "char_count": len(chars),
            "avg_intensity": round(avg_intensity, 2),
            "intensity_std": round(intensity_std, 2),
            "text_ratio": round(text_ratio, 2)
        }
    }))


@p.tool
async def generate_audio_braille(
    context: p.ToolContext,
    audio_type: str = "sine",
    length: int = 32
) -> p.ToolResult:
    """
    Generate braille-encoded audio features.
    
    Args:
        audio_type: Type of audio (sine, noise, chirp, speech, rhythm)
        length: Number of braille characters (time steps)
    
    Returns:
        Braille-encoded audio representation
    """
    if audio_type == "sine":
        values = np.sin(np.linspace(0, 4*np.pi, length)) * 50 + 128
    elif audio_type == "noise":
        values = np.random.randint(0, 256, length)
    elif audio_type == "chirp":
        values = np.linspace(50, 200, length)
    elif audio_type == "speech":
        # Simulate speech-like modulation
        values = np.sin(np.linspace(0, 8*np.pi, length)) * 30 + 128
        values += np.random.randn(length) * 10
    elif audio_type == "rhythm":
        # Alternating pattern
        values = np.array([200 if i % 4 < 2 else 50 for i in range(length)])
    else:
        values = np.sin(np.linspace(0, 4*np.pi, length)) * 50 + 128
    
    values = np.clip(values, 0, 255).astype(int)
    braille = ''.join(BRAILLE_CHARS[v] for v in values)
    
    return p.ToolResult(json.dumps({
        "audio_type": audio_type,
        "braille": braille,
        "length": length,
        "encoding": "mfcc-features",
        "description": f"Simulated {audio_type} audio encoded as braille"
    }))


@p.tool
async def query_multimodal_braille_model(
    context: p.ToolContext,
    query: str
) -> p.ToolResult:
    """
    Query the multimodal-braille Ollama model directly.
    
    Args:
        query: The query to send to the model
    
    Returns:
        Response from the multimodal braille model
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "multimodal-braille", query],
            capture_output=True,
            text=True,
            timeout=60
        )
        return p.ToolResult(json.dumps({
            "query": query,
            "response": result.stdout.strip(),
            "model": "multimodal-braille"
        }))
    except subprocess.TimeoutExpired:
        return p.ToolResult(json.dumps({"error": "Model query timed out"}))
    except Exception as e:
        return p.ToolResult(json.dumps({"error": str(e)}))


@p.tool
async def create_video_sequence(
    context: p.ToolContext,
    motion_type: str = "left_to_right",
    frames: int = 4,
    size: int = 4
) -> p.ToolResult:
    """
    Generate a braille-encoded video sequence showing motion.
    
    Args:
        motion_type: Type of motion (left_to_right, fade_in, fade_out, expanding, flashing)
        frames: Number of frames
        size: Size of each frame grid
    
    Returns:
        Braille-encoded video sequence
    """
    frame_brailles = []
    
    for f in range(frames):
        if motion_type == "left_to_right":
            frame = np.zeros((size, size), dtype=np.uint8)
            col = f % size
            frame[:, col] = 255
        elif motion_type == "fade_in":
            intensity = int(255 * f / (frames - 1)) if frames > 1 else 255
            frame = np.full((size, size), intensity, dtype=np.uint8)
        elif motion_type == "fade_out":
            intensity = int(255 * (1 - f / (frames - 1))) if frames > 1 else 0
            frame = np.full((size, size), intensity, dtype=np.uint8)
        elif motion_type == "expanding":
            frame = np.zeros((size, size), dtype=np.uint8)
            radius = (f + 1) * size // frames
            center = size // 2
            for i in range(size):
                for j in range(size):
                    if abs(i - center) <= radius and abs(j - center) <= radius:
                        frame[i, j] = 255
        elif motion_type == "flashing":
            intensity = 255 if f % 2 == 0 else 0
            frame = np.full((size, size), intensity, dtype=np.uint8)
        else:
            frame = np.zeros((size, size), dtype=np.uint8)
        
        braille = ''.join(BRAILLE_CHARS[int(v)] for row in frame for v in row)
        frame_brailles.append(f"Frame {f+1}: {braille}")
    
    video = '\n'.join(frame_brailles)
    
    return p.ToolResult(json.dumps({
        "motion_type": motion_type,
        "frames": frames,
        "size": f"{size}x{size}",
        "video": video,
        "encoding": "frame-sequence"
    }))


# ============================================================================
# PARLANT AGENT SETUP
# ============================================================================

async def main():
    # Use Ollama as the NLP backend (local LLM)
    async with p.Server(nlp_service=p.NLPServices.ollama) as server:
        # Create the braille agent
        agent = await server.create_agent(
            name="BrailleNative",
            description="A multimodal AI that natively understands 8-dot braille as a universal encoding for text, images, audio, and video."
        )
        
        print("🔤 Creating BrailleNative agent...")
        
        # ================================================================
        # GUIDELINES - Control agent behavior with natural language
        # ================================================================
        
        # Text encoding/decoding
        await agent.create_guideline(
            condition="User wants to encode text to braille",
            action="Use the encode_text_to_braille tool to convert their text, then explain the braille output",
            tools=[encode_text_to_braille]
        )
        
        await agent.create_guideline(
            condition="User provides braille text to decode or asks what braille means",
            action="Use decode_braille_to_text to decode it, then explain the meaning",
            tools=[decode_braille_to_text]
        )
        
        # Image generation and interpretation
        await agent.create_guideline(
            condition="User wants to see or generate a braille image pattern",
            action="Use encode_image_to_braille to generate the pattern, explain what it represents visually",
            tools=[encode_image_to_braille]
        )
        
        # Audio generation
        await agent.create_guideline(
            condition="User asks about audio or sound in braille",
            action="Use generate_audio_braille to create audio features, explain how braille encodes spectral characteristics",
            tools=[generate_audio_braille]
        )
        
        # Video/motion
        await agent.create_guideline(
            condition="User asks about video, motion, or animation in braille",
            action="Use create_video_sequence to generate frames, explain the motion pattern",
            tools=[create_video_sequence]
        )
        
        # Pattern interpretation
        await agent.create_guideline(
            condition="User provides a braille pattern and asks what it represents",
            action="Use interpret_braille_pattern to analyze it, then explain the modality and meaning",
            tools=[interpret_braille_pattern]
        )
        
        # Deep queries to the multimodal model
        await agent.create_guideline(
            condition="User asks a complex question about braille semantics or needs deep interpretation",
            action="Use query_multimodal_braille_model to get a detailed response from the specialized model",
            tools=[query_multimodal_braille_model]
        )
        
        # General braille education
        await agent.create_guideline(
            condition="User asks about braille basics or how braille encoding works",
            action="Explain that 8-dot braille uses Unicode U+2800-U+28FF (256 patterns) and can encode text (letters), images (pixel intensity), audio (MFCC features), and video (frame sequences). Offer to demonstrate with examples."
        )
        
        # Multimodal explanation
        await agent.create_guideline(
            condition="User asks why braille is used for images/audio/video",
            action="Explain that braille provides a universal 256-symbol vocabulary that maps naturally to byte values (0-255), making it ideal for encoding any data type. This enables 'braille-native cognition' where AI can reason directly in braille space."
        )
        
        print("✅ BrailleNative agent created with multimodal guidelines!")
        print("🌐 Test playground ready at http://localhost:8800")
        print("\nExample queries to try:")
        print("  - 'Encode hello world in braille'")
        print("  - 'What does ⠓⠑⠇⠇⠕ mean?'")
        print("  - 'Show me a gradient image in braille'")
        print("  - 'Generate a rhythmic audio pattern'")
        print("  - 'Create a video of something moving left to right'")
        print("  - 'What type of data is this: ⠀⠁⠃⠇⠏⠟⠿⣿'")
        print("\nPress Ctrl+C to stop the server.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
