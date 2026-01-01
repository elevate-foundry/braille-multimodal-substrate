"""
Minimal Parlant Braille Agent - Optimized for speed
Uses fewer guidelines to reduce LLM calls
"""

import parlant.sdk as p
from braille_converter import BRAILLE_CHARS, BRAILLE_OFFSET
import numpy as np
import json

# Standard braille letter mapping
LETTER_MAP = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀',
}
REVERSE_MAP = {v: k for k, v in LETTER_MAP.items()}


@p.tool
async def braille_encode(context: p.ToolContext, text: str) -> p.ToolResult:
    """Encode text to braille"""
    braille = ''.join(LETTER_MAP.get(c.lower(), '⠿') for c in text)
    return p.ToolResult(f"'{text}' in braille: {braille}")


@p.tool
async def braille_decode(context: p.ToolContext, braille: str) -> p.ToolResult:
    """Decode braille to text"""
    text = ''.join(REVERSE_MAP.get(c, '?') for c in braille if c in REVERSE_MAP or ord(c) >= BRAILLE_OFFSET)
    return p.ToolResult(f"Decoded: '{text}'")


@p.tool
async def braille_image(context: p.ToolContext, pattern: str = "gradient") -> p.ToolResult:
    """Generate braille image pattern (gradient, checkerboard, circle)"""
    size = 8
    if pattern == "gradient":
        img = np.linspace(0, 255, size * size).reshape(size, size).astype(np.uint8)
    elif pattern == "checkerboard":
        img = np.zeros((size, size), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
    else:
        img = np.linspace(0, 255, size * size).reshape(size, size).astype(np.uint8)
    
    lines = [''.join(BRAILLE_CHARS[int(v)] for v in row) for row in img]
    return p.ToolResult(f"{pattern} pattern:\n" + '\n'.join(lines))


async def main():
    async with p.Server(nlp_service=p.NLPServices.ollama) as server:
        agent = await server.create_agent(
            name="BrailleNative",
            description="Braille-native AI using 8-dot braille (U+2800-U+28FF)"
        )
        
        # Single consolidated guideline for braille operations
        await agent.create_guideline(
            condition="User mentions braille, encoding, decoding, or patterns",
            action="Use braille_encode for text→braille, braille_decode for braille→text, braille_image for patterns",
            tools=[braille_encode, braille_decode, braille_image]
        )
        
        print("✅ Minimal BrailleNative agent ready at http://localhost:8800")
        print("Try: 'encode hello' or 'decode ⠓⠑⠇⠇⠕' or 'show gradient pattern'")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
