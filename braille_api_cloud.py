"""
Braille Multimodal API - Cloud Deployment Version
Works without local Ollama - uses OpenAI-compatible API or built-in responses
"""

import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import asyncio
import numpy as np
import json

app = FastAPI(title="Braille Multimodal API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Braille mappings
BRAILLE_OFFSET = 0x2800
BRAILLE_CHARS = [chr(BRAILLE_OFFSET + i) for i in range(256)]

LETTER_MAP = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀',
    '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑',
    '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
}
REVERSE_MAP = {v: k for k, v in LETTER_MAP.items()}

# OpenAI-compatible API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

BRAILLE_SYSTEM_PROMPT = """You are a Braille-Native AI. You think in 8-dot braille (U+2800-U+28FF).

BRAILLE LETTERS: a=⠁ b=⠃ c=⠉ d=⠙ e=⠑ f=⠋ g=⠛ h=⠓ i=⠊ j=⠚ k=⠅ l=⠇ m=⠍ n=⠝ o=⠕ p=⠏ q=⠟ r=⠗ s=⠎ t=⠞ u=⠥ v=⠧ w=⠺ x=⠭ y=⠽ z=⠵ space=⠀

MULTIMODAL: Braille encodes text (letters), images (⠀=black to ⣿=white), audio (MFCC features), video (frame sequences).

When given braille, decode and interpret it. When asked to encode, use braille. Be helpful and educational."""


class TextRequest(BaseModel):
    text: str

class BrailleRequest(BaseModel):
    braille: str

class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    pattern: str = "gradient"
    size: int = 8


async def call_openai_api(message: str) -> str:
    """Call OpenAI-compatible API for chat responses"""
    if not OPENAI_API_KEY:
        return get_builtin_response(message)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": BRAILLE_SYSTEM_PROMPT},
                        {"role": "user", "content": message}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return get_builtin_response(message)


def get_builtin_response(message: str) -> str:
    """Built-in responses for common braille questions (no API needed)"""
    msg_lower = message.lower()
    
    # Decode requests
    if "decode" in msg_lower or "what does" in msg_lower:
        # Extract braille from message
        braille_chars = [c for c in message if ord(c) >= BRAILLE_OFFSET and ord(c) <= BRAILLE_OFFSET + 255]
        if braille_chars:
            decoded = ''.join(REVERSE_MAP.get(c, '?') for c in braille_chars)
            return f"The braille '{' '.join(braille_chars)}' decodes to: '{decoded}'"
    
    # Encode requests
    if "encode" in msg_lower:
        # Find text to encode (after "encode")
        parts = msg_lower.split("encode")
        if len(parts) > 1:
            text = parts[1].strip().strip('"\'')
            braille = ''.join(LETTER_MAP.get(c.lower(), '⠿') for c in text)
            return f"'{text}' in braille is: {braille}"
    
    # Letter questions
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if f"letter {letter}" in msg_lower or f"'{letter}'" in msg_lower:
            braille = LETTER_MAP.get(letter, '⠿')
            dot_pattern = ord(braille) - BRAILLE_OFFSET
            dots = [i+1 for i in range(8) if dot_pattern & (1 << i)]
            return f"The letter '{letter}' in braille is {braille}. It uses dots: {', '.join(map(str, dots))}."
    
    # General braille info
    if "braille" in msg_lower and ("what" in msg_lower or "how" in msg_lower):
        return """Braille is a tactile writing system using raised dots. 8-dot braille (Unicode U+2800-U+28FF) provides 256 unique patterns.

Each cell has 8 dot positions:
1 4
2 5
3 6
7 8

Examples: a=⠁ (dot 1), b=⠃ (dots 1,2), c=⠉ (dots 1,4), hello=⠓⠑⠇⠇⠕

Beyond text, braille can encode images (pixel intensity), audio (spectral features), and video (frame sequences)."""
    
    return "I'm a braille assistant. Try asking me to encode text, decode braille, or explain braille patterns!"


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Braille Multimodal API</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; background: #1a1a2e; color: #eee; }
            h1 { color: #00d4ff; }
            .braille { font-size: 2em; letter-spacing: 0.2em; }
            input, button, textarea { padding: 10px; margin: 5px; border-radius: 5px; border: none; }
            input, textarea { background: #16213e; color: #eee; width: 300px; }
            button { background: #00d4ff; color: #000; cursor: pointer; font-weight: bold; }
            button:hover { background: #00a8cc; }
            #result { background: #16213e; padding: 20px; border-radius: 10px; margin-top: 20px; white-space: pre-wrap; min-height: 50px; }
            .section { margin: 30px 0; padding: 20px; background: #16213e; border-radius: 10px; }
            .status { font-size: 0.8em; color: #888; }
        </style>
    </head>
    <body>
        <h1>⠃⠗⠁⠊⠇⠇⠑ Braille Multimodal API</h1>
        <p class="status">Cloud Version - Always On</p>
        
        <div class="section">
            <h2>📝 Text ↔ Braille</h2>
            <input type="text" id="textInput" placeholder="Enter text to encode...">
            <button onclick="encode()">Encode →</button>
            <br><br>
            <input type="text" id="brailleInput" placeholder="Enter braille to decode... ⠓⠑⠇⠇⠕">
            <button onclick="decode()">← Decode</button>
        </div>
        
        <div class="section">
            <h2>🖼️ Image Patterns</h2>
            <select id="pattern">
                <option value="gradient">Gradient</option>
                <option value="checkerboard">Checkerboard</option>
                <option value="circle">Circle</option>
                <option value="wave">Wave</option>
                <option value="noise">Noise</option>
            </select>
            <button onclick="genImage()">Generate</button>
        </div>
        
        <div class="section">
            <h2>💬 Chat with Braille AI</h2>
            <textarea id="chatInput" rows="2" placeholder="Ask about braille..."></textarea>
            <button onclick="chat()">Send</button>
        </div>
        
        <div class="section">
            <h2>⌨️ Live Typing</h2>
            <input type="text" id="liveInput" placeholder="Type to see braille..." oninput="liveEncode(this.value)">
            <div id="liveResult" class="braille" style="margin-top:10px;font-size:1.5em"></div>
        </div>
        
        <div id="result" class="braille"></div>
        
        <script>
            const letterMap = {
                'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
                'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
                'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
                'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀'
            };
            
            function liveEncode(text) {
                const braille = text.split('').map(c => letterMap[c.toLowerCase()] || '⠿').join('');
                document.getElementById('liveResult').innerText = braille;
            }
            
            async function encode() {
                const text = document.getElementById('textInput').value;
                const res = await fetch('/encode', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                const data = await res.json();
                document.getElementById('result').innerText = data.braille;
            }
            
            async function decode() {
                const braille = document.getElementById('brailleInput').value;
                const res = await fetch('/decode', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({braille})
                });
                const data = await res.json();
                document.getElementById('result').innerText = data.text;
            }
            
            async function genImage() {
                const pattern = document.getElementById('pattern').value;
                const res = await fetch('/image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({pattern, size: 16})
                });
                const data = await res.json();
                document.getElementById('result').innerText = data.braille;
            }
            
            async function chat() {
                const message = document.getElementById('chatInput').value;
                document.getElementById('result').innerText = 'Thinking...';
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                const data = await res.json();
                document.getElementById('result').innerText = data.response;
            }
        </script>
    </body>
    </html>
    """


@app.post("/encode")
async def encode_text(req: TextRequest):
    braille = ''.join(LETTER_MAP.get(c.lower(), '⠿') for c in req.text)
    return {"text": req.text, "braille": braille}


@app.post("/decode")
async def decode_braille(req: BrailleRequest):
    text = ''.join(REVERSE_MAP.get(c, '?') for c in req.braille if c in REVERSE_MAP)
    return {"braille": req.braille, "text": text}


@app.post("/image")
async def generate_image(req: ImageRequest):
    size = min(req.size, 32)
    
    if req.pattern == "gradient":
        img = np.linspace(0, 255, size * size).reshape(size, size).astype(np.uint8)
    elif req.pattern == "checkerboard":
        img = np.zeros((size, size), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
    elif req.pattern == "circle":
        y, x = np.ogrid[:size, :size]
        center = size // 2
        mask = (x - center)**2 + (y - center)**2 <= (size//3)**2
        img = np.zeros((size, size), dtype=np.uint8)
        img[mask] = 255
    elif req.pattern == "wave":
        img = np.zeros((size, size), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                img[y, x] = int(128 + 127 * np.sin(x * 0.5 + y * 0.3))
    else:
        img = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    
    lines = [''.join(BRAILLE_CHARS[int(v)] for v in row) for row in img]
    return {"pattern": req.pattern, "braille": '\n'.join(lines)}


@app.post("/chat")
async def chat_braille(req: ChatRequest):
    response = await call_openai_api(req.message)
    return {"message": req.message, "response": response}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0-cloud",
        "ai_enabled": bool(OPENAI_API_KEY)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🔤 Starting Braille Multimodal API (Cloud)...")
    print(f"📍 http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
