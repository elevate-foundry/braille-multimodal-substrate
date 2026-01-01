"""
Lightweight Braille API - Direct Ollama integration without Parlant overhead
Fast, simple REST API for multimodal braille operations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import subprocess
import numpy as np
import json
from braille_converter import BRAILLE_CHARS, BRAILLE_OFFSET

app = FastAPI(title="Braille Multimodal API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Braille mappings
LETTER_MAP = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛',
    'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝',
    'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥',
    'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': '⠀',
}
REVERSE_MAP = {v: k for k, v in LETTER_MAP.items()}


class TextRequest(BaseModel):
    text: str

class BrailleRequest(BaseModel):
    braille: str

class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    pattern: str = "gradient"
    size: int = 8


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
            #result { background: #16213e; padding: 20px; border-radius: 10px; margin-top: 20px; white-space: pre-wrap; }
            .section { margin: 30px 0; padding: 20px; background: #16213e; border-radius: 10px; }
        </style>
    </head>
    <body>
        <h1>⠃⠗⠁⠊⠇⠇⠑ Braille Multimodal API</h1>
        
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
        
        <div id="result" class="braille"></div>
        
        <script>
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
    try:
        result = subprocess.run(
            ["ollama", "run", "braille-fast", req.message],
            capture_output=True,
            text=True,
            timeout=60
        )
        return {"message": req.message, "response": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"message": req.message, "response": "Request timed out"}
    except Exception as e:
        return {"message": req.message, "response": f"Error: {str(e)}"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "braille-fast"}


if __name__ == "__main__":
    import uvicorn
    print("🔤 Starting Braille Multimodal API...")
    print("📍 Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
