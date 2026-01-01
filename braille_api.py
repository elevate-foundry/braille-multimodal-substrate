"""
Lightweight Braille API - Direct Ollama integration without Parlant overhead
Fast, simple REST API for multimodal braille operations
Includes WebSocket streaming for real-time AI responses
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import subprocess
import asyncio
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
            <h2>💬 Chat with Braille AI <span id="wsStatus" style="font-size:0.5em;color:#666">(REST)</span></h2>
            <textarea id="chatInput" rows="2" placeholder="Ask about braille..."></textarea>
            <button onclick="chat()">Send (REST)</button>
            <button onclick="streamChat()" style="background:#9c27b0">Stream (WebSocket)</button>
        </div>
        
        <div class="section">
            <h2>⌨️ Live Typing (WebSocket)</h2>
            <input type="text" id="liveInput" placeholder="Type to see braille in real-time..." oninput="liveEncode(this.value)">
            <div id="liveResult" class="braille" style="margin-top:10px;font-size:1.5em"></div>
        </div>
        
        <div id="result" class="braille"></div>
        
        <script>
            // WebSocket connections
            let chatWs = null;
            let encodeWs = null;
            
            function connectChatWs() {
                if (chatWs && chatWs.readyState === WebSocket.OPEN) return;
                chatWs = new WebSocket('ws://' + window.location.host + '/ws/chat');
                chatWs.onopen = () => document.getElementById('wsStatus').textContent = '(WebSocket Connected)';
                chatWs.onclose = () => document.getElementById('wsStatus').textContent = '(WebSocket Disconnected)';
                chatWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    const result = document.getElementById('result');
                    if (data.type === 'start') {
                        result.innerText = '';
                    } else if (data.type === 'token') {
                        result.innerText += data.content;
                    } else if (data.type === 'complete') {
                        // Done streaming
                    } else if (data.type === 'error') {
                        result.innerText = 'Error: ' + data.error;
                    }
                };
            }
            
            function connectEncodeWs() {
                if (encodeWs && encodeWs.readyState === WebSocket.OPEN) return;
                encodeWs = new WebSocket('ws://' + window.location.host + '/ws/encode');
                encodeWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    document.getElementById('liveResult').innerText = data.braille;
                };
            }
            
            function streamChat() {
                connectChatWs();
                const message = document.getElementById('chatInput').value;
                document.getElementById('result').innerText = 'Streaming...';
                if (chatWs.readyState === WebSocket.OPEN) {
                    chatWs.send(JSON.stringify({message}));
                } else {
                    chatWs.onopen = () => chatWs.send(JSON.stringify({message}));
                }
            }
            
            function liveEncode(text) {
                connectEncodeWs();
                if (encodeWs.readyState === WebSocket.OPEN) {
                    encodeWs.send(text);
                }
            }
            
            // Connect on page load
            setTimeout(() => { connectChatWs(); connectEncodeWs(); }, 500);
            
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


# ============================================================================
# WEBSOCKET STREAMING
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming AI chat responses.
    Send a message, receive tokens as they're generated.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                await websocket.send_json({"error": "No message provided"})
                continue
            
            # Send start signal
            await websocket.send_json({"type": "start", "message": user_message})
            
            # Stream response from Ollama
            try:
                process = await asyncio.create_subprocess_exec(
                    "ollama", "run", "braille-fast", user_message,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Stream stdout
                full_response = ""
                while True:
                    chunk = await process.stdout.read(50)  # Read in small chunks
                    if not chunk:
                        break
                    text = chunk.decode('utf-8', errors='ignore')
                    full_response += text
                    await websocket.send_json({
                        "type": "token",
                        "content": text
                    })
                
                # Wait for process to complete
                await process.wait()
                
                # Send completion signal
                await websocket.send_json({
                    "type": "complete",
                    "full_response": full_response.strip()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


@app.websocket("/ws/encode")
async def websocket_encode(websocket: WebSocket):
    """
    WebSocket for real-time text-to-braille encoding.
    Useful for live typing feedback.
    """
    await manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            braille = ''.join(LETTER_MAP.get(c.lower(), '⠿') for c in text)
            await websocket.send_json({
                "text": text,
                "braille": braille
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    print("🔤 Starting Braille Multimodal API...")
    print("📍 REST API: http://localhost:8000")
    print("📍 WebSocket Chat: ws://localhost:8000/ws/chat")
    print("📍 WebSocket Encode: ws://localhost:8000/ws/encode")
    uvicorn.run(app, host="0.0.0.0", port=8000)
