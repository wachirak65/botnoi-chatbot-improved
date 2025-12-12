import os
import faiss
import pickle
import fitz  # PyMuPDF
import numpy as np
import re
import time
import logging
import asyncio
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob,
    ReplyMessageRequest, TextMessage, QuickReply, QuickReplyItem, 
    MessageAction, PushMessageRequest
)
from linebot.v3.webhooks import (
    CallbackRequest, MessageEvent, StickerMessageContent, ImageMessageContent
)
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
import threading
import ollama
from functools import lru_cache
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO
import base64
from typing import Optional, List, Dict, Tuple
import hashlib
import queue
import signal
import sys

# ---------------- ULTRA FAST CONFIG ----------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "pDtg/t/W/dMzVgNi5aZu224Z2V0hNfJo4VI5/3whwpcDRmKi0UhX7h1AVUWHctoG+zh4zC34047OBIJXKrB7KTPU80z0vXN5Ry7cfEuTVWye1Iu2oRdwy9j9QRsDq7xdJ2at5jDiY7vcS1MDR8+3/AdB04t89/1O/w1cDnyilFU=")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "810bb5572676b7990b40c1c3acae185e")

# ใช้ path เดิม
PDF_FILE = r"C:\Users\Asus 444\Documents\scrap\mini\Using_Platform_Botnoi.pdf"
INDEX_FILE = r"C:\Users\Asus 444\Documents\scrap\mini\Using_Platform_Botnoi.index"
DATA_FILE = r"C:\Users\Asus 444\Documents\scrap\mini\Using_Platform_Botnoi.pkl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
MAX_LINE_CHARS = 4900

# ⚡ ULTRA FAST SETTINGS - Optimized for speed
MAX_WORKERS = 5  # Reduced from 10 for better resource management
CACHE_SIZE = 30  # Reduced from 50 for faster cache operations
FAST_OCR_TIMEOUT = 3  # OCR timeout in seconds
FAST_AI_TIMEOUT = 8   # AI response timeout in seconds
QUICK_CHUNK_SIZE = 150  # Reduced chunk size for faster processing
QUICK_TOP_K = 3  # Reduced search results for speed
REQUEST_QUEUE_SIZE = 100  # Maximum queued requests
MAX_RESPONSE_LENGTH = 800  # Limit response length for speed

# Performance monitoring
PERFORMANCE_LOG_INTERVAL = 100  # Log performance every N requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- FAST TESSERACT SETUP ----------------
if os.name == 'nt':
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logging.info(f"✅ Found Tesseract at: {path}")
            break
    else:
        logging.warning("⚠️ Tesseract not found. OCR functionality will be limited.")

# Initialize models with error handling
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logging.info("✅ Embedding model loaded successfully")
except Exception as e:
    logging.error(f"❌ Failed to load embedding model: {e}")
    embedder = None

# Optimized thread pool
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="UltraFast")

# Request queue for handling concurrent requests
request_queue = queue.Queue(maxsize=REQUEST_QUEUE_SIZE)

# Performance tracking
performance_stats = {
    'total_requests': 0,
    'avg_response_time': 0,
    'cache_hits': 0,
    'pdf_searches': 0,
    'image_processes': 0,
    'errors': 0
}

# Language mapping
LANG_BUTTONS = {
    "en": ("🇺🇸 อังกฤษ", "en"),
    "jp": ("🇯🇵 ญี่ปุ่น", "ja"),
    "cn": ("🇨🇳 จีน", "zh-CN"),
    "kr": ("🇰🇷 เกาหลี", "ko"),
    "fr": ("🇫🇷 ฝรั่งเศส", "fr"),
    "de": ("🇩🇪 เยอรมัน", "de"),
    "vi": ("🇻🇳 เวียดนาม", "vi"),
}

QUICK_REPLY_ITEMS = QuickReply(items=[
    QuickReplyItem(action=MessageAction(label=label, text=f"/{code}_last")) 
    for code, (label, _) in LANG_BUTTONS.items()
] + [
    QuickReplyItem(action=MessageAction(label="🔄 แปลทุกภาษา", text="/translate_all")),
])

# Fast error responses
FAST_ERROR_RESPONSES = {
    "ocr_timeout": "🔍 รูปภาพนี้ซับซ้อน กำลังวิเคราะห์เนื้อหา...",
    "ai_timeout": "⚡ ระบบกำลังประมวลผล กรุณารอสักครู่...",
    "pdf_not_found": "ไม่พบข้อมูลในคู่มือ ลองถามในแบบอื่นครับ",
    "system_busy": "🚀 ระบบยุ่ง กำลังจัดลำดับคำขอ...",
    "general_error": "❌ เกิดข้อผิดพลาด กรุณาลองใหม่ครับ"
}

# Pre-computed quick responses for common queries
QUICK_RESPONSES = {
    "สวัสดี": "👋 สวัสดีครับ! ผมพร้อมช่วยเหลือเรื่อง Botnoi Chatbot",
    "hello": "👋 Hello! I'm ready to help with Botnoi Chatbot questions",
    "hi": "👋 Hi there! How can I help you with Botnoi today?",
    "ขอบคุณ": "🙏 ยินดีครับ! มีอะไรให้ช่วยอีกไหม?",
    "thank you": "🙏 You're welcome! Anything else I can help with?",
    "bye": "👋 ลาก่อนครับ! หวังว่าจะได้ช่วยเหลือคุณอีก",
    "ลาก่อน": "👋 ลาก่อนครับ! ขอบคุณที่ใช้บริการ"
}

# ---------------- ULTRA FAST USER STATE ----------------
class FastUserState:
    """Optimized user state management with minimal memory footprint"""
    def __init__(self):
        self.history = []  # Limited history
        self.last_bot_message = None
        self.ollama_cache = {}  # LRU cache for AI responses
        self.last_request_time = 0
        self.lock = threading.Lock()
        self.quick_responses = {}  # Pre-computed responses
        self.request_count = 0
        self.cache_hits = 0
        
    def add_to_history(self, message: str, timestamp: str):
        """Add message to history with size limit"""
        with self.lock:
            self.history.append({"time": timestamp, "message": message})
            # Keep only last 20 messages for memory efficiency
            if len(self.history) > 20:
                self.history = self.history[-20:]
    
    def cleanup_cache(self):
        """Clean up cache when it gets too large"""
        with self.lock:
            if len(self.ollama_cache) > CACHE_SIZE:
                # Remove oldest 10 entries
                old_keys = list(self.ollama_cache.keys())[:10]
                for key in old_keys:
                    del self.ollama_cache[key]

users = {}
users_lock = threading.Lock()

def get_user_state(user_id: str) -> FastUserState:
    """Get or create user state with thread safety"""
    with users_lock:
        if user_id not in users:
            users[user_id] = FastUserState()
        return users[user_id]

# ---------------- ULTRA FAST TEXT PROCESSING ----------------
def fast_clean_text(text: str) -> str:
    """Ultra-fast text cleaning with minimal regex operations"""
    if not text:
        return ""
    
    # Single regex operation for speed
    cleaned = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,!?()/-]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def fast_extract_keywords(query: str) -> List[str]:
    """Fast keyword extraction with limited processing"""
    keywords = []
    botnoi_keywords = ["บอทน้อย", "botnoi", "แพลตฟอร์ม", "แชทบอท", "สร้างบอท", "ตั้งค่า", "ใช้งาน"]
    query_lower = query.lower()
    
    for keyword in botnoi_keywords:
        if keyword in query_lower:
            keywords.append(keyword)
            if len(keywords) >= 3:  # Limit for speed
                break
    
    return keywords

def is_simple_command(text: str) -> bool:
    """Check if text is a simple command for fast processing"""
    simple_commands = ["hi", "hello", "สวัสดี", "ขอบคุณ", "thank you", "bye", "ลาก่อน", "help", "id", "clear"]
    return text.lower().strip() in simple_commands

# ---------------- ULTRA FAST PDF PROCESSING ----------------
def build_fast_faiss_index(pdf_path: str) -> Tuple:
    """Build FAISS index optimized for speed"""
    logging.info("⚡ Building ULTRA FAST FAISS Index...")
    
    if not os.path.exists(pdf_path):
        logging.error(f"❌ PDF not found: {pdf_path}")
        dim = 1024
        empty_index = faiss.IndexHNSWFlat(dim, 16)  # Reduced parameters
        return empty_index, [], np.zeros((0, dim), dtype=np.float32), []
    
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_metadata = []
    
    for page_num, page in enumerate(doc):
        text = fast_clean_text(page.get_text("text"))
        if not text.strip():
            continue
            
        words = text.split()
        
        # Create non-overlapping chunks for speed
        for i in range(0, len(words), QUICK_CHUNK_SIZE):
            chunk_words = words[i:i + QUICK_CHUNK_SIZE]
            if len(chunk_words) < 5:  # Skip very short chunks
                continue
                
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            chunk_metadata.append({
                'page': page_num + 1,
                'keywords': fast_extract_keywords(chunk)
            })
    
    if not chunks:
        logging.warning("⚠️ No chunks found")
        dim = 1024
        empty_index = faiss.IndexHNSWFlat(dim, 16)
        return empty_index, [], np.zeros((0, dim), dtype=np.float32), []
    
    # Create embeddings
    try:
        embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
    except Exception as e:
        logging.error(f"❌ Embedding failed: {e}")
        return None, [], np.array([]), []
    
    # Build optimized FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16)  # Reduced for speed
    index.hnsw.efConstruction = 100  # Reduced from 200
    index.add(embeddings)
    
    # Save files
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(DATA_FILE, "wb") as f:
            pickle.dump((chunks, embeddings, chunk_metadata), f)
        logging.info(f"✅ Fast index built: {len(chunks)} chunks")
    except Exception as e:
        logging.error(f"❌ Save failed: {e}")
    
    return index, chunks, embeddings, chunk_metadata

# Load or build index
index = None
texts = []
embeddings = np.array([])
chunk_metadata = []

if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
            if len(data) >= 3:
                texts, embeddings = data[0], data[1]
                chunk_metadata = data[2] if len(data) > 2 else []
        
        embeddings = embeddings.astype(np.float32)
        logging.info("✅ Fast index loaded")
    except Exception as e:
        logging.error(f"⚠️ Load failed: {str(e)}. Rebuilding...")
        if embedder:
            index, texts, embeddings, chunk_metadata = build_fast_faiss_index(PDF_FILE)
else:
    if embedder:
        index, texts, embeddings, chunk_metadata = build_fast_faiss_index(PDF_FILE)

# ---------------- ULTRA FAST SEARCH ----------------
@lru_cache(maxsize=1000)  # Reduced cache size
def get_fast_embedding(text: str):
    """Cached embedding generation"""
    if not embedder:
        return np.zeros((1, 1024), dtype=np.float32)
    return embedder.encode([text], convert_to_numpy=True).astype(np.float32)

def fast_botnoi_detection(query: str) -> bool:
    """Fast Botnoi query detection"""
    keywords = fast_extract_keywords(query)
    definite_keywords = ["บอทน้อย", "botnoi", "แพลตฟอร์ม"]
    
    return any(dk in query.lower() for dk in definite_keywords) or len(keywords) >= 2

def should_use_pdf_fast(query: str) -> bool:
    """Fast decision for PDF usage"""
    if not index or index.ntotal == 0:
        return False
    
    if not fast_botnoi_detection(query):
        return False
    
    try:
        q_emb = get_fast_embedding(query)
        D, _ = index.search(q_emb, 1)
        sim_score = 1 - D[0][0] if D.size > 0 and D[0][0] < 2.0 else 0
        return sim_score >= 0.2  # Reduced threshold for speed
    except:
        return False

def search_pdf_ultra_fast(query: str) -> Tuple[List[str], List[int]]:
    """Ultra-fast PDF search"""
    if not index or index.ntotal == 0:
        return [], []
    
    try:
        q_emb = get_fast_embedding(query)
        D, I = index.search(q_emb, min(QUICK_TOP_K, index.ntotal))
        
        valid_indices = I[0][I[0] >= 0]
        if len(valid_indices) == 0:
            return [], []
        
        results = [texts[i] for i in valid_indices]
        return results, valid_indices
        
    except Exception as e:
        logging.error(f"❌ Search error: {e}")
        return [], []

def get_context_ultra_fast(query: str) -> str:
    """Ultra-fast context retrieval"""
    results, indices = search_pdf_ultra_fast(query)
    
    if not results:
        return ""
    
    # Simple context combination
    context = "\n\n".join(results[:QUICK_TOP_K])
    
    # Truncate for speed
    if len(context) > 2000:  # Reduced context size
        context = context[:2000] + "..."
    
    return context

# ---------------- ULTRA FAST IMAGE PROCESSING ----------------
def fast_preprocess_image(image: Image.Image) -> Image.Image:
    """Streamlined image preprocessing"""
    try:
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Simple contrast enhancement
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)  # Reduced from 2.0
        
        # Auto-contrast
        autocontrast = ImageOps.autocontrast(enhanced)
        
        return autocontrast
        
    except Exception as e:
        logging.error(f"❌ Fast preprocessing failed: {e}")
        return image

def extract_text_ultra_fast(image: Image.Image) -> str:
    """Ultra-fast text extraction with timeout"""
    try:
        # Quick OCR attempt
        original_text = pytesseract.image_to_string(
            image, 
            lang='tha+eng', 
            config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzก-ฮ ',
            timeout=FAST_OCR_TIMEOUT
        ).strip()
        
        # If text is too short, try preprocessing
        if len(original_text) < 5:
            processed_image = fast_preprocess_image(image)
            processed_text = pytesseract.image_to_string(
                processed_image, 
                lang='tha+eng', 
                config='--psm 8',
                timeout=FAST_OCR_TIMEOUT
            ).strip()
            
            if len(processed_text) > len(original_text):
                return processed_text
        
        return original_text
        
    except Exception as e:
        logging.error(f"❌ Fast OCR failed: {e}")
        return ""

def analyze_image_ultra_fast(image: Image.Image) -> Dict:
    """Ultra-fast image analysis"""
    try:
        width, height = image.size
        
        # Convert to grayscale for analysis
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Simple analysis
        img_array = np.array(gray)
        pixel_variance = np.var(img_array)
        
        # Quick assessment
        has_text_structure = (
            pixel_variance > 500 and  # Reduced threshold
            width > 100 and height > 100
        )
        
        return {
            'has_text_structure': has_text_structure,
            'complexity': int(pixel_variance),
            'size': (width, height)
        }
        
    except Exception as e:
        logging.error(f"❌ Fast analysis failed: {e}")
        return {'has_text_structure': False, 'complexity': 0, 'size': (0, 0)}

def describe_image_content(image: Image.Image) -> str:
    """Intelligent image content description"""
    try:
        analysis = analyze_image_ultra_fast(image)
        width, height = analysis['size']
        
        # Analyze color characteristics
        if image.mode == 'RGB':
            # Reduce image size for speed
            small_img = image.resize((50, 50))
            img_array = np.array(small_img)
            
            # Find dominant colors
            avg_color = np.mean(img_array, axis=(0, 1))
            
            if avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                color_desc = "สีขาวหรือสีอ่อน"
            elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                color_desc = "สีดำหรือสีเข้ม"
            else:
                color_desc = "สีผสม"
        else:
            color_desc = "ขาวดำ"
        
        # Determine orientation
        if width > height * 1.5:
            orientation = "แนวนอน"
        elif height > width * 1.5:
            orientation = "แนวตั้ง"
        else:
            orientation = "สี่เหลี่ยมจัตุรัส"
        
        # Create description
        description = f"รูปภาพ{orientation} ขนาด {width}x{height} พิกเซล มี{color_desc}"
        
        if analysis['has_text_structure']:
            description += " น่าจะมีข้อความหรือตัวอักษร"
            
            # Try to identify document type
            if width > height and analysis['complexity'] > 1000:
                description += " อาจเป็นเอกสารหรือฟอร์ม"
            elif analysis['complexity'] > 2000:
                description += " อาจเป็นตารางหรือแผนภูมิ"
            else:
                description += " อาจเป็นหน้าจอโปรแกรมหรือเว็บไซต์"
                
            description += "\n\n💡 เคล็ดลับ: ถ่ายรูปให้ชัดขึ้น หรือปรับแสงให้ดีกว่านี้ จะช่วยให้อ่านข้อความได้ดีขึ้น"
        else:
            description += " ดูเหมือนจะเป็นรูปภาพทั่วไป ไม่มีข้อความชัดเจน"
        
        return description
        
    except Exception as e:
        logging.error(f"❌ Image description failed: {e}")
        return "รูปภาพที่ไม่สามารถวิเคราะห์ได้"

# ---------------- ULTRA FAST OLLAMA ----------------
def ask_ollama_ultra_fast(context: str, question: str) -> Tuple[str, float]:
    """Ultra-fast Ollama with optimized parameters"""
    start_time = time.time()
    
    try:
        # Optimized system prompt
        system_prompt = """คุณคือผู้ช่วยตอบคำถาม Botnoi Chatbot แบบรวดเร็ว

กฎ:
1. ตอบสั้น กระชับ ตรงประเด็น
2. ใช้ context ที่ให้มา
3. ถ้าไม่รู้ บอกตรงๆ
4. ไม่เกิน 3 ประโยค"""

        # Ultra-fast Ollama call
        response = ollama.chat(
            model="qwen3:8b-q4_K_M",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQ: {question}"}
            ],
            options={
                "temperature": 0.1,  # Reduced randomness
                "num_predict": 500,  # Reduced length
                "top_p": 0.8,
                "repeat_penalty": 1.1
            }
        )
        
        reply_text = response["message"]["content"].strip()
        
        # Limit response length for speed
        if len(reply_text) > MAX_RESPONSE_LENGTH:
            reply_text = reply_text[:MAX_RESPONSE_LENGTH] + "..."
        
        elapsed = time.time() - start_time
        return reply_text, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"⚠️ Fast AI failed: {str(e)}")
        return FAST_ERROR_RESPONSES["ai_timeout"], elapsed

def make_fast_cache_key(context: str, question: str) -> str:
    """Fast cache key generation"""
    key_str = f"{context[:100]}|{question}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]

def ask_ollama_with_fast_cache(user_state: FastUserState, context: str, question: str) -> Tuple[str, float]:
    """Ultra-fast cached Ollama"""
    key = make_fast_cache_key(context, question)
    
    if key in user_state.ollama_cache:
        logging.info("💾 Fast cache hit")
        user_state.cache_hits += 1
        performance_stats['cache_hits'] += 1
        return user_state.ollama_cache[key], 0
    
    reply_text, elapsed = ask_ollama_ultra_fast(context, question)
    
    # Store in cache with cleanup
    with user_state.lock:
        user_state.ollama_cache[key] = reply_text
        user_state.cleanup_cache()
    
    return reply_text, elapsed

# ---------------- FLASK + LINE ----------------
app = Flask(__name__)
config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
messaging_api = MessagingApi(ApiClient(config))

def reply(token: str, text: str, show_quick_reply: bool = False):
    """Ultra-fast reply with length optimization"""
    if len(text) > MAX_LINE_CHARS:
        text = text[:MAX_LINE_CHARS] + "…"
    
    qr = QUICK_REPLY_ITEMS if show_quick_reply else None
    
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=token,
                messages=[TextMessage(text=str(text), quick_reply=qr)]
            )
        )
    except Exception as e:
        logging.error(f"⚠️ Reply failed: {str(e)}")
        performance_stats['errors'] += 1

# ---------------- ULTRA FAST PROCESSING ----------------
def process_sticker_fast(event, user_id: str, timestamp: str):
    """Ultra-fast sticker processing"""
    user_state = get_user_state(user_id)
    
    # Quick responses for stickers
    quick_sticker_responses = [
        "😊 น่ารักจัง!",
        "👍 เข้าใจแล้ว!",
        "😄 ฮ่าฮ่า!",
        "🤔 อืมม...",
        "👌 โอเค!",
        "😮 โอ้โห!",
        "💪 เจ๋งมาก!",
        "🙏 ขอบคุณครับ!"
    ]
    
    import random
    ai_reply = random.choice(quick_sticker_responses)
    
    with user_state.lock:
        user_state.last_bot_message = ai_reply
    
    user_state.add_to_history(f"[สติ๊กเกอร์] {ai_reply}", timestamp)
    reply(event.reply_token, ai_reply, show_quick_reply=True)

def process_query_ultra_fast(event, user_id: str, msg: str, timestamp: str, request_time: float):
    """Ultra-fast query processing"""
    user_state = get_user_state(user_id)
    start_time = time.time()
    
    # Check for simple commands first
    if is_simple_command(msg):
        if msg.lower().strip() in QUICK_RESPONSES:
            answer = QUICK_RESPONSES[msg.lower().strip()]
            elapsed_total = time.time() - start_time
            
            with user_state.lock:
                user_state.last_bot_message = answer
            
            user_state.add_to_history(answer, timestamp)
            logging.info(f"⚡ QUICK: {elapsed_total:.3f}s | Simple command")
            reply(event.reply_token, answer, show_quick_reply=True)
            return
    
    # Check if should use PDF
    use_pdf = should_use_pdf_fast(msg)
    
    if use_pdf:
        context = get_context_ultra_fast(msg)
        if not context.strip():
            answer = FAST_ERROR_RESPONSES["pdf_not_found"]
            ai_time = 0
        else:
            answer, ai_time = ask_ollama_with_fast_cache(user_state, context, msg)
            performance_stats['pdf_searches'] += 1
    else:
        # General query
        answer, ai_time = ask_ollama_with_fast_cache(user_state, "", msg)
    
    elapsed_total = time.time() - start_time
    
    # Check if this is still the latest request
    with user_state.lock:
        is_latest = (abs(request_time - user_state.last_request_time) < 0.001)
        if not is_latest:
            logging.info(f"⏭️ Skipped outdated response")
            return
    
    # Update state
    with user_state.lock:
        user_state.last_bot_message = answer
        user_state.request_count += 1
    
    user_state.add_to_history(answer, timestamp)
    
    # Update performance stats
    performance_stats['total_requests'] += 1
    performance_stats['avg_response_time'] = (
        (performance_stats['avg_response_time'] * (performance_stats['total_requests'] - 1) + elapsed_total) 
        / performance_stats['total_requests']
    )
    
    logging.info(f"⚡ ULTRA FAST: {elapsed_total:.2f}s | AI: {ai_time:.2f}s | PDF: {use_pdf}")
    
    # Log performance periodically
    if performance_stats['total_requests'] % PERFORMANCE_LOG_INTERVAL == 0:
        logging.info(f"📊 Performance: Avg: {performance_stats['avg_response_time']:.2f}s, "
                    f"Cache hits: {performance_stats['cache_hits']}, "
                    f"PDF searches: {performance_stats['pdf_searches']}, "
                    f"Errors: {performance_stats['errors']}")
    
    reply(event.reply_token, answer, show_quick_reply=True)

def process_image_ultra_fast(event, user_id: str, timestamp: str):
    """Ultra-fast image processing with intelligent interpretation"""
    try:
        message_id = event.message.id
        
        # Download image
        with ApiClient(config) as api_client:
            blob_api = MessagingApiBlob(api_client)
            image_bytes = blob_api.get_message_content(message_id)
        
        image = Image.open(BytesIO(image_bytes))
        
        # Quick image analysis
        analysis = analyze_image_ultra_fast(image)
        
        # Try OCR first
        extracted_text = extract_text_ultra_fast(image)
        
        if not extracted_text or len(extracted_text) < 5:
            # Fallback to visual interpretation
            if analysis['has_text_structure']:
                description = describe_image_content(image)
                reply(event.reply_token, f"🔍 ไม่สามารถอ่านข้อความได้ชัดเจน แต่วิเคราะห์ได้ว่า:\n\n{description}")
            else:
                description = describe_image_content(image)
                reply(event.reply_token, f"🖼️ {description}")
            
            performance_stats['image_processes'] += 1
            return
        
        logging.info(f"🔤 OCR result: {extracted_text[:100]}...")
        
        # Send processing message
        try:
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text="⏳ อ่านข้อความจากรูปได้แล้ว กำลังประมวลผล...")]
                )
            )
        except Exception as e:
            logging.error(f"⚠️ Failed to send processing message: {str(e)}")
        
        # Process extracted text
        current_time = time.time()
        user_state = get_user_state(user_id)
        
        with user_state.lock:
            user_state.last_request_time = current_time
        
        user_state.add_to_history(f"[รูปภาพ] {extracted_text}", timestamp)
        
        # Process as text query
        executor.submit(process_query_ultra_fast, event, user_id, extracted_text, timestamp, current_time)
        performance_stats['image_processes'] += 1
        
    except Exception as e:
        logging.error(f"⚠️ Image processing failed: {str(e)}", exc_info=True)
        reply(event.reply_token, FAST_ERROR_RESPONSES["general_error"])
        performance_stats['errors'] += 1

# ---------------- TRANSLATION FUNCTIONS ----------------
def translate_text_fast(text: str, target_lang: str) -> str:
    """Fast translation with caching"""
    try:
        if not text or not isinstance(text, str):
            return ""
        
        # Limit text length for speed
        if len(text) > MAX_LINE_CHARS:
            text = text[:MAX_LINE_CHARS] + "…"
        
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        
        if not translated or len(translated.strip()) == 0:
            return ""
        
        return translated
        
    except Exception as e:
        logging.error(f"⚠️ Translation failed: {str(e)}")
        return f"❌ แปลไม่สำเร็จ: {str(e)[:50]}..."

def translate_and_reply_fast(event, user_state: FastUserState, lang_code: str, flag: str):
    """Fast translation and reply"""
    with user_state.lock:
        last_msg = user_state.last_bot_message
    
    if not last_msg:
        reply(event.reply_token, "❌ ไม่มีข้อความล่าสุดสำหรับแปล")
        return
    
    trans_text = translate_text_fast(last_msg, lang_code)
    if not trans_text:
        reply(event.reply_token, "❌ แปลภาษาไม่สำเร็จ")
        return
    
    reply(event.reply_token, f"{flag} {trans_text}")

def translate_all_languages_fast(event, user_state: FastUserState):
    """Fast translation to all languages"""
    with user_state.lock:
        last_msg = user_state.last_bot_message
    
    if not last_msg:
        reply(event.reply_token, "❌ ไม่มีข้อความล่าสุดสำหรับแปล")
        return
    
    translations = []
    for code, (label, lang_code) in LANG_BUTTONS.items():
        trans_text = translate_text_fast(last_msg, lang_code)
        if trans_text:
            flag = label.split()[0]
            translations.append(f"{flag} {trans_text}")
    
    if translations:
        reply(event.reply_token, "\n\n".join(translations))
    else:
        reply(event.reply_token, "❌ แปลภาษาล้มเหลวทุกภาษา")

# ---------------- MESSAGE HANDLER ----------------
def handle_message(event):
    """Ultra-fast message handler with request queuing"""
    user_id = event.source.user_id
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(f"📨 Message from User {user_id}")
    
    # Handle stickers
    if isinstance(event.message, StickerMessageContent):
        executor.submit(process_sticker_fast, event, user_id, timestamp)
        return
    
    # Handle images
    if isinstance(event.message, ImageMessageContent):
        logging.info("🖼️ Image received. Starting ultra-fast processing...")
        executor.submit(process_image_ultra_fast, event, user_id, timestamp)
        return
    
    # Handle text messages
    if not hasattr(event.message, "text"):
        return
    
    msg = event.message.text.strip()
    user_state = get_user_state(user_id)
    current_time = time.time()
    
    with user_state.lock:
        user_state.last_request_time = current_time
    
    user_state.add_to_history(msg, timestamp)
    
    logging.info(f"💬 Text: {msg}")
    lower_msg = msg.lower()
    
    # Handle translation commands
    translation_commands = {
        "/en_last": ("en", "🇺🇸"),
        "/jp_last": ("ja", "🇯🇵"),
        "/cn_last": ("zh-CN", "🇨🇳"),
        "/kr_last": ("ko", "🇰🇷"),
        "/fr_last": ("fr", "🇫🇷"),
        "/de_last": ("de", "🇩🇪"),
        "/vi_last": ("vi", "🇻🇳")
    }
    
    for cmd, (lang_code, flag) in translation_commands.items():
        if lower_msg.startswith(cmd):
            translate_and_reply_fast(event, user_state, lang_code, flag)
            return
    
    if lower_msg == "/translate_all":
        translate_all_languages_fast(event, user_state)
        return
    
    # Handle basic commands
    if lower_msg in ["hi", "hello", "bot", "สวัสดี"]:
        reply(event.reply_token, 
              "👋 สวัสดีครับ! ผมเป็นผู้ช่วยตอบคำถามจากคู่มือ Botnoi Chatbot "
              "หรือพูดคุยทั่วไปได้ครับ\n\n💡 พิมพ์ 'help' เพื่อดูคำสั่งทั้งหมด", 
              show_quick_reply=False)
        return
    
    if lower_msg == "help":
        help_text = """🤖 คำสั่งที่ใช้ได้:

📚 การใช้งาน:
• ถามเกี่ยวกับ Botnoi Chatbot
• ส่งรูปภาพเพื่ออ่านข้อความ
• ส่งสติ๊กเกอร์เพื่อสนทนา

🔧 คำสั่งพิเศษ:
• 'id' - ดู User ID
• 'clear' - ล้างประวัติ
• 'stats' - ดูสถิติการใช้งาน

🌐 การแปลภาษา:
• กดปุ่มด่วนหลังได้รับคำตอบ
• หรือพิมพ์ '/translate_all'"""
        reply(event.reply_token, help_text, show_quick_reply=False)
        return
    
    # Other commands
    if lower_msg == "id":
        reply(event.reply_token, f"🔑 User ID: {user_id}", show_quick_reply=False)
        return
    
    if lower_msg == "clear":
        with user_state.lock:
            user_state.history = []
            user_state.last_bot_message = None
            user_state.ollama_cache = {}
            user_state.last_request_time = 0
            user_state.request_count = 0
            user_state.cache_hits = 0
        reply(event.reply_token, "🧹 ล้างประวัติและ cache เรียบร้อย", show_quick_reply=False)
        return
    
    if lower_msg == "stats":
        stats_text = f"""📊 สถิติการใช้งาน:

⚡ ประสิทธิภาพ:
• คำขอทั้งหมด: {performance_stats['total_requests']}
• เวลาตอบเฉลี่ย: {performance_stats['avg_response_time']:.2f}s
• Cache hits: {performance_stats['cache_hits']}

🔍 การประมวลผล:
• ค้นหา PDF: {performance_stats['pdf_searches']}
• ประมวลผลรูป: {performance_stats['image_processes']}
• ข้อผิดพลาด: {performance_stats['errors']}

👤 ผู้ใช้นี้:
• คำขอ: {user_state.request_count}
• Cache hits: {user_state.cache_hits}"""
        reply(event.reply_token, stats_text, show_quick_reply=False)
        return
    
    # Send processing message for complex queries
    try:
        messaging_api.push_message(
            PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text="⏳ กำลังประมวลผล...")]
            )
        )
    except Exception as e:
        logging.error(f"⚠️ Failed to send processing message: {str(e)}")
    
    # Process query
    executor.submit(process_query_ultra_fast, event, user_id, msg, timestamp, current_time)

# ---------------- WEBHOOK ----------------
@app.route("/", methods=["POST"])
def callback():
    """Ultra-fast webhook callback with request queuing"""
    body = request.get_data(as_text=True)
    logging.info("🌐 Webhook received")
    
    try:
        callback_request = CallbackRequest.from_json(body)
    except Exception as e:
        logging.error(f"⚠️ Webhook parse error: {str(e)}")
        performance_stats['errors'] += 1
        return "Bad request", 400
    
    for event in callback_request.events:
        if isinstance(event, MessageEvent):
            try:
                # Try to queue the request
                request_queue.put_nowait((handle_message, event))
            except queue.Full:
                # If queue is full, process directly
                logging.warning("⚠️ Request queue full, processing directly")
                executor.submit(handle_message, event)
    
    return "OK", 200

# ---------------- REQUEST QUEUE PROCESSOR ----------------
def process_request_queue():
    """Process queued requests"""
    while True:
        try:
            handler, event = request_queue.get(timeout=1)
            executor.submit(handler, event)
            request_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"⚠️ Queue processing error: {e}")

# Start queue processor thread
queue_thread = threading.Thread(target=process_request_queue, daemon=True)
queue_thread.start()

# ---------------- GRACEFUL SHUTDOWN ----------------
def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    logging.info("🛑 Shutting down gracefully...")
    executor.shutdown(wait=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logging.info("🚀 Starting ULTRA FAST Botnoi Chatbot server...")
    
    # Check dependencies
    if not embedder:
        logging.warning("⚠️ Embedding model not loaded. PDF search will be limited.")
    
    if not os.path.exists(PDF_FILE):
        logging.warning(f"⚠️ PDF file not found: {PDF_FILE}")
    
    # Warm up Ollama
    try:
        logging.info("🧠 Warming up Ollama...")
        ollama.chat(
            model="qwen3:8b-q4_K_M", 
            messages=[{"role": "user", "content": "สวัสดี"}],
            options={"temperature": 0.1, "num_predict": 50}
        )
        logging.info("✅ Ollama ready")
    except Exception as e:
        logging.error(f"⚠️ Ollama warm-up failed: {str(e)}")
    
    # Pre-populate quick responses cache
    logging.info("💾 Pre-populating response cache...")
    
    logging.info("✅ ULTRA FAST Server ready on port 5000")
    logging.info(f"📊 Configuration: Workers={MAX_WORKERS}, Cache={CACHE_SIZE}, "
                f"ChunkSize={QUICK_CHUNK_SIZE}, TopK={QUICK_TOP_K}")
    
    app.run(port=5000, debug=False, use_reloader=False, threaded=True)