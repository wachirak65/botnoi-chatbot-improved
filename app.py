import os
import faiss
import pickle
import fitz  # PyMuPDF
import numpy as np
import re
import time
import logging
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
from PIL import Image
from io import BytesIO
from PIL import ImageEnhance, ImageFilter, ImageOps
import base64

# ---------------- CONFIG ----------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN_HERE")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET_HERE")

# File paths - adjust these to your setup
PDF_FILE = os.path.join("data", "Using_Platform_Botnoi.pdf")
INDEX_FILE = os.path.join("data", "Using_Platform_Botnoi.index")
DATA_FILE = os.path.join("data", "Using_Platform_Botnoi.pkl")

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
MAX_LINE_CHARS = 4900

# ⚡ ULTRA FAST SETTINGS
FAST_OCR_TIMEOUT = 3  # OCR timeout in seconds
FAST_AI_TIMEOUT = 8   # AI response timeout in seconds
MAX_RESPONSE_LENGTH = 800  # Limit response length for speed

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(PDF_FILE), exist_ok=True)

# Quick responses for common queries
QUICK_RESPONSES = {
    "สวัสดี": "👋 สวัสดีครับ! ผมพร้อมช่วยเหลือเรื่อง Botnoi Chatbot",
    "hello": "👋 Hello! I'm ready to help with Botnoi Chatbot questions",
    "hi": "👋 Hi there! How can I help you with Botnoi today?",
    "ขอบคุณ": "🙏 ยินดีครับ! มีอะไรให้ช่วยอีกไหม?",
    "thank you": "🙏 You're welcome! Anything else I can help with?",
    "bye": "👋 ลาก่อนครับ! หวังว่าจะได้ช่วยเหลือคุณอีก",
    "ลาก่อน": "👋 ลาก่อนครับ! ขอบคุณที่ใช้บริการ"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- TESSERACT SETUP ----------------
if os.name == 'nt':
    # Common Tesseract installation paths on Windows
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

# Initialize models
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logging.info("✅ Embedding model loaded successfully")
except Exception as e:
    logging.error(f"❌ Failed to load embedding model: {e}")
    embedder = None

executor = ThreadPoolExecutor(max_workers=5)  # Reduced workers for speed

# Language mapping for Quick Reply
LANG_BUTTONS = {
    "en": ("🇺🇸 อังกฤษ", "en"),
    "jp": ("🇯🇵 ญี่ปุ่น", "ja"),
    "cn": ("🇨🇳 จีน", "zh-CN"),
    "kr": ("🇰🇷 เกาหลี", "ko"),
    "fr": ("🇫🇷 ฝรั่งเศส", "fr"),
    "de": ("🇩🇪 เยอรมัน", "de"),
    "vi": ("🇻🇳 เวียดนาม", "vi"),
}

# Quick Reply buttons
QUICK_REPLY_ITEMS = QuickReply(items=[
    QuickReplyItem(action=MessageAction(label=label, text=f"/{code}_last")) 
    for code, (label, _) in LANG_BUTTONS.items()
] + [
    QuickReplyItem(action=MessageAction(label="🔄 แปลทุกภาษา", text="/translate_all")),
])

# ---------------- USER STATE ----------------
class UserState:
    def __init__(self):
        self.history = []
        self.last_bot_message = None
        self.ollama_cache = {}
        self.last_request_time = 0
        self.lock = threading.Lock()

users = {}
users_lock = threading.Lock()

def get_user_state(user_id):
    with users_lock:
        if user_id not in users:
            users[user_id] = UserState()
        return users[user_id]

# ---------------- IMPROVED TEXT PROCESSING ----------------
def clean_text(text):
    """Improved text cleaning"""
    if not text:
        return ""
    
    # Keep Thai, English, numbers and important punctuation
    cleaned = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,!?()/-:;"\'\n\r]', '', text)
    
    # Remove excessive whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    return cleaned.strip()

def extract_keywords(text):
    """Extract keywords from text"""
    keywords = []
    
    # Botnoi-related keywords
    botnoi_keywords = [
        "บอทน้อย", "botnoi", "แพลตฟอร์ม", "platform", "แชทบอท", "chatbot",
        "สร้างบอท", "create bot", "ตั้งค่า", "setting", "configuration",
        "ใช้งาน", "usage", "คู่มือ", "manual", "guide", "tutorial"
    ]
    
    text_lower = text.lower()
    for keyword in botnoi_keywords:
        if keyword in text_lower:
            keywords.append(keyword)
    
    return keywords

# ---------------- IMPROVED PDF PROCESSING ----------------
def build_faiss_index(pdf_path, chunk_size=200, overlap=50):
    """Build improved FAISS index with overlapping chunks"""
    logging.info("📄 Building improved FAISS HNSW Index...")
    
    if not os.path.exists(pdf_path):
        logging.error(f"❌ PDF file not found: {pdf_path}")
        dim = 1024  # Default dimension
        empty_index = faiss.IndexHNSWFlat(dim, 32)
        return empty_index, [], np.zeros((0, dim), dtype=np.float32)
    
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_metadata = []
    
    for page_num, page in enumerate(doc):
        text = clean_text(page.get_text("text"))
        if not text.strip():
            continue
            
        words = text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 10:  # Skip very short chunks
                continue
                
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)
                chunk_metadata.append({
                    'page': page_num + 1,
                    'start_word': i,
                    'keywords': extract_keywords(chunk)
                })
    
    if not chunks:
        logging.warning("⚠️ No text chunks found in PDF")
        dim = 1024
        empty_index = faiss.IndexHNSWFlat(dim, 32)
        return empty_index, [], np.zeros((0, dim), dtype=np.float32)
    
    # Create embeddings
    try:
        embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)
    except Exception as e:
        logging.error(f"❌ Failed to create embeddings: {e}")
        return None, [], np.array([])
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200  # Improve search quality
    index.add(embeddings)
    
    # Save files
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(DATA_FILE, "wb") as f:
            pickle.dump((chunks, embeddings, chunk_metadata), f)
        logging.info(f"✅ FAISS index built: {len(chunks)} chunks")
    except Exception as e:
        logging.error(f"❌ Failed to save index: {e}")
    
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
            if len(data) == 4:
                texts, embeddings, chunk_metadata = data[0], data[1], data[3]
            else:
                texts, embeddings = data[0], data[1]
                chunk_metadata = [{'page': 1, 'keywords': []} for _ in texts]
        
        embeddings = embeddings.astype(np.float32)
        logging.info("✅ Loaded FAISS index successfully")
    except Exception as e:
        logging.error(f"⚠️ Failed loading FAISS: {str(e)}. Rebuilding...")
        if embedder:
            index, texts, embeddings, chunk_metadata = build_faiss_index(PDF_FILE)
else:
    if embedder:
        index, texts, embeddings, chunk_metadata = build_faiss_index(PDF_FILE)

# ---------------- IMPROVED SEARCH ----------------
@lru_cache(maxsize=5000)
def get_embedding_cached(text: str):
    if not embedder:
        return np.zeros((1, 1024), dtype=np.float32)
    return embedder.encode([text], convert_to_numpy=True).astype(np.float32)

def get_embedding(text):
    return get_embedding_cached(text.strip().lower())

def improved_botnoi_detection(query):
    """Improved Botnoi query detection"""
    keywords = extract_keywords(query)
    
    # Definite Botnoi keywords
    definite_keywords = ["บอทน้อย", "botnoi", "แพลตฟอร์ม", "platform"]
    
    # Possible related keywords
    possible_keywords = ["แชทบอท", "chatbot", "สร้างบอท", "ตั้งค่า", "ใช้งาน"]
    
    definite_score = sum(1 for k in keywords if any(dk in k.lower() for dk in definite_keywords))
    possible_score = sum(1 for k in keywords if any(pk in k.lower() for pk in possible_keywords))
    
    return definite_score > 0 or possible_score >= 2

def should_use_pdf(query, threshold=0.25):
    """Improved PDF usage decision"""
    if not index or index.ntotal == 0:
        return False
    
    if not improved_botnoi_detection(query):
        return False
    
    try:
        q_emb = get_embedding(query)
        D, _ = index.search(q_emb, 1)
        sim_score = 1 - D[0][0] if D.size > 0 and D[0][0] < 2.0 else 0
        
        logging.info(f"📊 PDF similarity score: {sim_score:.3f}")
        return sim_score >= threshold
    except Exception as e:
        logging.error(f"❌ Error in PDF similarity check: {e}")
        return False

def search_pdf_improved(query, k=6):
    """Improved PDF search"""
    if not index or index.ntotal == 0:
        return [], np.array([], dtype=int), []
    
    try:
        q_emb = get_embedding(query)
        
        # Search with FAISS
        D, I = index.search(q_emb, min(k * 3, index.ntotal))
        
        # Calculate additional cosine similarity
        valid_indices = I[0][I[0] >= 0]
        if len(valid_indices) == 0:
            return [], np.array([]), []
        
        valid_embeddings = embeddings[valid_indices]
        sims = np.dot(valid_embeddings, q_emb[0].T)
        
        # Sort and select top-k
        sorted_idx = np.argsort(sims)[::-1][:k]
        final_indices = valid_indices[sorted_idx]
        
        results = [texts[i] for i in final_indices]
        metadata = [chunk_metadata[i] if i < len(chunk_metadata) else {} for i in final_indices]
        
        return results, final_indices, metadata
        
    except Exception as e:
        logging.error(f"❌ Error in PDF search: {e}")
        return [], np.array([]), []

def get_context_from_pdf(query, top_k=5, max_len=3500):
    """Improved context extraction from PDF"""
    results, indices, metadata = search_pdf_improved(query, top_k)
    
    if not results:
        return "", []
    
    # Combine context with metadata
    context_parts = []
    for i, (result, meta) in enumerate(zip(results, metadata)):
        page_info = f"[หน้า {meta.get('page', '?')}]" if meta else ""
        context_parts.append(f"{page_info} {result}")
    
    context = "\n\n".join(context_parts)
    
    # Truncate to fit size
    if len(context) > max_len:
        context = context[:max_len] + "..."
    
    logging.info(f"🔎 Retrieved {len(results)} chunks from PDF")
    return context, indices

# ---------------- IMPROVED IMAGE PROCESSING (PIL-based) ----------------
def preprocess_image(image):
    """Improve image before OCR with PIL"""
    try:
        # Convert to grayscale
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image.copy()
        
        # 1. Contrast enhancement
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced = enhancer.enhance(2.0)
        
        # 2. Sharpness enhancement
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = sharpness_enhancer.enhance(2.0)
        
        # 3. Noise reduction with filter
        filtered = sharpened.filter(ImageFilter.MedianFilter(size=3))
        
        # 4. Auto-level (adjust histogram)
        autocontrast = ImageOps.autocontrast(filtered)
        
        # 5. Threshold for binary image
        img_array = np.array(autocontrast)
        
        # Simple Otsu's method
        threshold = np.mean(img_array)
        binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        
        # Convert back to PIL Image
        binary_image = Image.fromarray(binary_array, mode='L')
        
        return binary_image
        
    except Exception as e:
        logging.error(f"❌ Image preprocessing failed: {e}")
        return image

def extract_text_from_image(image):
    """Improved text extraction from image"""
    try:
        # Try OCR with original image first
        original_text = pytesseract.image_to_string(
            image, 
            lang='tha+eng', 
            config='--psm 6',
            timeout=FAST_OCR_TIMEOUT
        ).strip()
        
        # If little text found, try with processed image
        if len(original_text) < 10:
            processed_image = preprocess_image(image)
            processed_text = pytesseract.image_to_string(
                processed_image, 
                lang='tha+eng', 
                config='--psm 6',
                timeout=FAST_OCR_TIMEOUT
            ).strip()
            
            # Choose better result
            if len(processed_text) > len(original_text):
                return processed_text
        
        return original_text
        
    except Exception as e:
        logging.error(f"❌ OCR failed: {e}")
        return ""

def describe_image_with_ai(image):
    """Use AI to interpret image when OCR fails"""
    try:
        analysis = analyze_image_content(image)
        width, height = analysis['size']
        
        # Create basic description
        if width > height * 1.5:
            orientation = "แนวนอน"
        elif height > width * 1.5:
            orientation = "แนวตั้ง"
        else:
            orientation = "สี่เหลี่ยมจัตุรัส"
        
        # Analyze main colors
        if image.mode == 'RGB':
            small_img = image.resize((50, 50))
            img_array = np.array(small_img)
            avg_color = np.mean(img_array, axis=(0, 1))
            
            if avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                color_desc = "สีขาวหรือสีอ่อน"
            elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                color_desc = "สีดำหรือสีเข้ม"
            else:
                color_desc = "สีผสม"
        else:
            color_desc = "ขาวดำ"
        
        # Create basic description
        base_description = f"รูปภาพ{orientation} ขนาด {width}x{height} พิกเซล มี{color_desc}"
        
        # Use AI to interpret content
        ai_prompt = f"""วิเคราะห์รูปภาพนี้และอธิบายสิ่งที่เห็น:

ข้อมูลพื้นฐาน: {base_description}
ความซับซ้อน: {analysis['complexity']}
มีโครงสร้างข้อความ: {'ใช่' if analysis['has_text_structure'] else 'ไม่'}

กรุณาอธิบายว่าในรูปนี้น่าจะมีอะไร เช่น:
- เอกสาร, ฟอร์ม, ตาราง
- หน้าจอโปรแกรม, เว็บไซต์, แอป
- แผนภูมิ, กราฟ, ไดอะแกรม
- รูปภาพทั่วไป, ภาพถ่าย

ตอบแค่ 2-3 ประโยคสั้นๆ และให้คำแนะนำถ้าต้องการอ่านข้อความ"""

        ai_description, _ = ask_ollama_improved("", ai_prompt)
        
        return f"🔍 วิเคราะห์รูปภาพ:\n\n{ai_description}"
        
    except Exception as e:
        logging.error(f"❌ AI image description failed: {e}")
        return f"🖼️ รูปภาพ{orientation} ขนาด {width}x{height} พิกเซล - ไม่สามารถอ่านข้อความได้ชัดเจน"

def analyze_image_content(image):
    """Analyze image content with PIL"""
    try:
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Convert to numpy array
        img_array = np.array(gray)
        
        # Analyze image complexity
        # 1. Check pixel value variance
        pixel_variance = np.var(img_array)
        
        # 2. Check histogram distribution
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        hist_peaks = len([i for i in range(1, len(hist)-1) 
                         if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 10])
        
        # 3. Check edge density (simple)
        # Simple Sobel filter
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution (simple)
        height, width = img_array.shape
        edges = np.zeros_like(img_array)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                gx = np.sum(img_array[i-1:i+2, j-1:j+2] * sobel_x)
                gy = np.sum(img_array[i-1:i+2, j-1:j+2] * sobel_y)
                edges[i, j] = min(255, int(np.sqrt(gx**2 + gy**2)))
        
        edge_density = np.sum(edges > 50) / (height * width)
        
        # Assess if likely to have text
        has_text_structure = (
            pixel_variance > 1000 and  # Sufficient variance
            hist_peaks > 2 and        # Multiple histogram peaks
            edge_density > 0.05       # Sufficient edges
        )
        
        analysis = {
            'has_text_structure': has_text_structure,
            'complexity': int(pixel_variance + hist_peaks * 100 + edge_density * 1000),
            'size': image.size,
            'pixel_variance': pixel_variance,
            'hist_peaks': hist_peaks,
            'edge_density': edge_density
        }
        
        return analysis
        
    except Exception as e:
        logging.error(f"❌ Image analysis failed: {e}")
        return {'has_text_structure': False, 'complexity': 0, 'size': (0, 0)}

# ---------------- IMPROVED TRANSLATION ----------------
def translate_text(text, target_lang):
    """Improved translation"""
    try:
        if not text or not isinstance(text, str):
            return ""
        
        # Truncate if too long
        original_len = len(text)
        if original_len > MAX_LINE_CHARS:
            text = text[:MAX_LINE_CHARS] + "…"
        
        # Translate
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        
        if not translated or len(translated.strip()) == 0:
            return ""
        
        return translated
        
    except Exception as e:
        logging.error(f"⚠️ Translation failed: {str(e)}")
        return f"❌ แปลไม่สำเร็จ: {str(e)[:50]}..."

# ---------------- IMPROVED OLLAMA ----------------
def ask_ollama_improved(context, question):
    """Improved Ollama usage"""
    start_time = time.time()
    
    try:
        # Improved system prompt
        system_prompt = """คุณคือผู้เชี่ยวชาญด้านการใช้งานแพลตฟอร์ม Botnoi Chatbot 

กติกาการตอบ:
1. หากคำถามเกี่ยวข้องกับ Botnoi/แชทบอท:
   - ตอบตาม context ที่ให้มาอย่างละเอียดและครบถ้วน
   - แบ่งเป็นขั้นตอนชัดเจน ใช้หัวข้อย่อย
   - ให้ตัวอย่างการใช้งานถ้าเป็นไปได้

2. หากคำถามทั่วไป:
   - ตอบด้วยข้อมูลที่ถูกต้องและเป็นประโยชน์
   - ใช้ภาษาที่เข้าใจง่าย

3. หลักการสำคัญ:
   - ห้ามเดาหรือสร้างข้อมูลที่ไม่แน่ใจ
   - ถ้าไม่รู้ ให้บอกตรงๆ ว่าไม่มีข้อมูล
   - ใช้ภาษาพูดเป็นกันเอง แต่มีประโยชน์
   - ตอบสั้น กระชับ ไม่เกิน 3 ย่อหน้า"""

        # Call Ollama
        response = ollama.chat(
            model="qwen3:8b-q4_K_M",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nคำถาม: {question}"}
            ],
            options={
                "temperature": 0.1,  # Reduce randomness for speed
                "num_predict": 600,  # Reduce length for speed
                "top_p": 0.8,
                "repeat_penalty": 1.1
            }
        )
        
        reply_text = response["message"]["content"].strip()
        
        # Limit response length
        if len(reply_text) > MAX_RESPONSE_LENGTH:
            reply_text = reply_text[:MAX_RESPONSE_LENGTH] + "..."
        
        elapsed = time.time() - start_time
        
        return reply_text, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"⚠️ AI response failed: {str(e)}")
        return "❌ ระบบ AI ขัดข้อง กรุณาลองใหม่อีกครั้ง", elapsed

def make_cache_key(context, question):
    """Create cache key"""
    import hashlib
    key_str = f"{hashlib.md5(context.encode()).hexdigest()}|{question.strip()}"
    return key_str[:128]

def ask_ollama_with_cache(user_state, context, question):
    """Use cache for Ollama"""
    key = make_cache_key(context, question)
    
    if key in user_state.ollama_cache:
        logging.info("💾 Using cached AI response")
        return user_state.ollama_cache[key], 0
    
    reply_text, elapsed = ask_ollama_improved(context, question)
    
    # Store in cache
    with user_state.lock:
        # Limit cache size
        if len(user_state.ollama_cache) > 50:
            # Remove old cache entries
            old_keys = list(user_state.ollama_cache.keys())[:10]
            for old_key in old_keys:
                del user_state.ollama_cache[old_key]
        
        user_state.ollama_cache[key] = reply_text
    
    return reply_text, elapsed

# ---------------- FLASK + LINE ----------------
app = Flask(__name__)
config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
messaging_api = MessagingApi(ApiClient(config))

def reply(token, text, show_quick_reply=False):
    """Send reply message"""
    if len(text) > MAX_LINE_CHARS:
        cut_point = text.rfind(' ', 0, MAX_LINE_CHARS)
        if cut_point == -1:
            cut_point = MAX_LINE_CHARS
        text = text[:cut_point] + "…"
    
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

# ---------------- IMPROVED PROCESSING FUNCTIONS ----------------
def process_sticker(event, user_id, timestamp, package_id, sticker_id):
    """Process sticker with appropriate emotion"""
    user_state = get_user_state(user_id)
    
    # Quick responses for popular stickers
    quick_sticker_responses = [
        "😊 น่ารักจัง!",
        "👍 เข้าใจแล้ว!",
        "😄 ฮ่าฮ่า!",
        "🤔 อืมม...",
        "👌 โอเค!",
        "😮 โอ้โห!",
        "💪 เจ๋งมาก!",
        "🙏 ขอบคุณครับ!",
        "😍 สวยมาก!",
        "🎉 ยินดีด้วย!",
        "😢 เสียใจด้วยนะ",
        "😴 ง่วงแล้วเหรอ?",
        "🔥 เจ๋งสุดๆ!",
        "❤️ รักเลย!",
        "😂 ขำจัง!"
    ]
    
    # Use AI for more complex responses
    prompt = f"""คุณเป็นเพื่อนที่กำลังแชทกับคนสนิท เขาเพิ่งส่งสติ๊กเกอร์มา (package: {package_id}, sticker: {sticker_id})

ให้ตอบกลับด้วยอารมณ์ที่เหมาะสม:
- ถ้าเป็นสติ๊กเกอร์แสดงความยินดี → ตอบด้วยความยินดี
- ถ้าเป็นสติ๊กเกอร์เศร้า → ตอบด้วยความเข้าใจ
- ถ้าเป็นสติ๊กเกอร์ตลก → ตอบด้วยความสนุก
- ถ้าเป็นสติ๊กเกอร์น่ารัก → ตอบด้วยความชื่นชม

กฎสำคัญ:
1. ห้ามพูดถึงรหัสสติ๊กเกอร์
2. ตอบแค่ 1 ประโยคสั้นๆ
3. ใช้ภาษาพูดธรรมชาติ
4. แสดงอารมณ์ที่ตรงกับสติ๊กเกอร์"""

    try:
        # Try AI first, but use quick response if too slow
        import random
        if random.random() < 0.3:  # 30% use quick response for speed
            ai_reply = random.choice(quick_sticker_responses)
        else:
            ai_reply, elapsed = ask_ollama_improved("", prompt)
            if elapsed > 2:  # If slower than 2 seconds, use quick response
                ai_reply = random.choice(quick_sticker_responses)
    except:
        import random
        ai_reply = random.choice(quick_sticker_responses)
    
    with user_state.lock:
        user_state.last_bot_message = ai_reply
        user_state.history.append({
            "time": timestamp, 
            "message": f"[สติ๊กเกอร์] {ai_reply}"
        })
    
    reply(event.reply_token, ai_reply, show_quick_reply=True)

def process_query(event, user_id, msg, timestamp, request_time):
    """Process query - improved"""
    user_state = get_user_state(user_id)
    
    # Check quick responses first
    msg_lower = msg.lower().strip()
    if msg_lower in QUICK_RESPONSES:
        answer = QUICK_RESPONSES[msg_lower]
        
        with user_state.lock:
            user_state.last_bot_message = answer
            user_state.history.append({"time": timestamp, "message": answer})
        
        logging.info(f"⚡ QUICK RESPONSE: {msg_lower}")
        reply(event.reply_token, answer, show_quick_reply=True)
        return
    
    # Check if should use PDF
    use_pdf = should_use_pdf(msg)
    
    start_time = time.time()
    
    if use_pdf:
        context, indices = get_context_from_pdf(msg, top_k=5, max_len=2500)  # Reduced size for speed
        if not context.strip():
            answer = "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องในคู่มือ Botnoi กรุณาลองถามในแบบอื่น"
            ai_time = 0
        else:
            answer, ai_time = ask_ollama_with_cache(user_state, context, msg)
    else:
        # General questions
        answer, ai_time = ask_ollama_with_cache(user_state, "", msg)
    
    elapsed_total = time.time() - start_time
    
    # Check if this is the latest request
    with user_state.lock:
        is_latest = (abs(request_time - user_state.last_request_time) < 0.001)
        if not is_latest:
            logging.info(f"⏭️ Skipped outdated response for user {user_id}")
            return
    
    # Save history
    with user_state.lock:
        user_state.last_bot_message = answer
        user_state.history.append({"time": timestamp, "message": answer})
    
    logging.info(f"⚡ ULTRA FAST: {elapsed_total:.2f}s | AI: {ai_time:.2f}s | PDF: {use_pdf}")
    reply(event.reply_token, answer, show_quick_reply=True)

def process_image(event, user_id, timestamp):
    """Process image - improved with AI interpretation"""
    try:
        message_id = event.message.id
        
        # Download image
        with ApiClient(config) as api_client:
            blob_api = MessagingApiBlob(api_client)
            image_bytes = blob_api.get_message_content(message_id)
        
        image = Image.open(BytesIO(image_bytes))
        
        # Analyze image
        analysis = analyze_image_content(image)
        
        # Extract text
        extracted_text = extract_text_from_image(image)
        
        if not extracted_text or len(extracted_text) < 5:
            # Use AI to interpret image instead
            logging.info("🤖 Using AI to interpret image content...")
            
            try:
                ai_description = describe_image_with_ai(image)
                reply(event.reply_token, ai_description)
                
                # Save history
                user_state = get_user_state(user_id)
                user_state.history.append({
                    "time": timestamp, 
                    "message": f"[รูปภาพ - AI ตีความ] {ai_description}"
                })
                
            except Exception as ai_error:
                logging.error(f"❌ AI image interpretation failed: {ai_error}")
                
                # Fallback if AI fails
                if analysis['has_text_structure']:
                    fallback_msg = "🔍 พบโครงสร้างข้อความในรูป แต่ไม่สามารถอ่านได้ชัดเจน\n\n💡 เคล็ดลับ: ลองถ่ายรูปใหม่ให้ชัดขึ้น หรือปรับแสงให้ดีกว่านี้"
                else:
                    fallback_msg = "🖼️ รูปภาพนี้ไม่มีข้อความชัดเจน\n\nหากต้องการให้ช่วยอะไรเกี่ยวกับรูปนี้ กรุณาอธิบายเพิ่มเติมครับ"
                
                reply(event.reply_token, fallback_msg)
            
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
            user_state.history.append({
                "time": timestamp, 
                "message": f"[รูปภาพ] {extracted_text}"
            })
        
        executor.submit(process_query, event, user_id, extracted_text, timestamp, current_time)
        
    except Exception as e:
        logging.error(f"⚠️ Image processing failed: {str(e)}", exc_info=True)
        reply(event.reply_token, "❌ ประมวลผลรูปภาพไม่สำเร็จ กรุณาลองใหม่อีกครั้ง")

# ---------------- TRANSLATION FUNCTIONS ----------------
def translate_and_reply(event, user_state, lang_code, flag):
    """Translate latest message"""
    with user_state.lock:
        last_msg = user_state.last_bot_message
        if not last_msg:
            reply(event.reply_token, "❌ ไม่มีข้อความล่าสุดสำหรับแปล")
            return
    
    trans_text = translate_text(last_msg, lang_code)
    if not trans_text:
        reply(event.reply_token, "❌ แปลภาษาไม่สำเร็จ")
        return
    
    reply(event.reply_token, f"{flag} {trans_text}")

def translate_all_languages(event, user_state):
    """Translate to all languages"""
    with user_state.lock:
        last_msg = user_state.last_bot_message
        if not last_msg:
            reply(event.reply_token, "❌ ไม่มีข้อความล่าสุดสำหรับแปล")
            return
    
    translations = []
    for code, (label, lang_code) in LANG_BUTTONS.items():
        trans_text = translate_text(last_msg, lang_code)
        if trans_text:
            flag = label.split()[0]
            translations.append(f"{flag} {trans_text}")
    
    if translations:
        reply(event.reply_token, "\n\n".join(translations))
    else:
        reply(event.reply_token, "❌ แปลภาษาล้มเหลวทุกภาษา")

# ---------------- MESSAGE HANDLER ----------------
def handle_message(event):
    """Handle message - improved"""
    user_id = event.source.user_id
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(f"📨 Message from User {user_id} at {timestamp}")
    
    # Handle stickers
    if isinstance(event.message, StickerMessageContent):
        package_id = getattr(event.message, "package_id", "unknown")
        sticker_id = getattr(event.message, "sticker_id", "unknown")
        logging.info(f"🎭 Sticker: package {package_id}, sticker {sticker_id}")
        executor.submit(process_sticker, event, user_id, timestamp, package_id, sticker_id)
        return
    
    # Handle images
    if isinstance(event.message, ImageMessageContent):
        logging.info("🖼️ Image received. Starting improved OCR...")
        executor.submit(process_image, event, user_id, timestamp)
        return
    
    # Handle text messages
    if not hasattr(event.message, "text"):
        return
    
    msg = event.message.text.strip()
    user_state = get_user_state(user_id)
    
    current_time = time.time()
    with user_state.lock:
        user_state.last_request_time = current_time
        user_state.history.append({"time": timestamp, "message": msg})
    
    logging.info(f"💬 Text: {msg}")
    lower_msg = msg.lower()
    
    # Translation commands
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
            translate_and_reply(event, user_state, lang_code, flag)
            return
    
    if lower_msg == "/translate_all":
        translate_all_languages(event, user_state)
        return
    
    # Basic commands
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
• 'source' - ดูข้อมูล PDF
• 'summary [จำนวน]' - ดูประวัติล่าสุด
• 'fullhistory' - ดูประวัติทั้งหมด

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
        reply(event.reply_token, "🧹 ล้างประวัติและ cache เรียบร้อย", show_quick_reply=False)
        return
    
    if lower_msg == "source":
        reply(event.reply_token, f"📄 ข้อมูลจาก: {os.path.basename(PDF_FILE)}", show_quick_reply=False)
        return
    
    # History commands
    if lower_msg.startswith("summary") or lower_msg.startswith("สรุปผล"):
        try:
            parts = lower_msg.split()
            n = int(parts[1]) if len(parts) > 1 else 10
        except (ValueError, IndexError):
            n = 10
        
        n = max(1, min(n, 50))
        
        with user_state.lock:
            if not user_state.history:
                reply(event.reply_token, "❌ ยังไม่มีประวัติการสนทนา", show_quick_reply=False)
                return
            
            recent_history = user_state.history[-n:]
            summary = "\n".join([f"[{h['time']}] {h['message']}" for h in recent_history])
        
        reply(event.reply_token, f"📋 ประวัติล่าสุด {len(recent_history)} รายการ:\n{summary}", show_quick_reply=False)
        return
    
    if lower_msg in ["fullhistory", "สรุปทั้งหมด"]:
        with user_state.lock:
            if not user_state.history:
                reply(event.reply_token, "❌ ยังไม่มีประวัติการสนทนา", show_quick_reply=False)
                return
            
            full_history = "\n".join([f"[{h['time']}] {h['message']}" for h in user_state.history])
        
        if len(full_history) > MAX_LINE_CHARS:
            cut = full_history.rfind('\n', 0, MAX_LINE_CHARS)
            if cut == -1:
                cut = MAX_LINE_CHARS
            full_history = full_history[:cut] + "\n\n...(ประวัติยาวเกินไป)"
        
        reply(event.reply_token, f"📜 ประวัติทั้งหมด:\n\n{full_history}", show_quick_reply=False)
        return
    
    # Send processing message
    try:
        messaging_api.push_message(
            PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text="⏳ กำลังประมวลผล กรุณารอสักครู่...")]
            )
        )
        logging.info("📤 Sent processing message")
    except Exception as e:
        logging.error(f"⚠️ Failed to send processing message: {str(e)}")
    
    # Process query
    logging.info("🔄 Processing query...")
    executor.submit(process_query, event, user_id, msg, timestamp, current_time)

# ---------------- WEBHOOK ----------------
@app.route("/", methods=["POST"])
def callback():
    """Webhook callback"""
    body = request.get_data(as_text=True)
    logging.info("🌐 Webhook received")
    
    try:
        callback_request = CallbackRequest.from_json(body)
    except Exception as e:
        logging.error(f"⚠️ Webhook parse error: {str(e)}")
        return "Bad request", 400
    
    for event in callback_request.events:
        if isinstance(event, MessageEvent):
            executor.submit(handle_message, event)
    
    return "OK", 200

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logging.info("🚀 Starting improved Botnoi Chatbot server...")
    
    # Check dependencies
    if not embedder:
        logging.warning("⚠️ Embedding model not loaded. PDF search will be limited.")
    
    if not os.path.exists(PDF_FILE):
        logging.warning(f"⚠️ PDF file not found: {PDF_FILE}")
    
    # Warm up Ollama
    try:
        logging.info("🧠 Warming up Ollama...")
        ollama.chat(model="qwen3:8b-q4_K_M", messages=[{"role": "user", "content": "สวัสดี"}])
        logging.info("✅ Ollama ready")
    except Exception as e:
        logging.error(f"⚠️ Ollama warm-up failed: {str(e)}")
    
    logging.info("✅ Server ready on port 5000")
    app.run(port=5000, debug=False, use_reloader=False)