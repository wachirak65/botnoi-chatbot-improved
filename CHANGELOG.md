# 📝 Changelog

## v2.0 - PIL-based Image Processing (2024-12-11)

### ✨ New Features
- **PIL-based Image Processing** - แทนที่ OpenCV ด้วย PIL เพื่อรองรับ Python version ที่หลากหลาย
- **Improved PDF Search** - ใช้ overlapping chunks และ metadata
- **Smart OCR Strategy** - ลองหลายวิธีเพื่อผลลัพธ์ที่ดีที่สุด
- **Enhanced AI Responses** - ปรับปรุง system prompt และ context handling

### 🔧 Technical Improvements
- **Removed OpenCV dependency** - ใช้ PIL + NumPy แทน
- **Better Python compatibility** - รองรับ Python 3.7+
- **Optimized caching** - จำกัดขนาด cache และลบข้อมูลเก่า
- **Improved error handling** - จัดการข้อผิดพลาดได้ดีขึ้น

### 🖼️ Image Processing Changes
- **PIL-based preprocessing:**
  - Contrast enhancement ด้วย `ImageEnhance.Contrast`
  - Sharpness improvement ด้วย `ImageEnhance.Sharpness`
  - Noise reduction ด้วย `ImageFilter.MedianFilter`
  - Auto-contrast ด้วย `ImageOps.autocontrast`
  - Simple thresholding ด้วย NumPy

- **Image analysis without OpenCV:**
  - Pixel variance analysis
  - Histogram peak detection
  - Simple edge detection ด้วย Sobel filter

### 📦 Dependencies Removed
- `opencv-python` - ไม่ต้องใช้แล้ว

### 📦 Dependencies Updated
- `Pillow` - ใช้สำหรับ image processing หลัก
- `numpy` - ใช้สำหรับ array operations

### 🐛 Bug Fixes
- แก้ไขปัญหา compatibility กับ Python versions ต่างๆ
- ปรับปรุงการจัดการ memory สำหรับ image processing
- แก้ไข edge cases ในการวิเคราะห์รูปภาพ

### 📈 Performance
- ลดการใช้ memory ในการประมวลผลรูปภาพ
- เพิ่มความเร็วในการ startup (ไม่ต้องโหลด OpenCV)
- ปรับปรุงการ cache management

---

## v1.0 - Original Version

### Features
- Basic PDF search with FAISS
- Simple OCR with Tesseract
- LINE Bot integration
- Translation support
- Basic AI responses with Ollama

### Dependencies
- OpenCV for image processing
- All other core dependencies