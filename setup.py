#!/usr/bin/env python3
"""
Setup script for Botnoi Chatbot Improved
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_python_version():
    """ตรวจสอบเวอร์ชัน Python"""
    if sys.version_info < (3, 7):
        logging.error("❌ Python 3.7+ required")
        return False
    logging.info(f"✅ Python {sys.version}")
    return True

def install_requirements():
    """ติดตั้ง Python packages"""
    try:
        logging.info("📦 Installing Python packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("✅ Python packages installed")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """สร้างโฟลเดอร์ที่จำเป็น"""
    directories = ["data", "logs", "cache"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"📁 Created directory: {directory}")
    
    return True

def check_tesseract():
    """ตรวจสอบ Tesseract OCR"""
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            logging.info(f"✅ Found Tesseract at: {path}")
            return True
    
    logging.warning("⚠️ Tesseract not found. Please install from:")
    logging.warning("   https://github.com/UB-Mannheim/tesseract/wiki")
    return False

def check_ollama():
    """ตรวจสอบ Ollama"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("✅ Ollama is installed")
            
            # ตรวจสอบ model
            if "qwen3:8b-q4_K_M" in result.stdout:
                logging.info("✅ Qwen3 model found")
            else:
                logging.warning("⚠️ Qwen3 model not found. Run: ollama pull qwen3:8b-q4_K_M")
            return True
        else:
            logging.warning("⚠️ Ollama not responding")
            return False
    except FileNotFoundError:
        logging.warning("⚠️ Ollama not installed. Download from: https://ollama.ai")
        return False

def setup_environment():
    """ตั้งค่า environment file"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            logging.info("✅ Created .env from .env.example")
            logging.warning("⚠️ Please edit .env with your LINE Bot credentials")
        else:
            logging.error("❌ .env.example not found")
            return False
    else:
        logging.info("✅ .env file exists")
    
    return True

def check_pdf_file():
    """ตรวจสอบไฟล์ PDF"""
    pdf_path = os.path.join("data", "Using_Platform_Botnoi.pdf")
    
    if os.path.exists(pdf_path):
        logging.info("✅ PDF file found")
        return True
    else:
        logging.warning(f"⚠️ PDF file not found: {pdf_path}")
        logging.warning("   Please place your PDF file in the data/ directory")
        return False

def main():
    """Main setup function"""
    logging.info("🚀 Setting up Botnoi Chatbot Improved...")
    
    success = True
    
    # ตรวจสอบ Python version
    if not check_python_version():
        success = False
    
    # สร้างโฟลเดอร์
    if not create_directories():
        success = False
    
    # ติดตั้ง packages
    if not install_requirements():
        success = False
    
    # ตั้งค่า environment
    if not setup_environment():
        success = False
    
    # ตรวจสอบ dependencies
    check_tesseract()
    check_ollama()
    check_pdf_file()
    
    if success:
        logging.info("✅ Setup completed successfully!")
        logging.info("\n📋 Next steps:")
        logging.info("1. Edit .env with your LINE Bot credentials")
        logging.info("2. Place PDF file in data/ directory")
        logging.info("3. Run: python app.py")
    else:
        logging.error("❌ Setup failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())