import fitz  # PyMuPDF
from typing import List, Dict
import hashlib

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    מחלק PDF לחתיכות טקסט עם מטא-דאטה
    """
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        chunk_size = 500
        overlap = 50
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:
                chunks.append({
                    "text": chunk,
                    "page": page_num + 1,
                    "source": pdf_path
                })
    
    return chunks