# rag_engine.py
from groq import Groq
from vector_store import VectorStore
from typing import Dict
import os

class RAGEngine:
    def __init__(self):
        self.vector_store = VectorStore()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def answer(self, question: str) -> Dict:
        # שלב 1 — שאיבה
        relevant_chunks = self.vector_store.search(question, n_results=3)
        context = "\n\n".join([c["text"] for c in relevant_chunks])
        
        # שלב 2 — יצירה
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": """אתה עוזר משפטי מקצועי.
                    ענה רק על בסיס המסמכים שסופקו.
                    תמיד ציין את מספר העמוד שממנו לקחת את המידע.
                    אם התשובה לא במסמכים — אמור בדיוק: 'מידע זה לא נמצא במסמך.'"""
                },
                {
                    "role": "user",
                    "content": f"""
מסמכים רלוונטיים:
{context}

שאלה: {question}
                    """
                }
            ]
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [f"עמוד {c['page']}" for c in relevant_chunks]
        }