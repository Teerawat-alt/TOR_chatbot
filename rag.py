from __future__ import annotations
import io
import os
from dataclasses import dataclass
from typing import List, Iterable, Optional

from typing import Optional, Any, List
from pydantic import PrivateAttr, Field
from pydantic.config import ConfigDict  # pydantic v2

from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import pdfplumber
from PyPDF2 import PdfReader
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import Runnable


# LLM (HuggingFace local/Hub via transformers pipeline)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.base import LLM
from typing import Any

class PDFParseConfig:
    do_ocr_fallback: bool = True
    ocr_lang: str = "tha+eng" # Thai first
    
    
class PDFLoader:
    def __init__(self, cfg: PDFParseConfig = PDFParseConfig()):
        self.cfg = cfg

    def load(self, files: Iterable[io.BytesIO]) -> str:
        """Extract text & tables. Keep simple table structure via TSV blocks."""
        texts: List[str] = []
        for f in files:
            # First try pdfplumber (handles vector text + tables reasonably)
            try:
                f.seek(0)
                with pdfplumber.open(f) as pdf:
                    for page in pdf.pages:
                        # Text
                        page_text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                        # Tables
                        table_blocks = []
                        try:
                            tables = page.extract_tables(table_settings={
                                "vertical_strategy": "lines",
                                "horizontal_strategy": "lines",
                                "intersection_tolerance": 5,
                            })
                        except Exception:
                            tables = []
                        for t in tables or []:
                            rows = ["\t".join([cell.strip() if cell else "" for cell in row]) for row in t]
                            table_blocks.append("\n".join(rows))
                        combined = page_text
                        if table_blocks:
                            combined += "\n\n[[TABLE_START]]\n" + "\n\n[[TABLE_END]]\n\n[[TABLE_START]]\n".join(table_blocks) + "\n[[TABLE_END]]\n"
                        if combined.strip():
                            texts.append(combined)
            except Exception:
                # Fallback to PyPDF2 text
                f.seek(0)
                try:
                    reader = PdfReader(f)
                    txt = []
                    for p in reader.pages:
                        t = p.extract_text() or ""
                        if t.strip():
                            txt.append(t)
                    if txt:
                        texts.append("\n\n".join(txt))
                except Exception:
                    pass

            # OCR fallback if nothing extracted and allowed
            if self.cfg.do_ocr_fallback and (not texts or not "".join(texts).strip()):
                if not OCR_AVAILABLE:
                    continue
                try:
                    f.seek(0)
                    images = self._pdf_to_images(f)
                    ocr_texts = []
                    for img in images:
                        ocr_texts.append(pytesseract.image_to_string(img, lang=self.cfg.ocr_lang))
                    if ocr_texts:
                        texts.append("\n\n".join(ocr_texts))
                except Exception:
                    pass

        return "\n\n".join(texts)
    
    def _pdf_to_images(self, fileobj: io.BytesIO) -> List[Image.Image]:
        """Very light PDF->image path using pdfplumber rasterization."""
        out = []
        fileobj.seek(0)
        with pdfplumber.open(fileobj) as pdf:
            for page in pdf.pages:
                out.append(page.to_image(resolution=220).original)
        return out
    
# Chunking
class TextChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 180):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n[[TABLE_END]]\n", "\n\n", "\n", "。", ".", " "]
        )

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
# Vector Index (Embeddings)
class VectorIndex:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        # Good multilingual (Thai) sentence embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vs: Optional[FAISS] = None

    def build(self, chunks: List[str]) -> FAISS:
        self.vs = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        return self.vs

    def retriever(self):
        if not self.vs:
            raise RuntimeError("Vector index not built")
        return self.vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    

# RAG Conversation

def build_conversation(retriever, llm: LLM) -> Runnable:
    """Return a ConversationalRetrievalChain with Thai system prompt."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    system_prompt = (
        "คุณเป็นผู้ช่วยที่เชี่ยวชาญ TOR ภาครัฐไทย ช่วยตอบเป็นภาษาไทย สุภาพ กระชับ\n"
        "ให้ยกอ้างอิงจากข้อความที่ค้นพบ และสรุปประเด็นเป็นหัวข้อย่อยเมื่อเหมาะสม.\n"
        "ถ้าข้อมูลไม่พอให้ระบุว่ายังไม่พบในเอกสาร\n"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        verbose=False
    )
    # LangChain's built-in chain doesn't expose system_prompt easily for generic LLMs;
    # We'll prepend it to user question inside the UI layer.
    chain.system_prompt = system_prompt  # type: ignore[attr-defined]
    return chain

class ThaiTORRAG:
    def __init__(self,
                 llm,  # <--- รับ llm จากภายนอก
                 embed_model: str = "intfloat/multilingual-e5-base"):
        self.loader = PDFLoader()
        self.chunker = TextChunker()
        self.index = VectorIndex(model_name=embed_model)
        self.llm = llm
        self.conversation: Optional[Runnable] = None

    def process_pdfs(self, files: Iterable[io.BytesIO]):
        raw = self.loader.load(files)
        chunks = self.chunker.split(raw)
        vs = self.index.build(chunks)
        self.conversation = build_conversation(vs.as_retriever(), self.llm)

    def ask(self, question: str) -> dict:
        if not self.conversation:
            raise RuntimeError("Call process_pdfs first")
        system = getattr(self.conversation, "system_prompt", "")
        q = f"{system}\n\nคำถาม: {question}\nกรุณาตอบเป็นภาษาไทย"
        return self.conversation({"question": q})