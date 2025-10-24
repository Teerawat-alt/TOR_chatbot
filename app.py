import io, os
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from rag import ThaiTORRAG   
from htmltemplates import css, user_template, bot_template

load_dotenv()

# ---- model config----
EMBED_MODEL = "intfloat/multilingual-e5-base"
GEN_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"  # ต้องเป็นรูปแบบ org/repo

# ---- ฟังก์ชันแคช LLM (โหลดแค่ครั้งเดียว) ----
@st.cache_resource(show_spinner=False)
def load_llm(model_id: str, hf_token: str | None):
    # ป้องกันใส่ VL มาโดยไม่ตั้งใจ
    low = model_id.lower()
    # if "vl" in low or "vision" in low:
    #     raise ValueError(f"รุ่น {model_id} เป็น Vision-Language; โปรดใช้รุ่น text-only")

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", token=hf_token, trust_remote_code=True
    )
    gen = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=384, temperature=0.2, do_sample=False
    )
    return HuggingFacePipeline(pipeline=gen)

# ---- prepare token ----
HF_TOKEN = (os.getenv("HF_TOKEN")
            #or os.getenv("HUGGINGFACE_HUB_TOKEN")
            #or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )

# ---- โหลด/แคช LLM ครั้งเดียว ----
if "llm" not in st.session_state:
    st.session_state.llm = load_llm(GEN_MODEL, HF_TOKEN)

st.set_page_config(page_title="Thai TOR Chat (PDF)", page_icon="📄")

st.header("แชทกับเอกสาร TOR ของไทย 📄🇹🇭")
st.caption(f"Embedding: {EMBED_MODEL}  |  SLM: {GEN_MODEL}")

with st.sidebar:
    st.subheader("อัปโหลดไฟล์ PDF (หลายไฟล์ได้)")
    pdf_files = st.file_uploader("เลือกไฟล์", type=["pdf"], accept_multiple_files=True)

    if st.button("ประมวลผลเอกสาร"):
        if not pdf_files:
            st.warning("กรุณาเลือกไฟล์ PDF ก่อน")
        else:
            with st.spinner("กำลังประมวลผลเอกสาร..."):
                buffers = [io.BytesIO(f.read()) for f in pdf_files]
                rag = ThaiTORRAG(llm=st.session_state.llm, embed_model=EMBED_MODEL)
                rag.process_pdfs(buffers)
                st.session_state.rag = rag
                st.success("เสร็จสิ้น พร้อมใช้งาน!")

user_q = st.text_input("พิมพ์คำถามเกี่ยวกับ TOR:")
if user_q:
    if "rag" not in st.session_state or st.session_state.rag is None:
        st.error("ยังไม่ได้โหลดเอกสาร กด 'ประมวลผลเอกสาร' ทางซ้ายก่อน")
    else:
        resp = st.session_state.rag.ask(user_q)
        for i, m in enumerate(resp.get("chat_history", [])):
            st.write(m.content)