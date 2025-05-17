import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import Field

# --- CONFIGURE GEMINI ---
genai.configure(api_key="AIzaSyBmUYQdImYbjPJesYFoMHVEfibp5l1CKBc")  # Replace with your Google AI Studio key

# Custom LangChain-compatible wrapper for Gemini 1.5 Flash
class GeminiLLM(LLM):
    model: Optional[genai.GenerativeModel] = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "google-gemini-1.5-flash"

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI-Powered Industrial Safety Monitoring", layout="wide")
st.title("🛡️ AI-Powered Industrial Safety Monitoring System")

# Load FAISS DB
embeddings = HuggingFaceEmbeddings()

@st.cache_resource
def load_vector_db():
    try:
        return FAISS.load_local(
            "incident_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error("❌ Failed to load FAISS index. Ensure it is trusted and exists.")
        st.stop()

db = load_vector_db()
retriever = db.as_retriever()
llm = GeminiLLM()

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

# Sidebar Navigation
st.sidebar.title("🧠 Intelligent Agents")
agent_option = st.sidebar.selectbox(
    "Choose Agent",
    ["Incident Analysis Agent", "Prevention Agent", "Compliance Agent"]
)

# Main Interface
if agent_option == "Incident Analysis Agent":
    st.header("📄 Incident Analysis Agent")
    query = st.text_input(
        "Describe a situation or risk to analyze",
        "What incidents are related to gas leaks near furnace zones?"
    )
    if st.button("Analyze"):
        with st.spinner("Retrieving historical incidents..."):
            result = qa_chain(query)
            st.subheader("🔍 Relevant Incident Insights")
            st.write(result['result'])
            with st.expander("📂 Source Documents"):
                for doc in result['source_documents']:
                    st.markdown(f"**{doc.metadata.get('source', 'Unknown Source')}**")
                    st.code(doc.page_content[:500])

elif agent_option == "Prevention Agent":
    st.header("✅ Pre-Shift Prevention Checklist")
    checklist_query = "What safety checks should be performed before operating machinery in high-noise zones?"
    with st.spinner("Generating pre-emptive action checklist..."):
        result = qa_chain(checklist_query)
        st.write("### 🔧 Recommended Pre-Shift Checklist:")
        st.markdown(result['result'])

elif agent_option == "Compliance Agent":
    st.header("📊 Safety Compliance Report Generator")
    compare_query = "Compare current safety incident patterns with last year's trends and OSHA guidelines"
    with st.spinner("Generating compliance report..."):
        result = qa_chain(compare_query)
        st.write("### 📈 Safety Compliance Report:")
        st.markdown(result['result'])

# Footer
st.markdown("---")
st.markdown("🚧 _Powered by FAISS + LangChain + Gemini 1.5 Flash (AI Studio)_")
