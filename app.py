"""
AgroAI Assistant - Streamlit App
RAG Search Interface for STE AGRO MELANGE TECHNOLOGIE
"""

import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_query import RAGEngine, SearchResult

load_dotenv()

st.set_page_config(page_title="AgroAI Assistant", page_icon="🌾", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "followup_click" not in st.session_state:
    st.session_state.followup_click = None

st.markdown("""
<style>
.main-header { text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1.5rem; color: white; }
.answer-box { background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 5px solid #667eea; }
.result-card { background: #f7fafc; padding: 1.2rem; border-radius: 12px; margin: 0.8rem 0; border-left: 4px solid #764ba2; }
.score-badge { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; }
.product-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_engine():
    return RAGEngine()

def display_answer(answer):
    st.markdown(f'<div class="answer-box"><p>{answer.replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)

def display_result(r, i):
    st.markdown(f'<div class="result-card"><small>Fragment #{i}</small><span class="score-badge">Score: {r.score:.2f}</span><p>{r.text[:300]}...</p></div>', unsafe_allow_html=True)

def get_followups(query):
    followups = ["💡 Dosage recommande?", "💡 Comment l'utiliser?", "💡 Precautions?"]
    if "pain" in query.lower():
        followups = ["💡 Ameliorer le volume?", "💡 Enzymes pour conservation?", "💡 Conseils fermentation?"]
    return followups[:3]

def main():
    st.markdown('<div class="main-header"><h1>🌾 AgroAI Assistant</h1><h3>STE AGRO MELANGE TECHNOLOGIE</h3></div>', unsafe_allow_html=True)
    
    try:
        engine = init_engine()
        
        with st.sidebar:
            top_k = st.slider("Fragments", 1, 10, 3)
            if st.button("Effacer historique"):
                st.session_state.messages = []
                st.rerun()
            info = engine.get_collection_info()
            st.metric("Documents", f"{info['vectors_count']:,}")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if not st.session_state.messages:
            st.markdown("### 💡 Questions suggerees")
            starters = ["🦴 Enzymes pour texture du pain?", "🥐 Action de l'amylase?", "⚖️ Dosage BVZyme?"]
            cols = st.columns(3)
            for i, q in enumerate(starters):
                with cols[i%3]:
                    clean = q.split(" ", 1)[1]
                    if st.button(q, key=f"s_{i}"):
                        st.session_state.followup_click = clean
                        st.rerun()
        
        if prompt := st.chat_input("Posez votre question..."):
            st.session_state.followup_click = None
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍"):
                    results = engine.search(prompt, top_k=top_k)
                
                if results:
                    with st.spinner("🤖"):
                        answer = engine.generate_answer(prompt, results, st.session_state.messages)
                    display_answer(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    with st.expander("📚 Voir les sources"):
                        for i, r in enumerate(results, 1):
                            display_result(r, i)
                    
                    followups = get_followups(prompt)
                    st.markdown("### 🔗 Questions suivantes")
                    cols = st.columns(2)
                    for i, f in enumerate(followups):
                        if cols[i%2].button(f, key=f"f_{i}_{len(st.session_state.messages)}"):
                            st.session_state.followup_click = f.split(" ", 1)[1]
                            st.rerun()
                else:
                    st.info("😔 Aucune information trouvee.")
        
        if st.session_state.followup_click:
            prompt = st.session_state.followup_click
            st.session_state.followup_click = None
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🔍"):
                    results = engine.search(prompt, top_k=top_k)
                if results:
                    with st.spinner("🤖"):
                        answer = engine.generate_answer(prompt, results, st.session_state.messages)
                    display_answer(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.expander("📚 Voir les sources"):
                        for i, r in enumerate(results, 1):
                            display_result(r, i)
    
    except Exception as e:
        st.error(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
