# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .input-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and tokenizer with caching"""
    try:
        # Load from your merged model directory
        model_path = "./merged_summarizer"  # Make sure this path is correct
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Create pipeline
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">📝 AI Text Summarizer</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a fine-tuned T5 model with LoRA to generate summaries "
        "from long articles. The model was trained on news articles to produce "
        "concise and informative summaries."
    )
    
    st.sidebar.title("Settings")
    max_length = st.sidebar.slider("Maximum summary length", 50, 200, 130)
    min_length = st.sidebar.slider("Minimum summary length", 10, 100, 30)
    
    # Load model
    with st.spinner("Loading summarization model..."):
        summarizer = load_model()
    
    if summarizer is None:
        st.error("Failed to load the model. Please check if the model files are available.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        input_method = st.radio("Choose input method:", 
                               ["Text Input", "File Upload", "Sample Text"])
        
        input_text = ""
        
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter your text to summarize:",
                height=300,
                placeholder="Paste your article or long text here..."
            )
            
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload a text file", 
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: .txt, .pdf, .docx"
            )
            if uploaded_file is not None:
                # For simplicity, handling only .txt files in this example
                if uploaded_file.type == "text/plain":
                    input_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("Please upload a .txt file for this demo")
                    
        elif input_method == "Sample Text":
            sample_text = """
            Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making, and competing at the highest level in strategic game systems.

            As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.

            Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding, followed by new approaches, success and renewed funding. AI research has tried and discarded many different approaches during its lifetime, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge and imitating animal behavior. In the first decades of the 21st century, highly mathematical statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.
            """
            input_text = st.text_area("Sample text:", value=sample_text, height=300)
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        if st.button("Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text to summarize")
                return
                
            with st.spinner("Generating summary..."):
                try:
                    # Preprocess text for the model
                    if len(input_text) > 2000:
                        st.info(f"Text length: {len(input_text)} characters. Truncating to 2000 characters.")
                        input_text = input_text[:2000]
                    
                    # Generate summary
                    summary = summarizer(
                        input_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )[0]['summary_text']
                    
                    # Display results
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Original Length", f"{len(input_text)} chars")
                    with col_b:
                        st.metric("Summary Length", f"{len(summary)} chars")
                    with col_c:
                        compression = ((len(input_text) - len(summary)) / len(input_text)) * 100
                        st.metric("Compression", f"{compression:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
        
        # Add download button for the summary
        if 'summary' in locals():
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="generated_summary.txt",
                mime="text/plain"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit & Hugging Face Transformers | "
        "Model: Fine-tuned T5 with LoRA"
    )

if __name__ == "__main__":
    main()
