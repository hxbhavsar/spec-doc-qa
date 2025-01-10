import streamlit as st
import os
from PIL import Image
import pytesseract
import pdf2image
import anthropic
import google.generativeai as genai
from openai import OpenAI
from io import BytesIO
from typing import Dict, Any, Union
from abc import ABC, abstractmethod
import re

# Set page config
st.set_page_config(
    page_title="Spec Doc Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# Utility functions for formatting
def clean_text(text: str) -> str:
    """Clean up text by removing special characters and extra whitespace"""
    # Remove TextBlock formatting
    text = re.sub(r'\[?TextBlock\(text=\'|\', type=\'text\'\)\]?', '', text)
    
    # Convert escaped newlines to actual newlines
    text = text.replace('\\n', '\n')
    
    # Remove extra whitespace and empty lines
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)

def format_value_with_unit(value: str, unit: str) -> str:
    """Format a value-unit pair with consistent styling"""
    return f"**{value}** {unit}"

def format_specification(name: str, value: str, unit: str = "") -> str:
    """Format a specification with consistent styling"""
    if unit:
        return f"• {name}: **{value}** {unit}"
    return f"• {name}: **{value}**"

def display_formatted_answer(answer: Union[str, list]):
    """Format and display the answer with proper styling"""
    try:
        # Handle case where answer might be a list
        if isinstance(answer, list):
            answer = ' '.join(map(str, answer))
        
        # Ensure answer is a string and clean it
        answer = clean_text(str(answer))
        
        # Convert markdown-style bold to styled text
        formatted_answer = answer.replace('**', '__')
        
        # Add custom styling
        st.markdown("""
            <style>
            .stMarkdown ul {
                padding-left: 20px;
            }
            .stMarkdown li {
                margin-bottom: 10px;
            }
            .highlighted {
                background-color: #f0f2f6;
                padding: 2px 5px;
                border-radius: 3px;
            }
            .spec-value {
                font-weight: bold;
                color: #0066cc;
            }
            code {
                color: #FF4B4B;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Display the formatted answer
        st.markdown(formatted_answer)
    except Exception as e:
        st.error(f"Error formatting answer: {str(e)}")
        # Fallback: display raw answer
        st.write(answer)

# Abstract base class for LLM providers
class LLMProvider(ABC):
    @abstractmethod
    def initialize(self, api_key: str):
        pass
    
    @abstractmethod
    def get_response(self, prompt: str, document_context: str) -> str:
        pass

class AnthropicProvider(LLMProvider):
    def initialize(self, api_key: str):
        try:
            if not api_key or not api_key.startswith('sk-ant-'):
                st.error("Invalid Anthropic API key. Must start with 'sk-ant-'")
                return None
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f"Anthropic initialization error: {str(e)}")
            return None

    def get_response(self, prompt: str, document_context: str) -> str:
        try:
            system_prompt = """You are a helpful assistant specialized in analyzing technical specification documents. 
            When answering questions:
            1. Format your response with bullet points for better readability
            2. Put important values, measurements, and specifications in **bold**
            3. Organize information in clear sections if applicable
            4. If providing measurements or specifications, always include units
            5. If making comparisons, use clear visual formatting
            6. Start each main point with a "•" bullet point
            7. Use sub-bullets ("-") for supporting details
            8. Keep responses concise and focused
            
            Use the following document context to answer questions:
            
            {document_context}
            
            Provide clear, structured answers based on the information in the document."""
            
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user", 
                    "content": f"Context: {document_context}\n\nQuestion: {prompt}\n\nPlease format the response with bullet points and highlight important values in **bold**."
                }]
            )
            return str(message.content)
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            return f"An error occurred: {str(e)}"

class OpenAIProvider(LLMProvider):
    def initialize(self, api_key: str):
        try:
            if not api_key or not api_key.startswith('sk-'):
                st.error("Invalid OpenAI API key. Must start with 'sk-'")
                return None
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"OpenAI initialization error: {str(e)}")
            return None

    def get_response(self, prompt: str, document_context: str) -> str:
        try:
            messages = [
                {"role": "system", "content": """You are analyzing technical documents. Format your responses as follows:
                - Use bullet points ("•") for main points
                - Use sub-bullets ("-") for supporting details
                - Put important values and specifications in **bold**
                - Include units with measurements
                - Use clear sections for different topics
                - Highlight key specifications
                - Keep responses concise and focused
                
                Document context: {document_context}"""},
                {"role": "user", "content": f"{prompt}\n\nPlease format the response with bullet points and highlight important values in **bold**."}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0
            )
            return str(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error getting OpenAI response: {str(e)}")
            return f"An error occurred: {str(e)}"

class GeminiProvider(LLMProvider):
    def initialize(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Gemini initialization error: {str(e)}")
            return None

    def get_response(self, prompt: str, document_context: str) -> str:
        try:
            context = f"""Document context: {document_context}

            Please answer the following question, formatting your response with:
            - Bullet points for main points (use "•")
            - Sub-bullets for details (use "-")
            - Important values in **bold**
            - Clear sections
            - Units for all measurements
            - Keep responses concise and focused

            Question: {prompt}"""
            
            response = self.model.generate_content(context)
            return str(response.text)
        except Exception as e:
            st.error(f"Error getting Gemini response: {str(e)}")
            return f"An error occurred: {str(e)}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = None

def process_pdf(pdf_file):
    """Convert PDF to images and extract text"""
    try:
        # Set poppler path based on system
        import platform
        if platform.system() == 'Darwin':  # macOS
            if platform.machine() == 'arm64':  # Apple Silicon
                poppler_path = '/opt/homebrew/bin'
            else:  # Intel
                poppler_path = '/usr/local/bin'
        
        # Convert PDF to images with explicit poppler path
        images = pdf2image.convert_from_bytes(
            pdf_file.read(),
            poppler_path=poppler_path
        )
        
        processed_pages = []
        for i, image in enumerate(images):
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            text = pytesseract.image_to_string(image)
            
            processed_pages.append({
                'page_num': i + 1,
                'image': img_byte_arr,
                'text': text
            })
            
        return processed_pages
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.info("Make sure poppler is installed: brew install poppler")
        return None

def initialize_llm_provider(provider_name: str) -> LLMProvider:
    """Initialize selected LLM provider"""
    providers = {
        'Anthropic': AnthropicProvider,
        'OpenAI': OpenAIProvider,
        'Google Gemini': GeminiProvider
    }
    
    provider = providers[provider_name]()
    
    try:
        api_key = st.secrets.get(f"{provider_name.upper().replace(' ', '_')}_API_KEY")
        if not api_key:
            st.error(f"No API key found for {provider_name}")
            return None
            
        provider.initialize(api_key)
        return provider
    except Exception as e:
        st.error(f"Error initializing {provider_name}: {str(e)}")
        return None

def main():
    st.title("🤖 Specification Document Q&A Assistant")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # LLM Provider Selection
        llm_options = ['Anthropic', 'OpenAI', 'Google Gemini']
        selected_llm = st.selectbox("Select LLM Provider", llm_options)
        
        if selected_llm != st.session_state.get('selected_llm'):
            st.session_state.selected_llm = selected_llm
            st.session_state.llm_provider = initialize_llm_provider(selected_llm)
        
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    processed_pages = process_pdf(uploaded_file)
                    if processed_pages:
                        st.session_state.current_doc = processed_pages
                        st.success("Document processed successfully!")
    
    # Main content area
    if st.session_state.current_doc and st.session_state.llm_provider:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Document Viewer")
            page_num = st.number_input(
                "Page", 
                min_value=1, 
                max_value=len(st.session_state.current_doc),
                value=1
            )
            
            current_page = st.session_state.current_doc[page_num - 1]
            st.image(current_page['image'], use_column_width=True)
        
        with col2:
            st.subheader("Ask Questions")
            question = st.text_input("Enter your question about the document:")
            
            if st.button("Ask"):
                if question:
                    with st.spinner("Getting answer..."):
                        document_context = "\n\n".join([
                            f"Page {page['page_num']}:\n{page['text']}" 
                            for page in st.session_state.current_doc
                        ])
                        
                        response = st.session_state.llm_provider.get_response(
                            question, 
                            document_context
                        )
                        
                        if response:
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": response,
                                "provider": st.session_state.selected_llm
                            })
            
            # Display chat history with formatted answers
            st.subheader("Chat History")
            for chat in st.session_state.chat_history:
                with st.expander(
                    f"Q: {chat['question']} (via {chat['provider']})", 
                    expanded=True
                ):
                    display_formatted_answer(chat['answer'])
    
    else:
        if not st.session_state.llm_provider:
            st.warning("Please check your API keys in .streamlit/secrets.toml")
        if not st.session_state.current_doc:
            st.info("Please upload a specification document to begin.")

if __name__ == "__main__":
    main()