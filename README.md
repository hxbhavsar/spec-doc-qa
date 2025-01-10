# spec-doc-qa

# Multi-LLM Specification Document Q&A Assistant

A Streamlit application that allows users to upload technical specification documents (PDFs) and ask questions about them using multiple LLM providers (Anthropic's Claude, OpenAI's GPT-4, and Google's Gemini).

## Features

- 📄 PDF document processing with OCR capabilities
- 🤖 Multiple LLM provider support (Anthropic, OpenAI, Google Gemini)
- 💬 Interactive Q&A interface
- 📋 Chat history with formatted responses
- 📊 Document viewer with page navigation
- ✨ Formatted responses with highlighted specifications

## Prerequisites

- Python 3.8 or higher
- MacOS (for current installation instructions)
- Homebrew (for installing system dependencies)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/hxbhavsar/spec-doc-qa.git
cd spec-doc-qa
```

2. **Install Homebrew (if not already installed)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

3. **Install system dependencies**
```bash
brew install poppler tesseract
```

4. **Create and activate a virtual environment (optional but recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```

5. **Install Python dependencies**
```bash
pip3 install -r requirements.txt
```

## Configuration

1. Create a `.streamlit` directory and `secrets.toml` file:
```bash
mkdir .streamlit
touch .streamlit/secrets.toml
```

2. Add your API keys to `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-anthropic-key"
OPENAI_API_KEY = "your-openai-key"
GOOGLE_GEMINI_API_KEY = "your-gemini-key"
```

## Usage

1. **Start the application**
```bash
python3 -m streamlit run app.py
```

2. **Access the web interface**
- Open your browser and go to `http://localhost:8501`

3. **Using the application**
- Select your preferred LLM provider from the sidebar
- Upload a PDF specification document
- Click "Process Document" to extract text and prepare for Q&A
- Ask questions about the document in the input field
- View responses in the chat history section

## Project Structure

```
spec-doc-qa/
├── README.md
├── requirements.txt
├── app.py
└── .streamlit/
    └── secrets.toml
```

## Dependencies

Create a `requirements.txt` file with the following content:

```txt
streamlit
pdf2image
pytesseract
pillow
anthropic
openai
google-generativeai
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

1. **PDF Processing Issues**
   - Ensure poppler is installed: `brew install poppler`
   - Check poppler path in the code matches your system

2. **OCR Issues**
   - Ensure tesseract is installed: `brew install tesseract`
   - Verify tesseract installation: `tesseract --version`

3. **API Connection Issues**
   - Verify API keys in `.streamlit/secrets.toml`
   - Check internet connection
   - Ensure no firewall blocking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- Anthropic, OpenAI, and Google for their LLM APIs
- Poppler and Tesseract for document processing
