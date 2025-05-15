# PDF RAG Chatbot

A Streamlit application that allows users to upload PDF documents and ask questions about their content using Retrieval Augmented Generation (RAG).

## Features

- PDF document upload and text extraction
- Document chunking and vector embedding
- Question-answering using OpenAI's models
- Chat interface with conversation history

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf-rag-chatbot.git
   cd pdf-rag-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open the application in your browser (typically at http://localhost:8501)

3. Upload a PDF document using the file uploader

4. Ask questions about the content of the PDF in the chat interface

## How It Works

1. The uploaded PDF is processed and text is extracted
2. The text is split into smaller chunks for efficient processing
3. These chunks are embedded using OpenAI's embedding model
4. The embeddings are stored in a FAISS vector database
5. When you ask a question, the system:
   - Finds the most relevant chunks from the PDF
   - Sends these chunks along with your question to the OpenAI LLM
   - Returns a contextual answer based on the PDF content

## Technologies Used

- Streamlit: For the web interface
- LangChain: For the RAG implementation
- OpenAI: For embeddings and the LLM
- FAISS: For the vector database
- PyPDF2: For PDF text extraction

## License

MIT 