# LangChain Chatbot

This project provides a Flask-based API for a chatbot that utilizes LangChain for document retrieval and OpenAI's GPT for answering questions. It loads documents from URLs, processes them, and creates a FAISS index to facilitate fast retrieval.

## Requirements

To run this project, you need to install the following dependencies:

- `flask`
- `python-dotenv`
- `langchain`
- `faiss-cpu`
- `openai`
- `pickle5`

You can install these dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

Here’s a basic `README.md` file for your project:

```markdown

## Setup

1. **Environment Variables**

   Create a `.env` file in the root directory of the project and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Update URLs**

   Replace the placeholder in the `urls` list in `app.py` with the actual URLs of the documents you want to load:

   ```python
   urls = [
       "Enter your link here"
   ]
   ```

3. **Initial Setup**

   Run the script to load documents, split them, and create the FAISS index:

   ```bash
   python app.py
   ```

   This will also start the Flask server on port 5000.

## API Endpoints

### POST /chat

Send a POST request to this endpoint with a JSON body containing your question:

```json
{
  "question": "Your question here"
}
```

**Response:**

The API will return a JSON response with the answer to your question:

```json
{
  "answer": "The answer to your question",
  "sources": ["Source 1", "Source 2"]
}
```

## Usage

To interact with the chatbot, you can use tools like `curl`, Postman, or create a frontend that sends POST requests to the `/chat` endpoint.
