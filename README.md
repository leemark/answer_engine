# Answer Engine

Answer Engine is a Python script that combines web search, Retrieval-Augmented Generation (RAG), reflection, summarization, and follow-up question generation to provide comprehensive answers to user queries. It leverages OpenAI embeddings, Brave search API, and Groq's Mixtral model for inference.

## Features

- Rephrases user queries using an LLM to optimize search results.
- Performs web searches using the Brave search API.
- Scrapes and parses the content of the searched web pages.
- Embeds the scraped content into a Chroma database for efficient retrieval.
- Retrieves relevant documents from the Chroma database based on the user query.
- Generates an initial response using the Groq Mixtral model.
- Performs reflection on the generated response to ensure its relevance to the user query.
- Generates a refined response if the initial response is unsatisfactory.
- Provides follow-up questions based on the user query and the generated response.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/answer-engine.git
   ```

2. Change into the project directory:
   ```bash
   cd answer-engine
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the necessary environment variables:
   - Create a `.env` file in the project root directory.
   - Add the following variables to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     BRAVE_API_KEY=your_brave_api_key
     GROQ_API_KEY=your_groq_api_key
     ```
   - Replace `your_openai_api_key`, `your_brave_api_key`, and `your_groq_api_key` with your actual API keys.

## Usage

1. Open the `app.py` file and modify the `USER_PROMPT` variable to specify your desired query.

2. Run the script:
   ```bash
   python app.py
   ```

3. The script will perform the following steps:
   - Rephrase the user query using an LLM.
   - Search for relevant web pages using the Brave search API.
   - Scrape and parse the content of the searched web pages.
   - Embed the scraped content into a Chroma database.
   - Retrieve relevant documents from the Chroma database based on the rephrased query.
   - Generate an initial response using the Groq Mixtral model.
   - Perform reflection on the generated response and refine it if necessary.
   - Generate follow-up questions based on the user query and the generated response.

4. The script will output the following:
   - The rephrased query.
   - The final generated response.
   - The relevant sources used to generate the response.
   - The generated follow-up questions.

## Dependencies

The Answer Engine relies on the following libraries and APIs:

- OpenAI Embeddings: Used for generating embeddings of the scraped content.
- Brave Search API: Used for performing web searches.
- Groq Mixtral: Used for generating responses and follow-up questions.
- Chroma: Used for storing and retrieving embedded documents.
- BeautifulSoup: Used for web scraping and parsing HTML content.
- Requests: Used for making HTTP requests to web pages.

Make sure to install the required dependencies and set up the necessary API keys before running the script.

## License

This project is licensed under the [MIT License](LICENSE).
