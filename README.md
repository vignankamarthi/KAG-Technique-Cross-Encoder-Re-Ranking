# Cross-Encoder-Re-Ranking with Query Refinement: Advanced RAG Techniques

This project implements **Re-Ranking with Query Refinement**, an advanced **Retrieval-Augmented Generation (RAG)** approach that improves contextual understanding and retrieval precision for Large Language Models (LLMs). The pipeline refines the retrieval process by combining **ChromaDB, OpenAI’s API, SentenceTransformer Cross-Encoder**, and **UMAP** by scoring and re-ranking results for better alignment with the query.

### Workflow Overview:
1. **Extract and Process Text**: Extract raw text from a PDF document and format it into manageable chunks.
2. **Chunk and Embed**: Use ChromaDB to split the text into chunks and generate vector embeddings for efficient search.
3. **Generate Sub-Queries**: Query the document embeddings using OpenAI's API to retrieve initial context.
4. **Score and Re-Rank**: Use SentenceTransformer’s Cross-Encoder to score and re-rank the retrieved documents based on query relevance.
5. **Generate Augmented Results**: Feed the top-ranked documents into OpenAI’s API to produce a refined, context-rich response.

This project highlights the importance of scoring and re-ranking in RAG pipelines to ensure retrieval precision and optimal LLM performance.

## Getting Started
Follow these steps to set up the project and install the necessary dependencies:

1. **Clone the repository**:
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install Dependencies
    Install the required dependencies using the requirements.txt file:
    ```bash
    pip install -r requirements.txt

3. Set up OpenAI API access:
    This project requires access to OpenAI’s API. Follow these steps:
	* Create an OpenAI account: Visit OpenAI’s website to sign up if you don’t already have an account.
	* Generate an API key: 
         - Go to the API Keys page in your OpenAI dashboard.
	     - Click “Create new secret key” and copy the key.
	* Purchase API usage credits: Ensure that your OpenAI account has sufficient credits or a billing plan set up for API usage.

4. Add you OpenAI API Key to the Project
    * Create a .env file in the root of the repository:
     ```bash
        touch .env
     ```
    * Add the following line to the .env file, replacing your-api-key-here with your actual OpenAI API key:
     - OPENAI_API_KEY=place-your-api-key-here

## Conclusions
The project evaluates the effectiveness of **Cross-Encoder-Re-Ranking with Query Refinement** using UMAP visualizations:

- **Grey Dots**: Represent embeddings from the entire document.
- **Red X**: Denotes the original query embedding.
- **Orange X**: Denotes the re-ranked query embedding.
- **Green Circles**: Represent embeddings of the retrieved documents after re-ranking.

The visualizations illustrate that the **green circles** are significantly closer to the **orange X** (re-ranked query embedding) than to the **red X** (original query embedding). This demonstrates an improvement in relevance and retrieval quality.

The **Cross-Encoder-Re-Ranking with Query Refinement** pipeline demonstrates significant improvements in retrieval relevance and context alignment, making it a valuable technique for enhancing LLM performance in information retrieval tasks. By using re-ranking, the pipeline ensures the top documents align closely with the query, improving response quality and contextual understanding.