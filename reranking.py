from helper_utils import word_wrap, load_chroma
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt
import umap

# Loading  environment variables from .env file
load_dotenv()

# Setting environment variables
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Initializing ChromaDB's sentence tranformer embedding fucntion
embedding_function = SentenceTransformerEmbeddingFunction()

# Processing and extracting only text from the entire PDF file
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filtering the empty strings
pdf_texts = [text for text in pdf_texts if text]
# print(
#     word_wrap(
#         pdf_texts[0],
#         width=100,
#     )
# )

# Splitting the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Assigning 256 tokens per chunk, making the data more manageable to process. 
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#print(word_wrap(token_split_texts[0]))
#print(f"\nTotal chunks: {len(token_split_texts)}")

# Instantiating the sentence transforming embedding functions
embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

# Vectorizing the processed microsoft document 
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection3", embedding_function=embedding_function
)

# Embed the tokenized chunks of text to the chroma_collection database and assign IDs 
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

count = chroma_collection.count()
#print(count)

# Initial query
query = "What has been the investment in the research and development sector of the company?"

# get the raw results from a simple inital query
results = chroma_collection.query(query_texts=[query], n_results=10, include=["documents", "embeddings"])
retrieved_documents = results["documents"][0]

# print out the inital results
#for document in results["documents"][0]:
#    print(word_wrap(document))
#    print("")


# Small-Scale Inital Query Example

# Initializing Hugging Face's cross encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Creating pairs of the query and all the different documents in retrieved_documents
pairs = [[query, doc] for doc in retrieved_documents]

# Predicting scores between the query and all the generated documents
scores = cross_encoder.predict(pairs)

# Printing the scores the encoder provided
#print("Scores:")
#for score in scores:
#    print(score)

# Print the descending ordering (by index) of the 
# generated documents
#print("New Ordering:")
#for o in np.argsort(scores)[::-1]:
#    print(o + 1)


# Implementing the Full-Scale Reranking with Augmented Query Expansion

# A function utilizing OpenAI's models to generate augmented queries 
# in the context of a financial research assistant from the given paramter "query."
def generate_multi_query(query, model="gpt-3.5-turbo"):

    prompt = """
    You are a knowledgeable and expert financial research assistant. 
    Your users are investigating an annual report from a company. 
    For the given question, generate up to five related questions to aid them in finding the information they requested. 
    Provide concise, single-topic questions (no compounding sentences) that cover various aspects about and related to the topic. 
    Ensure each question is complete and directly related to the original query. 
    List each question on a separate line with no numbering.
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

# Original Query from queries expansion RAG technique project
original_query = (
    "What details can you provide about the factors that led to and/or aided revenue growth?"
)
generated_queries = generate_multi_query(original_query)

# Format the list to the remove the leading dashes and uncessary characters
generated_queries = [query.strip().lstrip("-").strip() for query in generated_queries]

# Printing and inspecting the updated list
print(generated_queries)

# Concatenating the original query with the generated queries
augmented_queries = [original_query] + generated_queries

# Generating results from the augmented queries (original queries + generated queries)
results = chroma_collection.query(
    query_texts=augmented_queries, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Remove potential duplicated documents from retrieved_documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# Making the unique documents in the format of a list
unique_documents = list(unique_documents)

# Creating pairs of the original query and query-expansion-generated unique_documents
for doc in unique_documents:
    pairs.append([original_query, doc])

# Generating scores for the newly generated documents
scores = cross_encoder.predict(pairs)

# Print the scores for the new documents
print("Scores:")
for score in scores:
    print(score)

# Print the sorted ordering of the new documents
print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)

# Seperating the top highest scoring documents and extracting the documents
# Ensuring we do not exceed the number of unique_documents or scores
top_n = min(5, len(scores), len(unique_documents))
top_indices = np.argsort(scores)[::-1][:top_n]
top_documents = [unique_documents[i] for i in top_indices]

# Concatenating the top documents into a single piece of context for the OpenAI LLM
context = "\n\n".join(top_documents)

# Generating the final answer utilixing the OpenAI API. 
# We utilize the cross-encoder-re-ranked queries to provide the abosulte best and fitlered-through
# context for the API call. 
def generate_multiple_queries(query, context, model="gpt-3.5-turbo"):

    prompt = f"""
    You are a knowledgeable and expert financial research assistant. 
    Your users are investigating an annual report from a company. 
    """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"Based on the following context:\n\n{context}\n\nAnswer the following query: '{query}'",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

# Generating and printing the final results of the reranking with a corss-encoder
final_results = generate_multiple_queries(query=original_query, context=context)
print("Final Answer:")
print(final_results)


# Visualizations

# Step 1: Generate embeddings for the full dataset
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]

# Step 2: Generate the UMAP transformations
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = umap_transform.transform(embeddings)

# Step 3: Generate the embeddings for the original query
original_query_embedding = embedding_function([original_query])
project_original_query = umap_transform.transform(original_query_embedding)

# Step 4: Generate embeddings for the final ranked-query provided context
# Ensure context is provided as input for this step
final_ranked_query_embeddings = embedding_function([context])
project_final_ranked_query = umap_transform.transform(final_ranked_query_embeddings)

# Step 5: Generate embeddings for the retrieved documents from the final results
retrieved_embeddings = embedding_function(final_results)  # Generate embeddings for final results
projected_result_embeddings = umap_transform.transform(retrieved_embeddings)  # Apply UMAP transformation

# Plotting the embeddings from the full dataset (derived from the entire document), the embeddings from the results 
# after inputting the reranked query, the embeddings from the original query, and the embeddings from the augmented queries. 
# The reranked queries are much closer to the final results than the original query, and there is sigificant improvement in performance. 
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray"
)

plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)

plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=200,
    marker="X",
    color="r",
)

plt.scatter(
    project_final_ranked_query[:, 0],
    project_final_ranked_query[:, 1],
    s=150,
    marker="X",
    color="orange",
)

# Final plot formatting
plt.gca().set_aspect("equal", "datalim")
plt.title(f'Original Query: "{original_query}"')
plt.axis("off")
plt.show()