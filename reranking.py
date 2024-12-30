from helper_utils import word_wrap, load_chroma
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import CrossEncoder

# Loading  environment variables from .env file
load_dotenv()

# Setting environment variables
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

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


# Importing text splitting functions from Langchain for breaking text into processable chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

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
    "microsoft-collection", embedding_function=embedding_function
)

# Embed the tokenized chunks of text to the chroma_collection database and assign IDs 
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

#count = chroma_collection.count()
#print(count)

# initial query
query = "What was the company's total revenue for the year?"

# get the raw results from a simple inital query
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# print out the inital results
#for document in retrieved_documents:
#    print(word_wrap(document))
#    print("\n")

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


original_query = (
    "What details can you provide about the factors that led to and/or aided revenue growth?"
)
augmented_queries = generate_multi_query(original_query)

# Step 1: Print out and inspect the augmented queries generated with OpenAI's API.
#for query in augmented_queries:
    #print("\n", query)

# Step 2: Concatenate the original query with the newly generated, augmented queries.
joint_query = [
    original_query
] + augmented_queries  
# the original query is placed in a list because the chroma database has the ability 
# to handle multiple queries, so we add it in a list format

#print("\n----------------------> \n\n", joint_query)

from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import umap
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Loading  environment variables from .env file
load_dotenv()

# Setting environment variables
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

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


# Importing text splitting functions from Langchain for breaking text into processable chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

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
    "microsoft-collection-processed", embedding_function=embedding_function
)

# Embed the tokenized chunks of text to the chroma_collection database and assign IDs 
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

#count = chroma_collection.count()
#print(count)

# initial query
query = "What was the company's total revenue for the year?"

# get the raw results from a simple inital query
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# print out the inital results
#for document in retrieved_documents:
#    print(word_wrap(document))
#    print("\n")

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


original_query = (
    "What details can you provide about the factors that led to and/or aided revenue growth?"
)
augmented_queries = generate_multi_query(original_query)

# Step 1: Print out and inspect the augmented queries generated with OpenAI's API.
for query in augmented_queries:
    print("\n", query)

# Step 2: Concatenate the original query with the newly generated, augmented queries.
joint_query = [
    original_query
] + augmented_queries  
# the original query is placed in a list because the chroma database has the ability 
# to handle multiple queries, so we add it in a list format

#print("\n----------------------> \n\n", joint_query)

# Initialize the final results query with the specified documnets. 
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"] # embeddings included for later analyiss
)
retrieved_documents = results["documents"]

# Remove potential duplicated documents from retrieved_documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# output the results documents
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)

# Getting the results for visualization and creating a UMAP transform object
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# Getting the embeddings from the original and augmented queries
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)

# Utilizing UMAP's tranformation function
project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

# Getting the embeddings from the final results 
retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

# Utilizing UMAP's tranformation function
projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

# Plotting the embeddings from the full dataset (derived from the entire document), the embeddings from the retrieved 
# documents after inputting the joint query, the embeddings from the original query, and the embeddings from the augmented queries. 
# The joint queries are somewhat closer to the retrieved documents than the original query on average, but there is room for improvement.
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
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
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f'Original Query: "{original_query}"')
plt.axis("off")
plt.show()  # displaying the plot