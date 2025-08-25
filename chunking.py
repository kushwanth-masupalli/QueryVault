import os
import getpass
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec  # Correct import from new SDK

# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = "AIzaSyCIer88Lpddnsyml76MRKlrjOZXl6ZssjQ"
# --- Google Gemini Setup ---
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google Gemini API Key: ")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

# --- Prompt & Structured Output ---
prompt_template = hub.pull("wfh/proposal-indexing")
runnable = prompt_template | llm

class Sentences(BaseModel):
    sentences: List[str]

structured_llm = llm.with_structured_output(Sentences)

def get_propositions(text_chunk):
    runnable_output = runnable.invoke({"input": text_chunk}).content
    result = structured_llm.invoke(runnable_output)
    return result.sentences

# --- Pinecone Setup (NEW SDK) ---
pinecone_api_key = "pcsk_7WZPgu_N668zCwfYQGiEUKggJppRKuwMMo8FYBKu7xcSFqcWFd5L8vgTc69ap5hYSUynkS"
index_name = "qv"

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# # Create index if not exists
# if index_name not in [idx.name for idx in pc.list_indexes()]:
#     print(f"Creating index '{index_name}'...")
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud='aws', region='us-east-1') )

# Connect to index
index = pc.Index(index_name)

# --- Embedding Model ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Process Text ---
text = """Text splitting in LangChain is a critical feature. Users can leverage LangChain for text splitting. LangChain allows users 
to efficiently navigate and analyze vast amounts of text data. Text splitting with LangChain facilitates a deeper understanding 
and more insightful conclusions.

Text splitting facilitates the division of large texts into smaller, manageable segments.
This capability is vital for improving comprehension and processing efficiency.
It is especially important in tasks that require detailed analysis or extraction of specific contexts.

ChatGPT was developed by OpenAI. OpenAI developed ChatGPT. ChatGPT represents a leap forward in natural language processing 
technologies. ChatGPT is a conversational AI model. ChatGPT is capable of understanding and generating human-like text. ChatGPT 
allows for dynamic interactions. ChatGPT provides responses that are remarkably coherent and contextually relevant. ChatGPT has 
been integrated into a multitude of applications. ChatGPT revolutionized the way we interact with machines. ChatGPT 
revolutionized the way we access information."""

paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
propositions = []

print("🔍 Extracting propositions...\n")
for i, para in enumerate(paragraphs):
    print(f"Processing paragraph {i+1}/{len(paragraphs)}...")
    extracted = get_propositions(para)
    propositions.extend(extracted)
    for sent in extracted:
        print(f"  → {sent}")

# --- Embed and Upsert ---
print(f"\n📤 Uploading {len(propositions)} propositions to Pinecone...")

vectors = []
for i, sent in enumerate(propositions):
    embedding = embedding_model.encode(sent).tolist()
    vectors.append({
        "id": f"prop_{i}",
        "values": embedding,
        "metadata": {"content": sent}
    })

index.upsert(vectors=vectors)
print(f"\n✅ Successfully uploaded {len(vectors)} vectors!")

# --- Final Output ---
print(f"\n📌 Total propositions: {len(propositions)}")
print("\n📄 First 10:")
for sent in propositions[:10]:
    print(f"  - {sent}")



# replace with the IDs you want to check
ids_to_check = [f"prop_{i}" for i in range(5)]  

resp = index.fetch(ids=ids_to_check)
# resp may be a dict-like object; print it raw first
print("RAW FETCH RESPONSE:\n", resp)

# Nicely print metadata if available
vectors = resp.get("vectors", {})  # defensive
for vid, v in vectors.items():
    metadata = v.get("metadata")
    print(f"\nID: {vid}")
    print("  - has_metadata:", bool(metadata))
    print("  - metadata:", metadata)
    print("  - values length:", len(v.get("values", [])))
