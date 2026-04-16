import os
import time
import datetime
from Bio import Entrez
from bs4 import BeautifulSoup
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # Added missing import
from queries import all_queries

# --- Configuration & Credentials ---
NCBI_EMAIL = "add email"
NCBI_API_KEY = "add key"
PINECONE_API_KEY = "add key"
OPENAI_API_KEY = "add key" # Required for embeddings
INDEX_NAME = "lite-rag" 
EMBEDDING_MODEL = "text-embedding-3-small"

# Set environment variables for LangChain to pick up automatically
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Set BioPython Entrez credentials
Entrez.email = NCBI_EMAIL
Entrez.api_key = NCBI_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Define the Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Max characters per chunk
    chunk_overlap=100, # Overlap to maintain context between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# --- Define get_pubmed_ids Function ---
def get_pubmed_ids(query, limit=20):
    """Searches PubMed and returns a list of PMIDs."""
    print(f"Searching PubMed for: '{query}'...")
    try:
        # usehistory="y" caches the search on NCBI's servers
        with Entrez.esearch(db="pubmed", term=query, retmax=limit, usehistory="y") as handle:
            results = Entrez.read(handle)
        print(f"Found {len(results['IdList'])} articles.")
        return results["IdList"], results.get("WebEnv"), results.get("QueryKey")
    except Exception as e:
        print(f"Error fetching IDs: {e}")
        return [], None, None

def fetch_and_parse_pubmed(pmid):
    """Fetches metadata and abstract for a given PMID."""
    try:
        # Rate limiting: 0.1s delay for API key users, 0.35s without
        time.sleep(0.1 if Entrez.api_key else 0.35)

        # Fetch Metadata
        with Entrez.efetch(db="pubmed", id=pmid, retmode="xml") as handle:
            soup = BeautifulSoup(handle.read(), "lxml-xml")
        
        # --- Metadata Extraction ---
        title_tag = soup.find("ArticleTitle")
        title = title_tag.get_text() if title_tag else "No Title"
        
        journal_tag = soup.find("Title")
        journal = journal_tag.get_text() if journal_tag else "Unknown Journal"
        
        # Year extraction
        year_tag = soup.find("Year") or (soup.find("PubDate").find("Year") if soup.find("PubDate") else None)
        year = year_tag.get_text() if year_tag else "N/A"
        
        keywords = [k.get_text() for k in soup.find_all("Keyword")]
        
        authors = []
        for author in soup.find_all("Author"):
            last = author.find("LastName")
            initials = author.find("Initials")
            collective = author.find("CollectiveName")
            if last and initials:
                authors.append(f"{last.get_text()} {initials.get_text()}")
            elif collective:
                authors.append(collective.get_text())

        pmc_tag = soup.find("ArticleId", IdType="pmc")
        pmcid = pmc_tag.get_text() if pmc_tag else None
        
        # --- Text Extraction Strategy: ABSTRACT ONLY ---
        abstract_parts = soup.find_all("AbstractText")
        final_text = " ".join([part.get_text() for part in abstract_parts])

        return {
            "id": pmid,
            "pmcid": pmcid,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors,
            "keywords": keywords,
            "text": final_text,
            "is_full_text": False, # Explicitly false since we only grab abstracts
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
    except Exception as e:
        print(f"General error fetching {pmid}: {e}")
        return None
    
def process_data(raw_articles):
    """Chunks the text from the raw articles into manageable pieces."""
    processed = []
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    for item in raw_articles:
        if not item or not item.get("text"):
            continue
            
        doc_id = str(item["id"])
        
        # 1. Smart Chunking using the global text_splitter
        chunks = text_splitter.split_text(item["text"])
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            processed.append({            
                "_id": f"{doc_id}_{idx}", 
                "chunk_text": chunk.strip(), 
                "document_id": doc_id, 
                "document_title": item.get("title"),
                "document_url": item.get("url"),
                "journal": item.get("journal"),
                "year": item.get("year"),
                "authors": item.get("authors", []), 
                "keywords": item.get("keywords", []),
                "chunk_number": idx + 1,
                "total_chunks_in_doc": total_chunks,
                "is_full_text": item.get("is_full_text", False),
                "created_at": current_time,
            })
    
    return processed

def pipe_to_pinecone(processed_chunks):
    """Converts processed dicts to LangChain Documents and upserts to Pinecone."""
    if not processed_chunks:
        print("No chunks to process. Exiting upload.")
        return

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        dimensions=512
    )
    
    docs = []
    for chunk in processed_chunks:
        # --- Updated Metadata: Now captures everything from process_data ---
        metadata = {
            "id": chunk.get("_id"),
            "doc_id": chunk.get("document_id"),
            "title": chunk.get("document_title", "Unknown Title"),
            "url": chunk.get("document_url", ""),
            "year": str(chunk.get("year", "N/A")),
            "journal": chunk.get("journal", "Unknown"),
            
            # Pinecone accepts lists of strings! Passing these as lists 
            # makes metadata filtering much easier later.
            "authors": chunk.get("authors", []), 
            "keywords": chunk.get("keywords", []), 
            
            # Additional context metadata
            "chunk_number": chunk.get("chunk_number", 1),
            "total_chunks_in_doc": chunk.get("total_chunks_in_doc", 1),
            "created_at": chunk.get("created_at", "")
        }
        
        docs.append(Document(
            page_content=chunk["chunk_text"],
            metadata=metadata
        ))
    
    print(f"Upserting {len(docs)} chunks to index: {INDEX_NAME}...")
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs, 
        embedding=embeddings, 
        index_name=INDEX_NAME
    )
    
    print("Upload to 'lite-rag' complete.")

if __name__ == "__main__":
    all_ids = set()

    # 1. Collect IDs from all queries
    for q in all_queries:
        ids, _, _ = get_pubmed_ids(q, limit=5)
        all_ids.update(ids)

    all_ids = list(all_ids)
    print(f"Total unique articles collected: {len(all_ids)}")


    # 2. Fetch and parse
    if all_ids:
        print(f"Fetching data for {len(all_ids)} articles...")
        raw_data = [fetch_and_parse_pubmed(pid) for pid in all_ids]

        # 3. Process and chunk
        print("Processing and chunking text...")
        final_chunks = process_data(raw_data)

        # 4. Upload ONCE
        pipe_to_pinecone(final_chunks)
    else:
        print("No IDs found to process.")