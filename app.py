# app.py - Updated for current Groq models
# Using Groq LLM with proper error handling

import os
import logging
import asyncio
import re
from typing import List, Optional
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import time
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "kenyan_constitution"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("Please set GROQ_API_KEY in .env file")
    exit(1)

# Available Groq models (as of 2026)
AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Production-ready, best overall",
    "llama-3.1-8b-instant": "Fast, good for simple queries",
    "llama3-70b-8192": "High quality, slower",
    "llama3-8b-8192": "Fast, good for testing",
    "mixtral-8x7b-32768": "DEPRECATED - DO NOT USE"
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Initialize ChromaDB
try:
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    chroma_client.heartbeat()
    logger.info("Connected to ChromaDB server")
except:
    logger.info("Using local ChromaDB storage")
    chroma_client = chromadb.PersistentClient(
        path="./chroma_data",
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# ===== IMPROVED ARTICLE EXTRACTION =====
def extract_article_numbers(text):
    """
    Extract ALL article numbers mentioned in a chunk of text
    Returns a list of article numbers found
    """
    articles = set()
    text_lower = text.lower()
    
    # Pattern 1: "Article 49" or "ARTICLE 49" or "article 49"
    pattern1 = r'(?:article|ARTICLE|Art\.|ART\.)\s+(\d+)'
    matches = re.findall(pattern1, text_lower)
    for match in matches:
        articles.add(f"Article {match}")
    
    # Pattern 2: "Articles 49-55" or "articles 49 to 55"
    pattern2 = r'(?:articles|sections|Articles|Sections)\s+(\d+)[-\s]+(\d+)'
    matches = re.findall(pattern2, text_lower)
    for start, end in matches:
        try:
            start_num, end_num = int(start), int(end)
            for num in range(start_num, end_num + 1):
                articles.add(f"Article {num}")
        except:
            pass
    
    # Pattern 3: "49." at start of line (common in legal documents)
    lines = text.split('\n')
    for line in lines:
        match = re.match(r'^\s*(\d+)\.', line)
        if match:
            articles.add(f"Article {match.group(1)}")
    
    # Pattern 4: "Section 49" or "section 49"
    pattern4 = r'(?:section|Section|SEC\.|sec\.)\s+(\d+)'
    matches = re.findall(pattern4, text_lower)
    for match in matches:
        articles.add(f"Article {match}")
    
    # Pattern 5: "Kifungu cha 49" (Swahili)
    pattern5 = r'(?:kifungu|Kifungu)\s+cha\s+(\d+)'
    matches = re.findall(pattern5, text_lower)
    for match in matches:
        articles.add(f"Kifungu {match}")
    
    # Pattern 6: "Ibara 49" (Swahili for Article)
    pattern6 = r'(?:ibara|Ibara)\s+(\d+)'
    matches = re.findall(pattern6, text_lower)
    for match in matches:
        articles.add(f"Ibara {match}")
    
    # If no articles found, return ["Unknown"] but try to extract any numbers
    if not articles:
        # Last resort: find any numbers that might be articles
        any_numbers = re.findall(r'\b(\d{1,3})\b', text)
        if any_numbers:
            # Take first number as probable article
            return [f"Article {any_numbers[0]} (probable)"]
        return ["Unknown"]
    
    return sorted(list(articles), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

def load_constitution_pdf():
    """Load the Kenyan Constitution PDF"""
    pdf_path = Path("Data/kenya_constitution_2010.pdf")
    
    if not pdf_path.exists():
        logger.error(f"Constitution PDF not found at {pdf_path.absolute()}")
        logger.error("Please download it from: http://kenyalaw.org/kl/index.php?id=398")
        return None
    
    logger.info(f"Loading {pdf_path.name}...")
    
    try:
        content = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    # Add page markers for reference
                    content += f"\n[PAGE {page_num}]\n{page_text}\n"
                
                if page_num % 20 == 0:
                    logger.info(f"Processed {page_num}/{total_pages} pages")
        
        logger.info(f"✅ Loaded {len(content)} characters from PDF")
        return content
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks (renamed to avoid conflict)"""
    # Split by pages first to maintain structure
    pages = text.split('[PAGE ')
    
    chunks = []
    current_chunk = ""
    
    for page in pages[1:]:  # Skip first empty split
        page_num = page.split(']')[0]
        page_content = page.split(']\n')[1] if ']\n' in page else page
        
        # Split page content into paragraphs
        paragraphs = page_content.split('\n\n')
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size * 5:
                current_chunk += f"\n[Page {page_num}] {para}\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = f"\n[Page {page_num}] {para}\n"
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def index_constitution():
    """Main function to index the constitution with enhanced metadata"""
    print("\n" + "="*60)
    print("📚 Indexing Kenyan Constitution")
    print("="*60)
    
    # Load PDF
    print("\n📖 Step 1: Loading PDF...")
    text = load_constitution_pdf()
    if not text:
        return False
    
    # Create chunks - USING RENAMED FUNCTION
    print("\n✂️ Step 2: Creating chunks...")
    chunks = split_into_chunks(text)
    
    # Create or get collection
    print("\n💾 Step 3: Creating ChromaDB collection...")
    try:
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection")
        except:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add chunks to collection in smaller batches
        print("\n📤 Step 4: Adding chunks to vector database...")
        batch_size = 10
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            ids = [f"chunk_{j}" for j in range(i, i+len(batch))]
            
            # Prepare metadata for each chunk
            metadatas = []
            for idx, chunk_content in enumerate(batch):  # Renamed variable to chunk_content
                # Extract ALL articles from this chunk
                articles = extract_article_numbers(chunk_content)
                
                # Store primary article and all articles
                metadata = {
                    "source": "Kenyan Constitution 2010",
                    "chunk": i + idx,
                    "article": articles[0] if articles else "Unknown",
                    "all_articles": ",".join(articles) if articles else "Unknown"
                }
                metadatas.append(metadata)
            
            # Add with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    collection.add(
                        documents=batch,
                        ids=ids,
                        metadatas=metadatas
                    )
                    logger.info(f"✅ Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for batch {i//batch_size + 1}")
                        time.sleep(2)
                    else:
                        logger.error(f"Failed to add batch after {max_retries} attempts: {e}")
        
        print(f"\n✅ Successfully indexed {len(chunks)} chunks!")
        
        # Verify indexing
        print("\n🔍 Step 5: Verifying index...")
        test_search = search_constitution("article 49", n_results=3)
        if test_search and test_search.get('metadatas'):
            print("✅ Verification successful - articles detected:")
            for meta in test_search['metadatas'][0][:3]:
                if meta.get('all_articles'):
                    print(f"   • Found: {meta['all_articles']}")
        else:
            print("⚠️ Verification returned no results - check your PDF")
        
        return True
        
    except Exception as e:
        logger.error(f"Error indexing: {e}")
        return False

def search_constitution(query, n_results=5):
    """Search the constitution for relevant chunks"""
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return None

def ask_groq(prompt, model=DEFAULT_MODEL, max_retries=2):
    """Ask Groq with retry logic and timeout handling"""
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are UrRight, a helpful assistant that explains the Kenyan Constitution (2010) to citizens in simple language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1}/{max_retries} for Groq request")
                time.sleep(1)
            else:
                logger.error(f"Groq error after {max_retries} attempts: {e}")
                return None

async def interactive_mode():
    """Interactive Q&A mode"""
    # First index the constitution
    if not index_constitution():
        return
    
    print("\n" + "="*60)
    print("🗣️  UrRight Chatbot - Ask about the Kenyan Constitution")
    print("="*60)
    print(f"Using model: {DEFAULT_MODEL}")
    print("Type 'exit' to quit, 'model' to change model\n")
    
    current_model = DEFAULT_MODEL
    
    sample_questions = [
        "What are my rights if arrested?",
        "How long can police hold me?",
        "What is the process for voting?",
        "Na haki zangu nikamatwa na polisi?"
    ]
    
    print("Sample questions you can ask:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                break
            elif question.lower() == 'model':
                print("\nAvailable models:")
                for model, desc in AVAILABLE_MODELS.items():
                    status = "✅ CURRENT" if model == current_model else "   "
                    print(f"  {status} {model}: {desc}")
                new_model = input("\nEnter model name: ").strip()
                if new_model in AVAILABLE_MODELS:
                    current_model = new_model
                    print(f"✅ Switched to {current_model}")
                else:
                    print("❌ Invalid model")
                continue
            
            if not question:
                continue
            
            print("🔍 Searching constitution...")
            results = search_constitution(question)
            
            if results and results['documents'] and results['documents'][0]:
                relevant_chunks = results['documents'][0]
                context = "\n\n---\n\n".join(relevant_chunks[:3])
                
                prompt = f"""You are UrRight, explaining the Kenyan Constitution (2010).

RELEVANT SECTIONS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer in simple language that anyone can understand
2. If the question is in Swahili, answer in Swahili
3. Cite the specific Article numbers when possible
4. If the answer isn't in the provided sections, say you don't know

ANSWER:"""
                
                print("🤔 Thinking...")
                answer = ask_groq(prompt, model=current_model)
                
                if answer:
                    print(f"\n📝 Answer: {answer}")
                    
                    # Show ALL sources found
                    if results['metadatas'] and results['metadatas'][0]:
                        print("\n📚 Sources:")
                        all_articles = set()
                        for meta in results['metadatas'][0]:
                            # Check for multiple articles in metadata
                            if meta.get('all_articles'):
                                articles = meta['all_articles'].split(',')
                                for article in articles:
                                    if article and article != "Unknown":
                                        all_articles.add(article.strip())
                            elif meta.get('article') and meta['article'] != "Unknown":
                                all_articles.add(meta['article'])
                        
                        # Display sorted articles
                        for article in sorted(all_articles, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
                            print(f"  • {article}")
                else:
                    print("❌ Failed to get response from Groq. Please try again.")
                
            else:
                print("❌ No relevant sections found in the constitution.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using UrRight.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("Sorry, an error occurred. Please try again.")

def quick_test():
    """Quick test to verify indexing worked"""
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "rights when arrested",
        "voting rights",
        "police detention"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search_constitution(query, n_results=2)
        if results and results['documents'] and results['documents'][0]:
            print(f"Found {len(results['documents'][0])} relevant chunks")
            if results['metadatas'] and results['metadatas'][0]:
                print("Articles found:")
                for meta in results['metadatas'][0]:
                    if meta.get('all_articles'):
                        print(f"  • {meta['all_articles']}")
                    elif meta.get('article'):
                        print(f"  • {meta['article']}")
        else:
            print("No results found")

if __name__ == "__main__":
    # quick_test()
    asyncio.run(interactive_mode())