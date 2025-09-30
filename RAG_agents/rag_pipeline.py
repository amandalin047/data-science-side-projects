import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_batch
import sqlparse
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
import ollama
from nltk.tokenize import sent_tokenize

class WebScraper:
    def __init__(self, search_keywords: list, search_domains: list = None,
                       num_results: int = 10, timeout: float = 10, ignore_length: int = 20):
        self.search_keywords = search_keywords
        self.search_domains = search_domains
        self.num_results = num_results
        self.timeout = timeout
        self.ignore_length = ignore_length
    
    def find_urls(self, num_results: int) -> list:
        urls = []
        for search_keyword in self.search_keywords:
            try:
                for result in search(search_keyword, num_results=num_results, advanced=True):
                    if self.in_domain(result.url):
                        urls.append(result.url)
            except Exception as e:
                print(f"Error searching for URLs: {e}")
        return urls
    
    def in_domain(self, url: str) -> bool:
        if self.search_domains is None:
            return True
        else:
            return any(domain in url for domain in self.search_domains)
    
    def fetch_content(self, url: str, timeout: float, ignore_length: int) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US"}
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            
            content = []        
            for element in soup.find_all(["p", "h1", "h2", "h3"]):
                text = element.get_text(strip=True)
                if text and len(text) > ignore_length:  # ignore short texts
                    content.append(text)
            return {"url": url, "content": " ".join(content)}
        
        except Exception as e:
            print(f"Error fetching content {url}: {e}")
            return {"url": url, "content": ""}
        
    def scrape(self):
        urls = self.find_urls(self.num_results)
        documents = []
        for url in urls:
            doc = self.fetch_content(url, self.timeout, self.ignore_length)
            if doc["content"]:
                documents.append(doc)   
        return documents
    

class Embedder:
    def __init__(self, model: str, documents: list, chunking=False, chunk_size=512):
        self.model = model
        self.documents = documents
        self.chunking = chunking
        self.chunk_size = chunk_size

    def preprocess(self):
        def clean_text(text):
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[^\w\s.,!?]", "", text)
            return text
        processed_docs = [{"url": doc["url"], "content": clean_text(doc["content"])} for doc in self.documents]
        return processed_docs
    
    def get_embeddings(self):
        if self.chunking:
            max_words = self.chunk_size

            def split_sentence_by_comma(sentence):
                return [part.strip() for part in sentence.split(",") if part.strip()]
            
            def add_to_chunk(chunks, current_chunk, current_length, words):
                if len(words) + current_length <= max_words:
                    current_chunk.extend(words)
                    current_length += len(words)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_length = words, len(words)
                return chunks, current_chunk, current_length
                   
            def chunk_text(text):
                sentences = sent_tokenize(text)
                chunks, current_chunk, current_length = [], [], 0
                for sentence in sentences:   
                    words = sentence.split()
                    if len(words) > max_words:
                        sub_sentences = split_sentence_by_comma(sentence)
                        for sub_sentence in sub_sentences:
                            sub_words = sub_sentence.split()
                            chunks, current_chunk, current_length = add_to_chunk(chunks, current_chunk, current_length, sub_words)
                    else:
                        chunks, current_chunk, current_length = add_to_chunk(chunks, current_chunk, current_length, words)

                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                return chunks
        else:
            def chunk_text(text):
                return [text]   
     
        model = self.model
        processed_docs = self.preprocess()
        
        embedded_docs = []
        for i, doc in enumerate(processed_docs):
            chunks = chunk_text(doc["content"])
            if len(chunks) > 1:
                print(f"Document {i+1} has been chunked into {len(chunks)} parts.")
            else:
                print(f"Document {i+1} is not chunked.")
            
            for chunk in chunks:
                try:
                    embedding = model.encode(chunk, normalize_embeddings=True)
                    embedded_docs.append({"document": f"doc_{i+1}", "url": doc["url"], "content": chunk, "embedding": embedding})
                except Exception as e:
                    print(f"""Error embedding {chunk} of doc_{i+1}: {e}""")
                    embedded_docs.append({"document": f"doc_{i+1}", "url": doc["url"], "content": chunk, "embedding": None})
                
        return embedded_docs        
    

def execute_sql_script(sql_script, connection, cursor, place_holders: tuple=()):
    statements = sqlparse.split(sql_script)
    for statement in statements:
        try:
            stmt = statement.strip()
            if stmt:
                if place_holders:
                    cursor.execute(stmt, place_holders)
                else:
                    cursor.execute(stmt)
        except psycopg2.Error as e:
            print(f"Error executing statement: {stmt}\nError: {e}")
            connection.rollback()
            raise
        except Exception as e:
            print(f"General error executing statement: {stmt}\nError: {e}")
            raise
    connection.commit()

class Database:
    def __init__(self, model, **dbparams):
        self.model = model
        self.dbparams = dbparams
        self.connection = psycopg2.connect(**dbparams)
        self.cursor = self.connection.cursor()
    
    def load_embeddings(self, table: str, embedded_docs: list, page_size: int = 100):
        register_vector(self.connection)

        data = [
            (   
                doc["document"],
                doc["url"],
                doc["content"],
                doc["embedding"]
            )
            for doc in embedded_docs
            if doc.get("embedding") is not None  # Skip documents with None embeddings
        ]

        statement = """
            INSERT INTO {} (document, url, content, embedding)
            VALUES (%s, %s, %s, %s)
        """.format(table)
        try:
            execute_batch(self.cursor, statement, data, page_size=page_size)
            self.connection.commit()
            print(f"Successfully loaded {len(data)} documents into {table}.")
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            self.connection.rollback()
            raise
        except Exception as e:
            print(f"General error: {e}")
            raise

    def create_hnsw_index(self, table):
        sql_script = """
            CREATE INDEX IF NOT EXISTS data_table_embedding_idx
            ON {}
            USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 100);
            """.format(table)
        execute_sql_script(sql_script, self.connection, self.cursor)
        print(f"HNSW index created successfully on {table}.")


    def similarity_search(self, query, table, top_k=3, similarity_threshold=0.8):
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        sql_script = """
            SELECT id, document, url, content, 1 - (embedding <=> %s) AS cosine_similarity
            FROM {}
            WHERE 1 - (embedding <=> %s) > %s
            ORDER BY embedding <=> %s
            LIMIT %s;
            """.format((table))
        execute_sql_script(sql_script, self.connection, self.cursor,
                           place_holders=(query_embedding, query_embedding, similarity_threshold, query_embedding, top_k))
        
        try:
            results = [
                {
                "id": row[0],
                "document": row[1],
                "url": row[2],
                "content": row[3],
                "cosine_similarity": row[4]
                }
            for row in self.cursor.fetchall()
            ]
            if results:
                print(f"Found {len(results)} results with similarity > {similarity_threshold}.\n")
                for result in results:
                    print(f"""ID: {result["id"]}\nDoc: {result["document"]}\nURL: {result["url"]}\nContent (first 100 chars): {result["content"][:100]}...\nSimilarity: {result["cosine_similarity"]:.4f}\n""")
            else:
                print(f"No results found with similarity > {similarity_threshold}.\n")
            return results
        
        except psycopg2.Error as e:
            print(f"Error during similarity search: {e}")
            raise

    def get_tsvector(self, table):
        sql_script = """
            ALTER TABLE {} ADD COLUMN IF NOT EXISTS tsv TSVECTOR;
            UPDATE {} SET tsv = to_tsvector('english', content);
            CREATE INDEX IF NOT EXISTS idx_fts ON {} USING GIN(tsv);
            CREATE TRIGGER tsvectorupdate
            BEFORE INSERT OR UPDATE ON {}
            FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger(tsv, 'pg_catalog.english', content);
            """.format(table, table, table, table)
        
        execute_sql_script(sql_script, self.connection, self.cursor)
        print("TSVECTOR and GIN index created successfully.")

    def bm25_search(self, query, table, top_k=3):
        sql_script = """
            SELECT id, document, url, content, ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
            FROM {}
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
            """.format(table)
        execute_sql_script(sql_script, self.connection, self.cursor,
                           place_holders=(query, query, top_k))
        
        try:
            results = [
                {
                "id": row[0],
                "document": row[1],
                "url": row[2],
                "content": row[3],
                "bm25_score": row[4]
                }
            for row in self.cursor.fetchall()
            ]
            for result in results:
                print(f"""ID: {result["id"]}\nDoc: {result["document"]}\nURL: {result["url"]}\nContent (first 100 chars): {result["content"][:100]}...\nBM25: {result["bm25_score"]:.4f}\n""")
            return results
        
        except psycopg2.Error as e:
            print(f"Error during BM25: {e}")
            raise

    def hybrid_search(self, query, table, top_k=3):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        sql_script = """
            WITH bm25_results AS (
                SELECT id, document, url, content, ts_rank(tsv, plainto_tsquery('english', %s)) AS bm25_score
                FROM {}
                WHERE tsv @@ plainto_tsquery('english', %s)
            ),
            dense_results AS (
                SELECT id, document, url, content, 1 - (embedding <=> %s) AS cosine_similarity
                FROM {}
                ORDER BY embedding <=> %s
                LIMIT 50
            )
            SELECT d.id, 
                d.document,
                d.url,
                d.content,
                COALESCE(b.bm25_score, 0) AS bm25_score,
                d.cosine_similarity,
                (0.5 * COALESCE(b.bm25_score, 0) + 0.5 * d.cosine_similarity) AS final_score
            FROM dense_results d
            LEFT JOIN bm25_results b ON d.id = b.id
            ORDER BY final_score DESC
            LIMIT %s;
             """.format(table, table)
        execute_sql_script(sql_script, self.connection, self.cursor,
                           place_holders=(query, query, query_embedding, query_embedding, top_k))

        try:
            results = [
                {
                "id": row[0],
                "document": row[1],
                "url": row[2],
                "content": row[3],
                "hybrid_score": row[6]
                }
            for row in self.cursor.fetchall()
            ]
            for result in results:
                print(f"""ID: {result["id"]}\nDoc: {result["document"]}\nURL: {result["url"]}\nContent (first 100 chars): {result["content"][:100]}...\nBM25: {result["hybrid_score"]:.4f}\n""")
            return results
        
        except psycopg2.Error as e:
            print(f"Error during BM25: {e}")
            raise                         
        
    def close(self):
        self.cursor.close()
        self.connection.close()
        print("Database connection closed.")


class RAGbot:
    def __init__(self, call_llm, database, table):
        self.call_llm = call_llm
        self.database = database
        self.table = table
    
    def generate_answer(self, query: str, top_k: int = 3, method="dense", threshold=0.7, **llm_option_params):
        if method == "dense":
            results = self.database.similarity_search(query, self.table, top_k=top_k, similarity_threshold=threshold)
        elif method == "bm25":
            results = self.database.bm25_search(query, self.table, top_k=top_k)
        elif method == "hybrid":
            results = self.database.hybrid_search(query, self.table, top_k=top_k)
        else:
            print("Invalid method; defaults to dense search (cosine similarity).")
            results = self.database.similarity_search(query, self.table, top_k=top_k, similarity_threshold=threshold)
        
        if results:
            context = "\n".join([f"""Document {i+1}: {doc["content"]}""" for i, doc in enumerate(results)])
            print("Context retrieved successfully:\n", context)

            prompt = f"""        
                You are a helpful assistant. Please answer the following query using the provided context from relevant documents. Provide a concise and accurate response, citing key information from the context. If the context is insufficient, say so by replying with "I'm not sure based on the current information."

                Query: {query}

                Context:
                {context}

                Answer:
            """
            answer = self.call_llm(prompt, **llm_option_params)
        else:
            answer = None
        return answer


def ask_llama3(prompt, **options): 
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ],
        options=options
    )
    return response["message"]["content"]

        

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import nltk
    nltk.download("punkt_tab")

    search_domains = ["wikipedia.org", "github.com", "medium.com", "researchgate.net", "arxiv.org"]
    web_scraper = WebScraper(["Kuhn Poker", "game theory"], search_domains=search_domains,
                         num_results=20, timeout=30, ignore_length=0)
    documents = web_scraper.scrape()
    for doc in documents:
        print(f"""URL: {doc["url"]}\nContent (first 50 chars): {doc["content"][:50]}...\n""")


    model = SentenceTransformer("BAAI/bge-large-en")
    embedder = Embedder(model, documents, chunking=True, chunk_size=512)
    embedded_docs = embedder.get_embeddings()
    for chunk in embedded_docs:
        if chunk["embedding"] is not None:
            print(f"""URL: {chunk["document"]}, {chunk["content"][:20]}\nEmbedding dim: {len(chunk["embedding"])}\n""")
        else:
            print(f"""URL: {chunk["document"]}, {chunk["content"][:20]}\nEmbedding: Error occurred\n""")

    
    table, query, password = "testing.test1", "Explain Kuhn Poker in simple terms.", "MY_PASSWORD"
    database = Database(model=model, dbname="rag_demo", user="postgres", password=password, host="localhost", port="5432")
    database.load_embeddings(table, embedded_docs[:3], page_size=100)
    database.create_hnsw_index(table)
    database.get_tsvector(table)

    results_dense = database.similarity_search(query, table, top_k=3, similarity_threshold=0.7)
    results_bm25 = database.bm25_search(query, table, top_k=3)
    results_hybrid = database.hybrid_search(query, table, top_k=3)

    for search_method, results in zip(["dense\n", "bm25\n", "hybrid\n"], [results_dense, results_bm25, results_hybrid]):
        print(search_method, "  ", results)

    rag_bot = RAGbot(call_llm=ask_llama3, database=database, table="testing.test1")
    answer = rag_bot.generate_answer(query, top_k=3, threshold=0.7, temperature=0.2)
    print("LLM answer:\n", answer)


    import gradio as gr
    def gradio_interface(query, method, top_k):
        try:
            answer = rag_bot.generate_answer(query, top_k=int(top_k), method=method)
            return answer
        except Exception as e:
            return f"Error: {e}"

    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Your Question"),
            gr.Radio(["dense", "bm25", "hybrid"], value="dense", label="Search Method"),
        gr.Slider(1, 10, value=3, label="Top K Results"),
    ],
    outputs=gr.Textbox(label="LLM Answer"),
    title="RAG QA Bot",
    description="Ask a question. The bot will retrieve documents using your selected method and generate an answer.",
    )

    iface.launch()




