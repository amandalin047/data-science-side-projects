from rag_pipeline import WebScraper, Embedder, Database
import rag_pipeline 
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from autogen import AssistantAgent, UserProxyAgent, register_function, GroupChat, GroupChatManager
from typing import List
from dataclasses import dataclass
import ollama

@dataclass
class AgentInfoParams:
    name: str
    system_message: str
    config: dict
    func: callable

class AgentSys:
    def __init__(self, table: str, **db_params):
        def ingest_articles(search_keywords: List[str], search_domains: List[str]=None,
                            num_results: int=20, timeout: float=10, ignore_length: int=20) -> Database:
            global global_db
            scraper = WebScraper(search_keywords=search_keywords, search_domains=search_domains,
                                 num_results=num_results, timeout=timeout, ignore_length=ignore_length)
            documents = scraper.scrape()

            embedder = Embedder(model=model, documents=documents, chunking=True)
            embedded_docs = embedder.get_embeddings()

            global_db = Database(model=model, **db_params)
            global_db.load_embeddings(table, embedded_docs)
            global_db.create_hnsw_index(table)
            if search_domains:
                print(f"Scraped, embedded, and stored {len(embedded_docs)} articles usinf search keywords {search_keywords} and search domains {search_domains}.")
            else :
                print(f"Scraped, embedded, and stored {len(embedded_docs)} articles using search keywords {search_keywords} (no search domains specified).")
            return global_db
        
        def answer_with_rag(query: str, db: Database,
                            top_k: int = 3, threshold: float = 0.8, method: str="dense",
                            **llm_option_params) -> str:
            if method == "dense":
                results = db.similarity_search(query, table, top_k=top_k, similarity_threshold=threshold)
            elif method == "bm25":
                results = db.bm25_search(query, table, top_k=top_k, similarity_threshold=threshold)
            elif method == "hybrid":
                results = db.hybrid_search(query, table, top_k=top_k, similarity_threshold=threshold)
            else:
                print("Invalid method; defaults to dense search (cosine similarity).")
                results = db.similarity_search(query, self.table, top_k=top_k, similarity_threshold=threshold)
            
            if not results:
                return "No relevant documents found above the similarity threshold."
            
            context = "\n".join([f"""Document {i+1}: {doc["content"]}""" for i, doc in enumerate(results)])
            print("Context retrieved for query:\n", context)

            prompt = f"""        
                 You are a helpful assistant. Please answer the following query using the provided context from relevant documents. Provide a concise and accurate response, citing key information from the context. If the context is insufficient, say so by replying with "I'm not sure based on the current information."

                 Query: {query}

                 Context:
                 {context}

                 Answer:
                 """
            try:
                answer = rag_pipeline.ask_llama3(prompt, **llm_option_params)
            except NameError as name_error:
                print(f"Name Error during LLM query: {str(name_error)}. Trying again with default LLM.")
                # not sure why im getting an error during ask_llama3 import from rag_pipeline
                def default_ask_llm(prompt, **options): 
                    response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    options=options
                   )
                    return response["message"]["content"]
                   
                answer = default_ask_llm(prompt, **llm_option_params)
            except Exception as e:
                print(f"Error during LLM query: {str(e)}.")
                answer = None
            return answer
        
        self.ingest_articles = ingest_articles
        self.answer_with_rag = answer_with_rag
        self.db_params = db_params
        self.default_agent_names_list = ["ScraperAgent", "EmbeddingAgent", "RAGAgent", "CoordinatorAgent"]

    def build_agents(self, agents_info: dict):
        default_agent_names_list = self.default_agent_names_list
        default_config = {
           "config_list": [{
           "model": "llama3",
           "api_key": "NotRequired",  
           "base_url": "localhost"  # Ollama
         }],
           "temperature": 0.7,
           "cache_seed": None
        }
        
        for name in default_agent_names_list:
            if name not in agents_info.keys():
                print(f"LLM Config not provided for default agent: {name}, using default config\n {default_config}")
                agents_info[name] = AgentInfoParams(name, None, default_config, None)
       
        coordinator_agent = AssistantAgent(
             name="CoordinatorAgent",
             system_message="You are the coordinator. You receive user queries, delegate tasks to appropriate agents, and compile the final response. For queries about NBA articles, instruct ScraperAgent and RAGAgent. For player stats, use StatsAgent and EfficiencyAgent. Ensure tasks are completed in order and results are clear.",
             llm_config=agents_info["CoordinatorAgent"].config
        )        
        scraper_agent = AssistantAgent(
            name="ScraperAgent",
            system_message="You are responsible for scraping NBA articles based on provided keywords. Use the ingest_nba_articles function to scrape, embed, and store articles. Return the database object.",
            llm_config=agents_info["ScraperAgent"].config
        )
        embedding_agent = AssistantAgent(
            name="EmbeddingAgent",
            system_message="You handle embedding documents and storing them in the database. Use the ingest_nba_articles function when instructed, and pass the database object to other agents.",
            llm_config=agents_info["EmbeddingAgent"].config
        )
        rag_agent = AssistantAgent(
             name="RAGAgent",
             system_message="You answer user queries using the RAG pipeline. Use the answer_with_rag function with the provided database, query, and method (dense, bm25, or hybrid). Ensure answers are concise and based on retrieved context.",
             llm_config=agents_info["RAGAgent"].config
        )
        default_agents = [coordinator_agent, scraper_agent, embedding_agent, rag_agent]
        for name, info in agents_info.items():
            if name in default_agent_names_list:
                agent = list(filter(lambda x: x.name == name, default_agents))[0]
                info.agent = agent
            else:
                agent = AssistantAgent(
                    name=name,
                    system_message=info.system_message,
                    llm_config=info.config
                )
                info.agent = agent
       
        self.agents_info = agents_info
        print(f"""Agents built successfully:
                  {", ".join([info.name for info in agents_info.values()])}""")

    def register_functions(self):
        def scraper_ingest(search_keywords: List[str], search_domains: List[str]=None,
                           num_results: int=20, timeout: float = 10, ignore_length: int = 20) -> str:
            global global_db
            try:
                db = self.ingest_articles(search_keywords=search_keywords, search_domains=search_domains,
                                              num_results=num_results, timeout=timeout, ignore_length=ignore_length)
                global_db = db  # stores Database object in shared state
                return "Successfully ingested articles."
            except Exception as e:
                return f"Error ingesting articles: {str(e)}"

        def rag_answer(query: str, threshold: float=0.7) -> str:
            global global_db
            if type(global_db) != Database:
                return "Error: Database not initialized. Please ingest articles first."
            return self.answer_with_rag(query, global_db, threshold=threshold)
        
        agents_info, default_agent_names_list = self.agents_info, self.default_agent_names_list
        register_function(
            scraper_ingest,
            caller=agents_info["ScraperAgent"].agent,
            executor=agents_info["EmbeddingAgent"].agent,
            name="scraper_ingest",
            description="Scrapes and ingests articles based on keywords."
        )
        register_function(
             rag_answer,
             caller=agents_info["RAGAgent"].agent,
             executor=agents_info["RAGAgent"].agent,
             name="rag_answer",
             description="Answers queries using the RAG pipeline."
        )
        for name, info in agents_info.items():
            if name not in default_agent_names_list:
                register_function(
                    info.func,
                    caller=info.agent,
                    executor=info.agent,
                    name=name + "_func",
                    description=f"Handles {info.func.__doc__} tasks."
                )
        print("Successfully registered functions for agents.")

    def setup_chat(self, manager_config):
        user_proxy = UserProxyAgent(
            name="UserProxy",
            system_message="You are the user interface. Forward user queries to the CoordinatorAgent and return the final response.",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "coding", "use_docker": False}
        )
        self.user_proxy = user_proxy
        group_chat = GroupChat(
            agents=[user_proxy]+[info.agent for info in self.agents_info.values()],
            messages=[],
            max_round=20
        )
        group_chat_manager = GroupChatManager(
             groupchat=group_chat,
             llm_config=manager_config
       )
        self.group_chat_manager = group_chat_manager

    def start(self, message: str):
        self.user_proxy.initiate_chat(
        self.group_chat_manager,
        message=message
    )


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    def fetch_player_stats(player_name: str):
        matches = players.find_players_by_full_name(player_name)
        if not matches:
            return f"No player found matching '{player_name}'."
    
        player_id = matches[0]["id"]
        stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        df = stats.get_data_frames()[0]
        return df.tail(1).to_dict(orient="records")[0]  # latest season

    def compute_player_efficiency(stats_dict: dict):
        points = stats_dict.get("PTS", 0)
        rebounds = stats_dict.get("REB", 0)
        assists = stats_dict.get("AST", 0)
        games = stats_dict.get("GP", 1)
        return {"efficiency_score": (points + rebounds + assists) / games}
    
    db_params = {"dbname": "rag_demo", "user": "postgres", "password": "Macbook58115", "host": "localhost", "port": "5432"}
    model = SentenceTransformer("BAAI/bge-large-en")
    
    agent_sys = AgentSys("nba_articles", **db_params)

    llm_config = {
    "config_list": [{
        "model": "gpt-4o",
        "api_key": "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",
        "api_type": "openai"
    }],
    "temperature": 0.2,
    "cache_seed": None
    }
    
    agents_info = {}
    for name in agent_sys.default_agent_names_list:  
        agents_info[name] = AgentInfoParams(name, None, llm_config, None)     
    agents_info["StatsAgent"] = AgentInfoParams("StatsAgent",
                                                 "You are responsible for scraping NBA articles based on provided keywords. Use the ingest_nba_articles function to scrape, embed, and store articles.",
                                                 llm_config,
                                                 fetch_player_stats)
    agents_info["EfficiencyAgent"] = AgentInfoParams("EfficiencyAgent",
                                                      "You are responsible for computing player efficiency. Use the compute_player_efficiency function to calculate efficiency based on stats.",
                                                      llm_config,
                                                      compute_player_efficiency)
    
    print("============================================")
    
    print("Building and registering agents...")  
    agent_sys.build_agents(agents_info)
    agent_sys.register_functions()
    agent_sys.setup_chat(llm_config)
    
    query = "What are Luka Doncic's latest stats and performance?"
    message = f"""
        CoordinatorAgent, please handle the following query:
        '{query}'
        For the performance part, instruct ScraperAgent to ingest articles using keywords ["Luka Doncic"], then RAGAgent to answer using similarity search.
        For the stats part, fetch Luka Doncic's latest season stats and compute his efficiency score.
        """
    agent_sys.start(message)
    

    
