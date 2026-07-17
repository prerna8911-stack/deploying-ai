
import requests
import pandas as pd
from typing import Dict, Any, Optional
import chromadb
import numpy as np
from typing import List, Tuple, Optional
from typing import List, Optional
import requests
from typing import List, Dict, Any
from datetime import datetime
import json
from IPython.display import display, Markdown
from gradio.components import Chatbot
from gradio import Interface
# Note: this file defines its own services and MemoryManager classes below.
# Avoid importing a top-level `app` package which doesn't exist in this workspace.


class APIService:
    """Service 1: Transform and rephrase API responses"""
    
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json"
        }
    
    def get_api_data(self, query: str) -> Optional[Dict]:
        """Fetch data from external API"""
        try:
            # Example: Using a weather API (WeatherAPI.io - free tier)
            params = {"q": query, "appid": api_key}
            response = requests.get(self.api_url, params=params, headers={"X-API-KEY": api_key})
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"Error fetching API data: {e}")
            return None
    
    def transform_response(self, api_data: Dict, query: str) -> str:
        """Transform raw API data into natural language"""
        
        # Example: Weather API transformation
        try:
            temp = api_data.get("main", {}).get("temp", "unknown")
            city = api_data.get("name", "Unknown Location")
            description = api_data.get("weather", [{}])[0].get("description", "Unknown")
            
            # Transform to natural language (NOT verbatim)
            response = f"""
            🌤️ Weather Update for {city}:
            
            Current Temperature: {temp}°C
            
            Condition: {description.capitalize()}
            
            Recommendation: It is {self._generate_recommendation(description, temp)}.
            
            Would you like more weather details or another topic?
            """
            
            return response.strip()
        
        except (KeyError, IndexError) as e:
            return f"""
            ⚠️ Something went wrong processing your weather query.
            
            API returned incomplete data. Please try again with a valid city name.
            """
    
    def _generate_recommendation(self, description: str, temp: str) -> str:
        """Generate dynamic recommendations based on weather"""
        if "clear" in description.lower():
            return "Perfect day for outdoor activities!"
        elif "rain" in description.lower():
            return "Keep an umbrella handy today."
        elif "storm" in description.lower():
            return "Stay indoors and stay safe."
        else:
            return "Standard conditions, enjoy your day."
        

class SemanticQueryService:
    """Service 2: Semantic search using ChromaDB"""
    
    def __init__(self, db_path: str = "data/chroma"):
        # Initialize persistent client
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            "knowledge_base"
        )
        
        # Embedding model
        self.embedding_model = "all-MiniLM-L6-v2"
        self.embedding_function = None
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector store"""
        
        if self.embedding_function is None:
            self._setup_embedding()
        
        embeddings = [
            self.embedding_function([doc.get("content", "")]).flatten()
            for doc in documents
        ]
        
        metadata = [
            {
                "title": doc.get("title", "Untitled"),
                "category": doc.get("category", "general"),
                "source": doc.get("source", "")
            }
            for doc in documents
        ]
        
        ids = [doc.get("id") for doc in documents]
        
        # Add documents to vector store
        self.collection.add(
            embeddings=embeddings,
            documents=[doc.get("content", "") for doc in documents],
            metadatas=metadata,
            ids=ids
        )
    
    def _setup_embedding(self):
        """Setup sentence transformer for embeddings"""
        from sentence_transformers import SentenceTransformer
        
        self.embedding_function = SentenceTransformer(self.embedding_model)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for documents semantically"""
        
        query_embedding = self.embedding_function([query])[0]
        self.embedding_function = self.embedding_function
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            (result, result_dist)
            for result, result_dist in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            )
        ]
    
class FunctionCallingService:
    """Service 3: Function Calling for external actions"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None  # Initialize when you have API key
    
    def get_current_time(self) -> str:
        """Example function: Get current time"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def get_stock_price(self, symbol: str) -> str:
        """Example function: Get stock price"""
        # Mock API call (replace with real API)
        mock_prices = {
            "AAPL": "$185.45",
            "GOOGL": "$156.32",
            "MSFT": "$412.75"
        }
        return mock_prices.get(symbol, "Data unavailable")
    
    def tool_description(self) -> List[Dict]:
        """Define tools for function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time in a readable format",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            }
        ]
    
    def execute_function(self, response: Dict, user_query: str) -> str:
        """Execute tool calls based on function calling response"""
        
        tool_calls = response.get("tool_calls")
        
        if not tool_calls:
            return response.get("content", "No action needed")
        
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            
            if function_name == "get_current_time":
                result = self.get_current_time()
                results.append(f"⏰ **Current Time:** {result}")
            
            elif function_name == "get_stock_price":
                symbol = tool_call.get("function", {}).get("arguments", {}).get("symbol")
                price = self.get_stock_price(symbol)
                results.append(f"📊 **Stock Price for {symbol}:** {price}")
        
        return " ".join(results)
    

class Guardrail:
    """Block restricted topics and system prompt"""
    
    RESTRICTED_TOPICS = ["cat", "dogs", "horoscope", "zodiac", "taylor swift"]
    RESTRICTED_WORDS = ["system prompt", "instructions", "prompt engineering"]
    
    def is_blocked(self, text: str) -> bool:
        """Check if user input is blocked"""
        text_lower = text.lower()
        
        # Check for restricted topics
        for topic in self.RESTRICTED_TOPICS:
            if topic in text_lower:
                return True
        
        # Check for system prompt access attempts
        for word in self.RESTRICTED_WORDS:
            if word in text_lower:
                return True
        
        return False
    
    def get_rejection_message(self, blocked_text: str) -> str:
        """Return appropriate rejection message"""
        
        if "cat" in blocked_text.lower() or "dog" in blocked_text.lower():
            return "🐾 I'm not programmed to talk about pets or animals. I'm here to help you with weather, stocks, or answering your questions! 😊"
        
        if "horoscope" in blocked_text.lower() or "zodiac" in blocked_text.lower():
            return "✨ I don't have access to horoscopes or zodiac signs. I'm better at answering factual questions! 🌟"
        
        if "taylor swift" in blocked_text.lower():
            return "🎵 I'm not familiar with that artist, but I'd love to help you with weather, stocks, or other questions! 😊"
        
        if any(word in blocked_text.lower() for word in self.RESTRICTED_WORDS):
            return "🔒 I can't help with that. I'm designed to provide helpful information about the world, not share system instructions!"
        
        return "I'm sorry, I can't assist with that request."

class MemoryManager:
    """Manage conversation memory (short-term)"""
    
    MAX_MESSAGES = 6  # Short-term memory
    MEMORY_FILE = "data/chat_history.json"
    
    def __init__(self):
        self.conversations = []
        self.max_turns = self.MAX_MESSAGES
        self.load_memory()
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add message to conversation history"""
        conversation = next((c for c in self.conversations if c["id"] == conversation_id), None)
        
        if conversation is None:
            conversation = {
                "id": conversation_id,
                "messages": []
            }
            self.conversations.append(conversation)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        conversation["messages"].append(message)
        
        # Trim if exceeds max turns
        if len(conversation["messages"]) > self.max_turns:
            conversation["messages"] = conversation["messages"][-self.max_turns:]
        
        self.save_memory()
    
    def get_conversation(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        conversation = next((c for c in self.conversations if c["id"] == conversation_id), None)
        return conversation["messages"] if conversation else []
    
    def load_memory(self):
        """Load memory from file"""
        try:
            import json
            with open(self.MEMORY_FILE, "r") as f:
                self.conversations = json.load(f)
        except FileNotFoundError:
            self.conversations = []
    
    def save_memory(self):
        """Save memory to file"""
        import json
        with open(self.MEMORY_FILE, "w") as f:
            json.dump(self.conversations, f, indent=2)
    
    def get_history(self, conversation_id: str) -> List[Dict]:
        """Get cleaned conversation history (user+assistant pairs)"""
        messages = self.get_conversation(conversation_id)
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    

class ChatController:
    """Orchestrates the AI chat system"""
    
    def __init__(self, conversation_id: str, memory: MemoryManager):
        self.conversation_id = conversation_id
        self.memory = memory
    
    def handle_message(self, user_message: str) -> str:
        """Process user message and return response"""
        
        # 1. Check guardrails first
        if guardrail.is_blocked(user_message):
            return guardrail.get_rejection_message(user_message)
        
        # 2. Determine service based on query
        response_text = self._route_query(user_message)
        
        # 3. Add to memory
        self.memory.add_message(self.conversation_id, "user", user_message)
        self.memory.add_message(self.conversation_id, "assistant", response_text)
        
        return response_text
    
    def _route_query(self, query: str) -> str:
        """Route query to appropriate service"""
        
        query_lower = query.lower()
        
        # Check for API service triggers
        if "weather" in query_lower or "temp" in query_lower:
            return self._handle_api_query(query)
        
        # Check for semantic search triggers
        if "what is" in query_lower or "define" in query_lower or "explain" in query_lower:
            return self._handle_semantic_search(query)
        
        # Check for function calling triggers
        if "time" in query_lower or "current time" in query_lower:
            return self._handle_function_call(query)
        
        # Default: Use a general LLM response
        return self._handle_default_query(query)
    
    def _handle_api_query(self, query: str) -> str:
        """Use API service"""
        api_data = api_service.get_api_data(query)
        
        if api_data is None:
            return "⚠️ Could not fetch weather data. Please check the API service is running."
        
        response = api_service.transform_response(api_data, query)
        return response
    
    def _handle_semantic_search(self, query: str) -> str:
        """Use semantic search service"""
        results = semantic_service.search(query, top_k=3)
        
        if not results:
            return f"⚠️ I couldn't find any relevant information about '{query}' in my knowledge base."
        
        summary_parts = []
        for doc, distance in results:
            summary_parts.append(f"- **{doc['title']}**: {doc['content'][:200]}... (Confidence: {distance:.2f})")
        
        response = f"""
        📚 **Knowledge Base Results for: {query}**
        
        Here are the most relevant documents I found:
        
        """ + "\n".join(summary_parts) + """
        
        Would you like me to summarize these or find more specific information?
        """
        
        return response
    
    def _handle_function_call(self, query: str) -> str:
        """Use function calling service"""
        # Call LLM with function calling parameters
        response_data = {
            "tool_calls": [
                {
                    "function": {
                        "name": "get_current_time"
                    }
                }
            ],
            "content": f"You asked about the time. Here is the current time."
        }
        
        return function_service.execute_function(response_data, query)
    
    def _handle_default_query(self, query: str) -> str:
        """Handle general queries with LLM"""
        # Simplified default response
        return f"""
        👋 **{query}**
        
        That's an interesting question! As a conversational AI, I can help you with:
        
        - **Weather updates** (Ask about your location)
        - **Stock prices** (Ask for stock symbols)
        - **General knowledge** (Ask me questions about the world)
        
        Just let me know what you'd like to know! 😊
        """
    
class ChatApp:
    """Main Gradio Chat Interface"""
    
    def __init__(self):
        self.conversation_id = "conversation_1"
        self.memory = MemoryManager()
        self.controller = ChatController(self.conversation_id, self.memory)
        
        # Set distinct personality
        self.personality = {
            "name": "WeatherBot Alex",
            "description": "A friendly, helpful AI assistant with a focus on practical information",
            "tone": "friendly",
            "style": "clear and concise"
        }
        
        # Build chat interface
        self.chatbot = Chatbot()
    
    def chat_bot(self, user_message: str, history: list, **kwargs):
        """Chat function that processes user messages"""
        
        # Clear chat if no history
        if not history:
            return "", []
        
        # Get last user message
        last_user_message = user_message
        
        # Get conversation history
        history_state = self.memory.get_conversation(self.conversation_id)
        
        # Add to state if not in history
        if {"role": "user", "content": last_user_message} not in history_state:
            history_state.append({"role": "user", "content": last_user_message})
        
        # Generate response
        response = self.controller.handle_message(last_user_message)
        
        # Format response as markdown
        display(Markdown(response))
        
        # Save to memory
        self.memory.add_message(self.conversation_id, "assistant", response)
        
        # Return updated history
        return response, history + [{"role": "assistant", "content": response}]
    
    def create_app(self):
        """Create Gradio app"""
        
        return Chatbot(
            value=[
                {
                    "role": "assistant",
                    "value": f"👋 Hi! I'm **WeatherBot Alex**. I can help you with:\n\n- Weather updates 🌤️\n- Knowledge search 📚\n- Time and stock info ⏰\n\nHow can I help you today?"
                }
            ]
        )
    
# Initialize services
api_service = APIService(api_url="YOUR_API_URL", api_key="YOUR_API_KEY")
semantic_service = SemanticQueryService(db_path="data/chroma")
function_service = FunctionCallingService(model="gpt-4o-mini")
guardrail = Guardrail()

# Create chat app
chat_app = ChatApp()

# Run Gradio app
if __name__ == "__main__":
    app = Interface(
        fn=lambda message: chat_app.controller.handle_message(message),
        inputs="text",
        outputs="text",
        title="WeatherBot AI Assistant",
        description="A conversational AI that can help with weather, knowledge, and more!",
        theme="soft"
    )
    
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)



