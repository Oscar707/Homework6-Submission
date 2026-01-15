"""
LLM Module with Function Calling Support

This enhanced version of the LLM module adds function calling capabilities.
The LLM can now call external tools (search_arxiv, calculate) by outputting JSON function calls.
"""

import requests
from conversation import conversation_manager
from function_router import route_llm_output
from tools import get_tool_descriptions


class LLMModuleWithFunctions:
    """Large Language Model module with function calling support using Ollama."""
    
    def __init__(self, model_name: str = "llama3.2:1b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize LLM with Ollama and function calling support.
        
        Args:
            model_name: Ollama model name (default: llama3.2:1b)
            ollama_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        print(f"[LLM-FUNC] Initialized with function calling support")
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt with tool descriptions.
        
        Returns:
            str: System prompt instructing the LLM on tool usage
        """
        system_prompt = """You are a helpful AI assistant.

    RULES:
    1. ARITHMETIC: If the user asks clearly for a math calculation (e.g., "calculate", "what is 5 + 5", "square root of..."), YOU MUST RESPONSE WITH A JSON tool call for 'calculate'.
    2. RESEARCH: If the user asks to "search arxiv" or "find papers", use 'search_arxiv'.
    3. OTHER: For all other questions (capitals, jokes, history), respond normally in text. DO NOT use tools.
    
    TOOL OUTPUT FORMAT:
    {"function": "function_name", "arguments": {"arg_name": "arg_value"}}
    
    """ + get_tool_descriptions() + """
    
    Examples:
    User: "Calculate 25 * 4"
    Assistant: {"function": "calculate", "arguments": {"expression": "25*4"}}
    
    User: "What is the square root of 144?"
    Assistant: {"function": "calculate", "arguments": {"expression": "sqrt(144)"}}
    
    User: "Find papers on quantum computing"
    Assistant: {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}
    
    User: "What is the capital of Canada?"
    Assistant: The capital of Canada is Ottawa.
    
    IMPORTANT: Do not explain your tools. Just use them if needed. If no tool is needed, just speak."""
        
        return system_prompt
    
    def generate(self, user_text: str) -> str:
        """
        Generate response based on user input and conversation history.
        Now supports function calling via JSON output detection.
        
        Args:
            user_text: Current user message
            
        Returns:
            str: Generated assistant response (may be from tool execution)
        """
        # Add user message to history
        conversation_manager.add_user_message(user_text)
        
        # Construct prompt from history
        history = conversation_manager.get_history()
        
        # Build prompt with system instruction and tool descriptions
        prompt = self._build_system_prompt() + "\n\n"
        
        # Add conversation history (last 5 turns = 10 messages)
        for turn in history[-10:]:
            role = turn["role"]
            text = turn["content"]
            prompt += f"{role}: {text}\n"
        
        # Add the assistant prompt to trigger response generation
        prompt += "assistant:"
        
        print(f"[LLM-FUNC] Constructed prompt from {len(history)} messages")
        print(f"[LLM-FUNC] Prompt length: {len(prompt)} chars")
        
        # Generate response using Ollama
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more precise function calling
                        "num_predict": 150,  # Increased to allow for JSON output
                        "stop": ["\nuser:", "\nassistant:", "user:"],  # Stop at next turn
                        "num_ctx": 2048,
                        "num_thread": 4
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            
            # Extract the generated text
            result = response.json()
            llm_output = result.get("response", "").strip()
            
            print(f"[LLM-FUNC] Raw LLM output: '{llm_output[:150]}{'...' if len(llm_output) > 150 else ''}'")
            
            # Clean up any role prefixes
            for prefix in ["assistant:", "Assistant:", "ASSISTANT:"]:
                if llm_output.startswith(prefix):
                    llm_output = llm_output[len(prefix):].strip()
                    break
            
            # Route through function router
            # This will either execute a function call or return the original text
            final_response, was_function_call = route_llm_output(llm_output)
            
            if was_function_call:
                print(f"[LLM-FUNC] Function was called, result: '{final_response[:100]}{'...' if len(final_response) > 100 else ''}'")
            else:
                print(f"[LLM-FUNC] No function call, using original response")
            
            # Add the final response to history
            conversation_manager.add_assistant_message(final_response)
            
            return final_response
            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out after 120 seconds."
            print(f"[LLM-FUNC] ERROR: {error_msg}")
            fallback = "Sorry, that took too long. Try clearing history or asking a shorter question."
            conversation_manager.add_assistant_message(fallback)
            return fallback
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to Ollama: {str(e)}"
            print(f"[LLM-FUNC] ERROR: {error_msg}")
            fallback = "I'm having trouble connecting to the language model. Please make sure Ollama is running."
            conversation_manager.add_assistant_message(fallback)
            return fallback


# Global instance with function calling
llm_module_with_functions = LLMModuleWithFunctions()


def generate_response_with_functions(user_text: str) -> str:
    """
    Convenience function for response generation with function calling support.
    
    Args:
        user_text: User's input text
        
    Returns:
        str: Generated assistant response (may be from tool execution)
    """
    return llm_module_with_functions.generate(user_text)