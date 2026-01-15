"""
Tool Functions for Voice Agent

This module provides external tools that the LLM can call:
- search_arxiv: Search for academic papers on arXiv
- calculate: Evaluate mathematical expressions
"""

from typing import Dict, Any
import requests
from sympy import sympify, SympifyError


def search_arxiv(query: str) -> str:
    """
    Search arXiv for papers matching the query.
    
    Args:
        query: Search query string (e.g., "quantum entanglement")
        
    Returns:
        str: Formatted search results or error message
    """
    print(f"[TOOLS] Searching arXiv for: '{query}'")
    
    try:
        # arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 3,  # Get top 3 results
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code != 200:
            return f"Error: arXiv API returned status {response.status_code}"
        
        # Parse XML response (arXiv returns XML)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Extract entries
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        
        if not entries:
            return f"No arXiv papers found for query: '{query}'"
        
        # Format results
        results = []
        for i, entry in enumerate(entries, 1):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            # Truncate summary to first 150 characters
            summary_short = summary[:150] + "..." if len(summary) > 150 else summary
            results.append(f"{i}. {title}\n   Summary: {summary_short}")
        
        result_text = f"Found {len(entries)} papers on arXiv:\n\n" + "\n\n".join(results)
        print(f"[TOOLS] arXiv search successful: {len(entries)} results")
        return result_text
        
    except requests.exceptions.Timeout:
        error_msg = "arXiv search timed out. Please try again."
        print(f"[TOOLS] ERROR: {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Error searching arXiv: {str(e)}"
        print(f"[TOOLS] ERROR: {error_msg}")
        return error_msg


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely using sympy.
    
    Args:
        expression: Mathematical expression as string (e.g., "2+2", "sqrt(16)", "sin(pi/2)")
        
    Returns:
        str: Calculation result or error message
    """
    print(f"[TOOLS] Calculating: '{expression}'")
    
    try:
        # Sanitize expression: remove python module prefixes that confuse sympy
        clean_expression = expression.replace("math.", "").replace("numpy.", "").replace("np.", "").replace("Math.", "")
        
        # Use sympy for safe evaluation
        result = sympify(clean_expression)
        
        # Try to evaluate to a numerical value if possible
        try:
            numerical_result = float(result.evalf())
            result_str = str(numerical_result)
        except:
            result_str = str(result)
        
        print(f"[TOOLS] Calculation result: {result_str}")
        return f"The result is: {result_str}"
        
    except SympifyError as e:
        error_msg = f"Invalid mathematical expression: {expression}"
        print(f"[TOOLS] ERROR: {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Error calculating expression: {str(e)}"
        print(f"[TOOLS] ERROR: {error_msg}")
        return error_msg


# Tool registry - maps function names to actual functions
AVAILABLE_TOOLS: Dict[str, Any] = {
    "search_arxiv": search_arxiv,
    "calculate": calculate
}


def get_tool_descriptions() -> str:
    """
    Get descriptions of available tools for the LLM prompt.
    
    Returns:
        str: Formatted tool descriptions
    """
    return """Available tools:
1. search_arxiv(query: str) - Search arXiv for academic papers
   Example: {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}

2. calculate(expression: str) - Evaluate a mathematical expression
   Example: {"function": "calculate", "arguments": {"expression": "2+2*3"}}
"""






