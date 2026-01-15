# Changes Made for Homework 6

## Overview
Implemented tool selection logic to enable the Voice Assistant to intelligently choose between performing calculations, searching arXiv, or responding with natural language.

## Modified Files

### 1. `voice_assistant_api.py`
- **Change**: Updated the import statement to use the new LLM module with function calling capabilities.
- **Before**: `from llm import generate_response`
- **After**: `from llm_with_functions import generate_response_with_functions as generate_response`
- **Reason**: To connect the API endpoints to the tool-aware LLM implementation.

### 2. `tools.py`
- **Change**: Added input sanitization to the `calculate` function.
- **Details**:
  - Implemented `.replace()` calls to strip `math.`, `Math.`, `numpy.`, and `np.` prefixes from expressions.
- **Reason**: The LLM often outputs Python-style syntax (e.g., `math.sqrt(16)`), which previously caused `SympifyError`. The tool now safely handles these prefixes.

### 3. `llm_with_functions.py`
- **Change**: Refined the System Prompt for purely robust tool selection.
- **Details**:
  - Defined explicit **RULES** for when to use tools vs. natural language.
  - Added a strict "ARITHMETIC" rule to force `calculate` tool usage for math queries (e.g., "What is the square root of...").
  - simplified the prompt structure to be more directive for the 1B model, reducing "chatty" explanations from the assistant.
- **Reason**: 
  - To prevent hallucinations (e.g., inventing a `capitals` tool).
  - To ensure math questions phrased as sentences are correctly routed to the calculator.
  - To stop the model from explaining its internal logic to the user.
