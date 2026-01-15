"""
Voice Assistant API - FastAPI Backend

This module provides the REST API endpoints for the Voice Assistant System.
It orchestrates the ASR, LLM, and TTS modules to process audio input and generate audio responses.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from asr import transcribe_audio
from llm_with_functions import generate_response_with_functions as generate_response
from tts import synthesize_speech
from conversation import conversation_manager

# Initialize FastAPI application
app = FastAPI(
    title="Voice Assistant API",
    description="REST API for voice-based conversational AI assistant",
    version="1.0.0"
)

# Enable CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    """
    Main endpoint for voice conversation.
    
    Accepts an audio file, transcribes it, generates a response using the LLM,
    converts the response to speech, and returns the audio file.
    
    Args:
        file: Audio file upload (WAV format recommended)
        
    Returns:
        FileResponse: Audio response in WAV format
        
    Raises:
        HTTPException: On validation or processing errors
    """
    print("\n" + "="*60)
    print("[API] /chat/ endpoint called")
    print("="*60)
    
    try:
        # Validate file type
        print(f"[API] Received file: {file.filename}, content_type: {file.content_type}")
        
        if file.content_type and not file.content_type.startswith("audio/"):
            print(f"[API] ERROR: Invalid file type")
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Read audio bytes from uploaded file
        audio_bytes = await file.read()
        print(f"[API] Audio file size: {len(audio_bytes)} bytes")
        
        if not audio_bytes:
            print(f"[API] ERROR: Empty audio file")
            raise HTTPException(
                status_code=400,
                detail="Empty audio file received."
            )
        
        # ASR: Transcribe audio to text
        print(f"[API] Starting ASR (speech-to-text)...")
        try:
            user_text = transcribe_audio(audio_bytes)
            print(f"[API] ASR result: '{user_text}'")
        except Exception as e:
            print(f"[API] ASR ERROR: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Speech recognition failed: {str(e)}"
            )
        
        # LLM: Generate response
        print(f"[API] Starting LLM (response generation)...")
        try:
            assistant_text = generate_response(user_text)
            print(f"[API] LLM result: '{assistant_text}'")
        except Exception as e:
            print(f"[API] LLM ERROR: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Response generation failed: {str(e)}"
            )
        
        # TTS: Convert response text to audio
        print(f"[API] Starting TTS (text-to-speech)...")
        try:
            audio_path = synthesize_speech(assistant_text)
            print(f"[API] TTS result: {audio_path}")
        except Exception as e:
            print(f"[API] TTS ERROR: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Speech synthesis failed: {str(e)}"
            )
        
        # Return audio file as response
        print(f"[API] Returning audio response")
        print("="*60 + "\n")
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename="response.wav"
        )
        
    except HTTPException:
        print("="*60 + "\n")
        raise
    except Exception as e:
        print(f"[API] UNEXPECTED ERROR: {str(e)}")
        print("="*60 + "\n")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        dict: Status information including service name and health status
    """
    return {
        "status": "healthy",
        "service": "voice-assistant"
    }


@app.get("/history")
async def get_history():
    """
    Retrieve conversation history.
    
    Returns the current conversation history to enable frontend synchronization
    with backend state.
    
    Returns:
        dict: Conversation history with list of messages
    """
    history = conversation_manager.get_history()
    print(f"[API] /history endpoint called - returning {len(history)} messages")
    return {
        "history": history
    }


@app.post("/clear")
async def clear_conversation():
    """
    Clear conversation history.
    
    Resets the conversation state by clearing all stored conversation turns.
    This allows users to start a fresh conversation.
    
    Returns:
        dict: Success message confirming the conversation was cleared
    """
    print("\n" + "="*60)
    print("[API] /clear endpoint called - clearing conversation history")
    print("="*60 + "\n")
    conversation_manager.clear()
    return {
        "status": "success",
        "message": "Conversation history cleared"
    }
