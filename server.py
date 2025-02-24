import os
import gc
import json
import time
import psutil
import uvicorn
import secrets
import logging
import asyncio
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import TextIteratorStreamer
from threading import Thread
from typing import List, Optional
from pydantic import BaseModel
from lib.models import ModelManager, AVAILABLE_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Server Configuration
PORT = int(os.getenv("PORT", str(config['server'].get('port', 8000))))
HOST = os.getenv("HOST", config['server'].get('host', "0.0.0.0"))
AUTO_LOAD = config['server'].get('auto_load_enabled', False)

# Initialize model manager
model_manager = ModelManager()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def auto_load_models():
    """Auto-load models marked for auto-loading in config."""
    if not AUTO_LOAD:
        return
        
    logger.info("Auto-loading enabled models...")
    for model_id, model_config in AVAILABLE_MODELS.items():
        if hasattr(model_config, 'auto_load') and model_config.auto_load:
            logger.info(f"Auto-loading model: {model_id}")
            success = model_manager.load_model(model_id)
            if success:
                logger.info(f"Successfully auto-loaded {model_id}")
            else:
                logger.error(f"Failed to auto-load {model_id}")

@app.on_event("startup")
async def startup_event():
    """Initialize server and auto-load models."""
    await auto_load_models()

@app.middleware("http")
async def cleanup_after_request(request: Request, call_next):
    response = await call_next(request)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    return response

@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    if "stream" in request.query_params and request.query_params["stream"] == "true":
        response.headers["Content-Type"] = "text/event-stream"
    else:
        response.headers["Content-Type"] = "application/json"
    return response

@app.get("/v1/models")
async def list_models():
    """List all available models."""
    models_list = []
    for model_name, config in AVAILABLE_MODELS.items():
        models_list.append({
            "id": model_name,
            "name": config.name,
            "description": config.description,
            "context_length": config.context_length,
            "loaded": model_name in model_manager.list_loaded_models()
        })
    return {"object": "list", "data": models_list}

@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    """Get details about a specific model."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    config = AVAILABLE_MODELS[model_id]
    return {
        "id": model_id,
        "name": config.name,
        "description": config.description,
        "context_length": config.context_length,
        "loaded": model_id in model_manager.list_loaded_models(),
        "gpu_allocation": list(config.device_map.values()) if config.device_map else None
    }

@app.post("/v1/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a specific model."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    success = model_manager.load_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")
    
    return {"status": "success", "message": f"Model {model_id} loaded successfully"}

@app.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a specific model."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    success = model_manager.unload_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to unload model {model_id}")
    
    return {"status": "success", "message": f"Model {model_id} unloaded successfully"}

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        stream = request.get("stream", False)
        messages = request.get("messages", [])
        model_id = request.get("model", "deepseek-r1-distil-32b")
        temperature = float(request.get("temperature", 0.3))
        top_p = float(request.get("top_p", 0.85))
        max_tokens = request.get("max_tokens", None)
        stop = request.get("stop", None)

        if model_id not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Load model if not loaded
        if model_id not in model_manager.list_loaded_models():
            success = model_manager.load_model(model_id)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")

        if stream:
            return StreamingResponse(
                stream_generate(
                    model_id, 
                    messages, 
                    temperature, 
                    top_p, 
                    max_tokens, 
                    stop
                ),
                media_type="text/event-stream"
            )

        response_text = model_manager.generate_response(
            model_id,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens if max_tokens else 2048,
            stop=stop
        )

        if response_text is None:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        completion_id = f"chatcmpl-{secrets.token_hex(12)}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(model_manager.get_tokenizer(model_id).encode(str(messages))),
                "completion_tokens": len(model_manager.get_tokenizer(model_id).encode(response_text)),
                "total_tokens": len(model_manager.get_tokenizer(model_id).encode(str(messages))) + len(model_manager.get_tokenizer(model_id).encode(response_text))
            }
        }

    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}}
        )

async def stream_generate(model_id: str, messages: List[dict], temperature: float, top_p: float, max_tokens: Optional[int], stop: Optional[List[str]]):
    try:
        chunk_id = f"chatcmpl-{secrets.token_hex(12)}"
        model = model_manager.get_model(model_id)
        tokenizer = model_manager.get_tokenizer(model_id)
        config = model_manager.get_config(model_id)

        if not all([model, tokenizer, config]):
            raise Exception(f"Model {model_id} not properly loaded")

        # Format and tokenize prompt
        prompt = model_manager._format_chat_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Setup streamer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            timeout=None
        )

        # Update generation config
        generation_config = config.to_generation_config()
        generation_config.update({
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens if max_tokens else config.max_new_tokens,
            "streamer": streamer
        })

        # Start generation in a separate thread
        thread = Thread(target=lambda: model.generate(**inputs, **generation_config))
        thread.start()

        buffer = ""
        thinking_closed = False
        in_thinking = False
        BUFFER_SIZE = 32

        # Use iterator instead of async for
        for text in streamer:
            # Handle think tags based on show_thinking parameter
            if "<think>" in text:
                in_thinking = True
            elif "</think>" in text:
                in_thinking = False
                thinking_closed = True

            # Skip content if we're inside thinking tags and show_thinking is False
            if not in_thinking and ("<think>" in text or "</think>" in text):
                continue

            # Check for stop tokens
            if any(stop_token in text for stop_token in config.stop_tokens):
                if not thinking_closed:
                    chunk = {
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': model_id,
                        'choices': [{'index': 0, 'delta': {'content': '</think>'}}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    thinking_closed = True
                break

            buffer += text
            if len(buffer) >= BUFFER_SIZE:
                chunk = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': model_id,
                    'choices': [{'index': 0, 'delta': {'content': buffer}}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                buffer = ""

            # Add a small delay to avoid overwhelming the connection
            await asyncio.sleep(0.01)

        # Send remaining buffer
        if buffer:
            chunk = {
                'id': chunk_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': model_id,
                'choices': [{'index': 0, 'delta': {'content': buffer}}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final completion chunk
        final_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model_id,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        error_chunk = {'error': {'message': str(e), 'type': 'internal_error'}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

@app.get("/health")
async def health_check():
    """Check server health and model status."""
    memory_info = {
        "gpu_memory": {
            i: {
                "total": f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f}GB",
                "used": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f}GB",
                "free": f"{(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024**3):.2f}GB"
            }
            for i in range(torch.cuda.device_count())
        },
        "system_memory": {
            "total": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
            "used": f"{psutil.virtual_memory().used / (1024**3):.2f}GB",
            "free": f"{psutil.virtual_memory().free / (1024**3):.2f}GB"
        }
    }
    
    return {
        "status": "ok",
        "loaded_models": model_manager.list_loaded_models(),
        "available_models": [m["id"] for m in (await list_models())["data"]],
        "memory_info": memory_info
    }

if __name__ == "__main__":
    print(f"\nStarting server on {HOST}:{PORT}")
    # Get available models synchronously
    models = [
        {
            "id": model_id,
            "name": config.name,
            "description": config.description
        }
        for model_id, config in AVAILABLE_MODELS.items()
    ]
    print("Available models:", [m["id"] for m in models])
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        loop="asyncio",
        workers=1
    )
