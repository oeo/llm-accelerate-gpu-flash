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
import base64
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import TextIteratorStreamer
from threading import Thread, Lock
from typing import List, Optional, Dict, Union
from pydantic import BaseModel
from lib.models import ModelManager, AVAILABLE_MODELS, ModelConfig
from contextlib import asynccontextmanager

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
VRAM_BUFFER = float(config['server'].get('vram_buffer_gb', 1.0))  # Buffer in GB
MAX_WAITING_REQUESTS = int(config['server'].get('max_waiting_requests', 10))

# Initialize model manager
model_manager = ModelManager()

# Request management
request_queue = asyncio.Queue(maxsize=MAX_WAITING_REQUESTS)
gpu_locks = {i: asyncio.Lock() for i in range(torch.cuda.device_count())}
model_locks = {}  # Will be populated with model-specific locks

class VRAMMonitor:
    def __init__(self, buffer_gb: float = 1.0):
        self.buffer_gb = buffer_gb * (1024**3)  # Convert to bytes
        self.device_count = torch.cuda.device_count()
        self.thermal_config = config['gpu']['overload_protection']['thermal_management']
        self.max_temp = config['gpu']['overload_protection']['max_temperature']
        self.throttle_temp = config['gpu']['overload_protection']['thermal_throttle']
        self.critical_temp = config['gpu']['overload_protection']['critical_temp']
        self.concurrent_requests = {i: 0 for i in range(self.device_count)}
    
    def get_gpu_temperature(self, device: int) -> float:
        """Get current temperature of GPU."""
        return float(torch.cuda.get_device_properties(device).temperature)
    
    def get_available_vram(self, device: int) -> float:
        """Get available VRAM in bytes for a device."""
        total = torch.cuda.get_device_properties(device).total_memory
        used = torch.cuda.memory_allocated(device)
        return total - used - self.buffer_gb
    
    def get_thermal_limit(self, device: int) -> int:
        """Get maximum concurrent requests based on temperature."""
        temp = self.get_gpu_temperature(device)
        
        # If temperature is above critical, don't allow new requests
        if temp >= self.critical_temp:
            return 0
            
        # Find appropriate throttle step
        for step in self.thermal_config['throttle_steps']:
            if temp >= step['temp']:
                return step['max_concurrent']
        
        # If temperature is below all throttle steps, no limit
        return float('inf')
    
    def can_accommodate(self, model_id: str, config: ModelConfig) -> Optional[List[int]]:
        """Check if model can be accommodated and return suitable devices."""
        if config.device_map:  # Multi-GPU model
            required_devices = set(config.device_map.values())
            
            # Check temperature limits first
            for device in required_devices:
                if self.get_gpu_temperature(device) >= self.critical_temp:
                    logger.warning(f"GPU {device} temperature ({self.get_gpu_temperature(device)}°C) above critical threshold")
                    return None
                
                # Check if adding this request would exceed thermal limits
                thermal_limit = self.get_thermal_limit(device)
                if self.concurrent_requests[device] >= thermal_limit:
                    logger.warning(f"GPU {device} at thermal limit ({self.get_gpu_temperature(device)}°C)")
                    return None
            
            # Then check VRAM
            for device in required_devices:
                if self.get_available_vram(device) < (15 * (1024**3)):  # Assume 15GB per GPU needed
                    return None
            
            return list(required_devices)
        else:  # Single GPU model
            for device in range(self.device_count):
                # Check temperature first
                if self.get_gpu_temperature(device) >= self.critical_temp:
                    continue
                
                # Check thermal limits
                thermal_limit = self.get_thermal_limit(device)
                if self.concurrent_requests[device] >= thermal_limit:
                    continue
                
                # Check VRAM
                if self.get_available_vram(device) >= (15 * (1024**3)):
                    return [device]
        return None
    
    async def acquire_gpu(self, devices: List[int]):
        """Increment concurrent request count for devices."""
        for device in devices:
            self.concurrent_requests[device] += 1
    
    async def release_gpu(self, devices: List[int]):
        """Decrement concurrent request count for devices."""
        for device in devices:
            self.concurrent_requests[device] = max(0, self.concurrent_requests[device] - 1)

vram_monitor = VRAMMonitor(VRAM_BUFFER)

async def wait_for_vram(model_id: str) -> bool:
    """Wait for sufficient VRAM to be available."""
    config = AVAILABLE_MODELS.get(model_id)
    if not config:
        return False
    
    try:
        # Try to get VRAM immediately
        devices = vram_monitor.can_accommodate(model_id, config)
        if devices:
            await vram_monitor.acquire_gpu(devices)
            return True
        
        # If queue is full, reject request
        try:
            await asyncio.wait_for(request_queue.put(model_id), timeout=0.1)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Server is at capacity. Please try again later."
            )
        
        # Wait for VRAM with timeout
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            devices = vram_monitor.can_accommodate(model_id, config)
            if devices:
                await request_queue.get()  # Remove our request from queue
                await vram_monitor.acquire_gpu(devices)
                return True
            await asyncio.sleep(0.5)
        
        # Timeout reached
        await request_queue.get()  # Remove our request from queue
        raise HTTPException(
            status_code=503,
            detail="Timed out waiting for GPU resources"
        )
    
    except Exception as e:
        logger.error(f"Error waiting for VRAM: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    if AUTO_LOAD:
        logger.info("Auto-loading enabled models...")
        for model_id, model_config in AVAILABLE_MODELS.items():
            if hasattr(model_config, 'auto_load') and model_config.auto_load:
                logger.info(f"Auto-loading model: {model_id}")
                success = model_manager.load_model(model_id)
                if success:
                    logger.info(f"Successfully auto-loaded {model_id}")
                else:
                    logger.error(f"Failed to auto-load {model_id}")
    
    yield
    
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def chat_completions(request: dict, background_tasks: BackgroundTasks):
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

        # Get model config
        model_config = AVAILABLE_MODELS[model_id]
        
        # Wait for VRAM and acquire GPU
        await wait_for_vram(model_id)
        
        try:
            # Process messages to handle image content
            processed_messages = []
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    processed_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image":
                                image_data = item.get("image_url", {})
                                if isinstance(image_data, dict) and image_data.get("url", "").startswith("data:image/"):
                                    b64_data = image_data["url"].split(",")[1]
                                    image = Image.open(BytesIO(base64.b64decode(b64_data)))
                                    processed_content.append({
                                        "type": "image",
                                        "image": image
                                    })
                                elif isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
                                    response = requests.get(image_data)
                                    response.raise_for_status()
                                    image = Image.open(BytesIO(response.content))
                                    processed_content.append({
                                        "type": "image",
                                        "image": image
                                    })
                            else:
                                processed_content.append(item)
                    processed_messages.append({
                        "role": message["role"],
                        "content": processed_content
                    })
                else:
                    processed_messages.append(message)

            # Load model if not loaded
            if model_id not in model_manager.list_loaded_models():
                success = model_manager.load_model(model_id)
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")

            # Get or create model lock
            if model_id not in model_locks:
                model_locks[model_id] = asyncio.Lock()
            
            async with model_locks[model_id]:  # Ensure sequential access to each model
                if stream:
                    return StreamingResponse(
                        stream_generate(
                            model_id, 
                            processed_messages, 
                            temperature, 
                            top_p, 
                            max_tokens, 
                            stop
                        ),
                        media_type="text/event-stream"
                    )

                response_text = model_manager.generate_response(
                    model_id,
                    processed_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens if max_tokens else 2048,
                    stop=stop
                )

            if response_text is None:
                raise HTTPException(status_code=500, detail="Failed to generate response")

            completion_id = f"chatcmpl-{secrets.token_hex(12)}"
            response = {
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

            # Release GPU after processing
            if model_config.device_map:
                await vram_monitor.release_gpu(list(set(model_config.device_map.values())))
            else:
                await vram_monitor.release_gpu([model_config.device])

            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"Failed to fetch image: {str(e)}", "type": "invalid_request_error"}}
            )
        except Exception as e:
            logger.error(f"Error in chat_completions: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            # Make sure to release GPU on error
            if model_config.device_map:
                await vram_monitor.release_gpu(list(set(model_config.device_map.values())))
            else:
                await vram_monitor.release_gpu([model_config.device])
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(e), "type": "internal_error"}}
            )

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
        BUFFER_SIZE = 32

        # Use iterator instead of async for
        for text in streamer:
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
