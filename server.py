import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ['TORCH_CUDA_ARCH_PATH'] = './cuda_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "True"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["ACCELERATE_USE_CPU_OFFLOAD"] = "True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import gc
import json
import time
import psutil
import uvicorn
import secrets
import logging
import asyncio
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from accelerate import Accelerator
from accelerate.utils import get_balanced_memory
from typing import List, Optional
from pydantic import BaseModel
from threading import Thread
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up torch configurations
torch.set_num_threads(40)
torch.set_num_interop_threads(40)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def optimize_model_memory():
    torch.cuda.empty_cache()
    gc.collect()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

def manage_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class ModelPermission(BaseModel):
    id: str
    object: str = "model_permission"
    created: int = int(time.time())
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "deepseek-ai"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []

###################
# CONFIGURATIONS  #
###################

# Model Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MODEL_REVISION = "main"

STOP_TOKENS = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>"
]

AVAILABLE_MODELS = {
    "deepseek-r1-distil-32b": {
        "id": "deepseek-r1-distil-32b",
        "name": "DeepSeek-R1-Distill-Qwen-32B",
        "description": "DeepSeek R1 32B Distilled Model",
        "context_length": 4096,
        "created": int(time.time())
    }
}

# Server Configuration
PORT = 8000
HOST = "0.0.0.0"

# GPU Configuration
NUM_GPUS = torch.cuda.device_count()
GPU_MEMORY_PER_DEVICE = {i: torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(NUM_GPUS)}

# Generation Configuration
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "max_new_tokens": 2048,
    "use_cache": True,
    "typical_p": 0.95,
    "num_beams": 1,
    "early_stopping": True,
    "return_dict_in_generate": True,
    "output_scores": False,
    "pad_token_id": 0,  # Will be updated after tokenizer initialization
    "eos_token_id": 0,  # Will be updated after tokenizer initialization
}

# Quantization Configuration
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True
)

# Model loading configuration
model_loading_kwargs = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
    "offload_folder": "offload",
    "offload_state_dict": True,
    "use_flash_attention_2": True,
    "_fast_init": True,
    "use_cache": True
}

def create_balanced_device_map():
    num_layers = 64  # DeepSeek-32B layers
    layers_per_gpu = num_layers // NUM_GPUS
    device_map = {}

    device_map.update({
        'model.embed_tokens': 0,
        'model.rotary_emb': 0,
    })

    for i in range(num_layers):
        gpu_id = i // layers_per_gpu
        device_map[f'model.layers.{i}'] = gpu_id

    device_map.update({
        'model.norm': NUM_GPUS - 1,
        'lm_head': NUM_GPUS - 1
    })

    return device_map

# Create the device map
device_map = create_balanced_device_map()

# Memory configuration
MAX_MEMORY = {i: f"{int(13.5)}GB" for i in range(NUM_GPUS)}
MAX_MEMORY["cpu"] = "250GB"

###################
# INITIALIZATION  #
###################

app = FastAPI()

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
    optimize_model_memory()
    return response

@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    if "stream" in request.query_params and request.query_params["stream"] == "true":
        response.headers["Content-Type"] = "text/event-stream"
    else:
        response.headers["Content-Type"] = "application/json"
    return response

print("Initializing model...")
start_time = time.time()

# Update model_loading_kwargs with the device_map and max_memory
model_loading_kwargs.update({
    "device_map": device_map,
    "max_memory": MAX_MEMORY,
    "quantization_config": QUANTIZATION_CONFIG,
    "revision": MODEL_REVISION,
    "trust_remote_code": True,
})

# Load the model with consolidated parameters
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_loading_kwargs
)

# Add model optimizations
model.config.use_cache = True
model.config.pretraining_tp = 1

# Enable flash attention if available
if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "flash_attention_2"

# Enable torch.compile
if hasattr(torch, 'compile'):
    print("Enabling torch compile mode...")
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=True
    )

print(f"Model loading took: {time.time() - start_time:.2f} seconds")

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    revision=MODEL_REVISION,
    additional_special_tokens=STOP_TOKENS
)

# Update generation config with tokenizer values
GENERATION_CONFIG["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
GENERATION_CONFIG["eos_token_id"] = tokenizer.eos_token_id

# Model warmup
print("\nPre-warming model...")
warmup_prompt = "Hello, how are you?"
inputs = tokenizer(warmup_prompt, return_tensors="pt", padding=True)
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

start_time = time.time()
with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
print(f"First generation warmup took: {time.time() - start_time:.2f} seconds")

###################
# UTILITY FUNCTIONS #
###################

def print_gpu_memory(message=""):
    print(f"\nGPU Memory Status {message}:")
    for i in range(NUM_GPUS):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def get_gpu_memory_usage():
    memory_usage = {}
    for i in range(NUM_GPUS):
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        memory_usage[f"gpu_{i}"] = {
            "allocated": f"{memory_allocated:.2f}GB",
            "reserved": f"{memory_reserved:.2f}GB"
        }
    return memory_usage

@lru_cache(maxsize=1024)
def cache_tokenize(text):
    return tokenizer(text, return_tensors="pt", padding=True)

def log_performance(start_time, num_tokens):
    elapsed = time.time() - start_time
    tokens_per_second = num_tokens / elapsed
    logger.info(f"Generated {num_tokens} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")

def format_chat_prompt(messages):
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"<｜User｜>{content}\n"
        elif role == "assistant":
            prompt += f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>\n"

    prompt += "<｜Assistant｜>"
    return prompt

def generate_response(messages, max_tokens=None, temperature=1.0, top_p=1.0,
                     stop=None, presence_penalty=0.0, frequency_penalty=0.0):
    try:
        generation_start = time.time()
        prompt = format_chat_prompt(messages)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")

        generation_config = GENERATION_CONFIG.copy()
        generation_config.update({
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens if max_tokens else 2048,
        })

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        response_text = response_text.strip()

        for stop_token in STOP_TOKENS:
            if stop_token in response_text:
                response_text = response_text[:response_text.index(stop_token)]

        print(f"Total generation took: {time.time() - generation_start:.2f} seconds")

        # Cleanup after generation
        manage_memory()

        return response_text.strip()

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        manage_memory()
        raise e

async def async_generator(streamer):
    for text in streamer:
        yield text

async def stream_generate(messages, temperature, top_p, max_tokens, stop):
    try:
        chunk_id = f"chatcmpl-{secrets.token_hex(12)}"
        prompt = format_chat_prompt(messages)

        # Send initial chunk
        initial_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': 'deepseek-r1-distil-32b',
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None
        )

        generation_config = GENERATION_CONFIG.copy()
        generation_config.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens if max_tokens else 2048,
            "streamer": streamer,
        })

        thread = Thread(target=lambda: model.generate(**generation_config))
        thread.start()

        buffer = ""
        async for text in async_generator(streamer):
            if any(stop_token in text for stop_token in STOP_TOKENS):
                break

            buffer += text
            if len(buffer) >= 4:
                chunk = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'deepseek-r1-distil-32b',
                    'choices': [{'index': 0, 'delta': {'content': buffer}}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                buffer = ""

        if buffer:
            chunk = {
                'id': chunk_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'deepseek-r1-distil-32b',
                'choices': [{'index': 0, 'delta': {'content': buffer}}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': 'deepseek-r1-distil-32b',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        manage_memory()

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        error_chunk = {'error': {'message': str(e), 'type': 'internal_error'}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        manage_memory()

###################
# API ENDPOINTS   #
###################

@app.get("/v1/models")
async def list_models():
    models_list = []
    for model_id, model_info in AVAILABLE_MODELS.items():
        model_card = ModelCard(
            id=model_id,
            created=model_info["created"],
            permission=[ModelPermission(id=f"modelperm-{model_id}", created=model_info["created"])]
        )
        models_list.append(model_card.model_dump())
    return {"object": "list", "data": models_list}

@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    model_info = AVAILABLE_MODELS[model_id]
    model_card = ModelCard(
        id=model_id,
        created=model_info["created"],
        permission=[ModelPermission(id=f"modelperm-{model_id}", created=model_info["created"])]
    )
    return model_card.model_dump()

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        stream = request.get("stream", False)
        messages = request.get("messages", [])
        model = request.get("model", "deepseek-r1-distil-32b")
        temperature = float(request.get("temperature", 0.7))
        top_p = float(request.get("top_p", 0.95))
        max_tokens = request.get("max_tokens", None)
        stop = request.get("stop", None)

        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")

        if stream:
            return StreamingResponse(
                stream_generate(messages, temperature, top_p, max_tokens, stop),
                media_type="text/event-stream"
            )

        response_text = generate_response(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

        completion_id = f"chatcmpl-{secrets.token_hex(12)}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(str(messages))),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": len(tokenizer.encode(str(messages))) + len(tokenizer.encode(response_text))
            }
        }

    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}")
        manage_memory()
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}}
        )

@app.post("/v1/completions")
async def completions(request: dict):
    try:
        prompt = request.get("prompt", "")
        messages = [{"role": "user", "content": prompt}]

        response = await chat_completions({
            "messages": messages,
            "model": request.get("model", "deepseek-r1-distil-32b"),
            "temperature": request.get("temperature", 0.7),
            "top_p": request.get("top_p", 0.95),
            "max_tokens": request.get("max_tokens", None),
            "stop": request.get("stop", None),
            "stream": request.get("stream", False)
        })

        if isinstance(response, StreamingResponse):
            return response

        completion_id = f"cmpl-{secrets.token_hex(12)}"
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.get("model", "deepseek-r1-distil-32b"),
            "choices": [{
                "text": response["choices"][0]["message"]["content"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": response["usage"]
        }

    except Exception as e:
        logger.error(f"Error in completions: {str(e)}")
        manage_memory()
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}}
        )

@app.get("/health")
async def health_check():
    memory_info = {
        "gpu_memory": get_gpu_memory_usage(),
        "system_memory_used": f"{psutil.virtual_memory().percent}%",
        "cpu_percent": psutil.cpu_percent(interval=1),
    }
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device_map": model.hf_device_map,
        "memory_info": memory_info,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": None
            }
        }
    )

###################
# MAIN EXECUTION  #
###################

if __name__ == "__main__":
    print(f"\nStarting server on {HOST}:{PORT}")
    print("Model is ready for inference")
    print_gpu_memory("at server start")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        loop="asyncio",
        workers=1
    )

