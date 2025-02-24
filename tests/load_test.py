import os
import sys
import time
import json
import asyncio
import aiohttp
import logging
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000/v1"
MODELS = [
  "deepseek-r1-distil-32b",
  "deepseek-r1-distil-14b",
  "deepseek-r1-distil-8b",
  "qwen-vl",
  "idefics-80b",
  "fuyu-8b"
]

TEST_PROMPTS = {
  "text": [
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main challenges in sustainable energy?",
    "Describe the process of photosynthesis step by step.",
    "Compare and contrast different programming paradigms."
  ],
  "vision": [
    {
      "type": "text",
      "text": "What's in this image?"
    },
    {
      "type": "image",
      "image_url": "https://raw.githubusercontent.com/microsoft/JARVIS/main/assets/demo1.jpg"
    }
  ]
}

async def make_request(
  session: aiohttp.ClientSession,
  model: str,
  prompt: List[Dict],
  request_id: int
) -> Dict:
  """Make a single request to the API."""
  try:
    start_time = time.time()
    async with session.post(
      f"{BASE_URL}/chat/completions",
      json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
      }
    ) as response:
      duration = time.time() - start_time
      result = await response.json()
      
      if response.status == 200:
        logger.info(f"Request {request_id} to {model} succeeded in {duration:.2f}s")
        return {
          "success": True,
          "model": model,
          "duration": duration,
          "status": response.status
        }
      else:
        logger.warning(
          f"Request {request_id} to {model} failed with status {response.status}: {result}"
        )
        return {
          "success": False,
          "model": model,
          "duration": duration,
          "status": response.status,
          "error": result
        }
  except Exception as e:
    logger.error(f"Error in request {request_id} to {model}: {str(e)}")
    return {
      "success": False,
      "model": model,
      "error": str(e)
    }

async def run_load_test(
  concurrent_requests: int,
  requests_per_model: int,
  delay: float = 0.1
):
  """Run load test with specified parameters."""
  async with aiohttp.ClientSession() as session:
    request_id = 0
    results = []
    
    for _ in range(requests_per_model):
      tasks = []
      for model in MODELS:
        prompt = (
          TEST_PROMPTS["vision"] 
          if model in ["qwen-vl", "idefics-80b", "fuyu-8b"]
          else TEST_PROMPTS["text"][request_id % len(TEST_PROMPTS["text"])]
        )
        
        tasks.append(
          make_request(session, model, prompt, request_id)
        )
        request_id += 1
        
        if len(tasks) >= concurrent_requests:
          batch_results = await asyncio.gather(*tasks)
          results.extend(batch_results)
          tasks = []
          await asyncio.sleep(delay)
    
    if tasks:  # Handle remaining tasks
      batch_results = await asyncio.gather(*tasks)
      results.extend(batch_results)
    
    return results

def analyze_results(results: List[Dict]):
  """Analyze and print test results."""
  total_requests = len(results)
  successful_requests = sum(1 for r in results if r["success"])
  failed_requests = total_requests - successful_requests
  
  model_stats = {}
  for result in results:
    model = result["model"]
    if model not in model_stats:
      model_stats[model] = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "durations": []
      }
    
    stats = model_stats[model]
    stats["total"] += 1
    if result["success"]:
      stats["success"] += 1
      stats["durations"].append(result["duration"])
    else:
      stats["failed"] += 1
  
  print("\nLoad Test Results:")
  print(f"Total Requests: {total_requests}")
  print(f"Successful: {successful_requests}")
  print(f"Failed: {failed_requests}")
  print("\nPer-Model Statistics:")
  
  for model, stats in model_stats.items():
    success_rate = (stats["success"] / stats["total"]) * 100
    avg_duration = (
      sum(stats["durations"]) / len(stats["durations"])
      if stats["durations"]
      else 0
    )
    print(f"\n{model}:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Average Duration: {avg_duration:.2f}s")
    print(f"  Failed Requests: {stats['failed']}")

def main():
  parser = argparse.ArgumentParser(description="API Load Testing Tool")
  parser.add_argument(
    "--concurrent",
    type=int,
    default=3,
    help="Number of concurrent requests"
  )
  parser.add_argument(
    "--requests",
    type=int,
    default=5,
    help="Number of requests per model"
  )
  parser.add_argument(
    "--delay",
    type=float,
    default=0.1,
    help="Delay between request batches in seconds"
  )
  
  args = parser.parse_args()
  
  logger.info(
    f"Starting load test with {args.concurrent} concurrent requests, "
    f"{args.requests} requests per model"
  )
  
  results = asyncio.run(
    run_load_test(
      args.concurrent,
      args.requests,
      args.delay
    )
  )
  
  analyze_results(results)

if __name__ == "__main__":
  main() 