"""
FastAPI inference server for model deployment.
Deploy the trained model as a REST API.
"""

import logging
import sys
import os
import subprocess
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize inference engine on startup."""
    global inference_engine
    logger.info("Initializing inference engine...")
    from inference.generation import InferenceEngine
    tokenizer_name = os.getenv("TOKENIZER_NAME", "google/gemma-2b")
    model_path = os.getenv("MODEL_PATH", "./outputs/best_model.pt")
    inference_engine = InferenceEngine(
        model_path=model_path,
        device="cpu",
        tokenizer_name=tokenizer_name,
    )
    logger.info("✓ Inference engine initialized")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="NPT Model Inference Server",
    description="REST API for NPT Language Model inference",
    version="0.1.0",
    lifespan=lifespan,
)

# Request/Response models
class GenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50


class GenerationResponse(BaseModel):
    """Text generation response."""
    prompt: str
    generated_text: str
    tokens_generated: int


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    texts: List[str]


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embeddings: List[List[float]]
    num_texts: int
    embedding_dim: int


class StageRunRequest(BaseModel):
    stage: str


class TrainRunRequest(BaseModel):
    tokenizer_name: str = "Qwen/Qwen2.5-7B"
    num_epochs: int = 1
    dataset_size: int = 256
    seq_length: int = 64
    batch_size: int = 2


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "NPT",
        "version": "0.1.0"
    }


@app.get("/", response_class=HTMLResponse)
async def web_console():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>NPT Web Test Console</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color:#111; }
    .wrap { max-width: 980px; margin: 0 auto; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 14px; }
    h2 { margin: 0 0 8px 0; }
    textarea, input, select { width: 100%; padding: 8px; margin: 6px 0; border:1px solid #ccc; border-radius: 8px; }
    button { padding: 9px 14px; border: none; border-radius: 8px; background: #0b62f2; color: #fff; cursor: pointer; }
    pre { white-space: pre-wrap; word-wrap: break-word; background:#0f172a; color:#e2e8f0; padding:10px; border-radius:8px; max-height:300px; overflow:auto; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NPT Web Test Console</h1>

    <div class="card">
      <h2>Health</h2>
      <button onclick="getHealth()">Check Health</button>
      <pre id="healthOut"></pre>
    </div>

    <div class="card">
      <h2>Generate</h2>
      <textarea id="prompt" rows="3">বাংলাদেশের রাজধানী</textarea>
      <input id="maxLength" type="number" value="24" />
      <input id="temperature" type="number" step="0.1" value="0.2" />
      <input id="topP" type="number" step="0.05" value="0.8" />
      <input id="topK" type="number" value="20" />
      <button onclick="runGenerate()">Run Generate</button>
      <pre id="genOut"></pre>
    </div>

    <div class="card">
      <h2>Embed</h2>
      <textarea id="embedTexts" rows="3">আমি বাংলা লিখি\\nThis is English text</textarea>
      <button onclick="runEmbed()">Run Embed</button>
      <pre id="embedOut"></pre>
    </div>

    <div class="card">
      <h2>Pipeline Stage</h2>
      <select id="stage">
        <option>collection</option>
        <option>cleaning</option>
        <option>filtering</option>
        <option>deduplication</option>
        <option>tokenization</option>
        <option>evaluation</option>
      </select>
      <button onclick="runStage()">Run Stage</button>
      <pre id="stageOut"></pre>
    </div>

    <div class="card">
      <h2>Quick Train</h2>
      <input id="tokenizerName" value="Qwen/Qwen2.5-7B" />
      <input id="numEpochs" type="number" value="1" />
      <input id="datasetSize" type="number" value="256" />
      <input id="seqLength" type="number" value="64" />
      <input id="batchSize" type="number" value="2" />
      <button onclick="runTrain()">Run Training</button>
      <pre id="trainOut"></pre>
    </div>
  </div>
<script>
async function show(outId, data) {
  document.getElementById(outId).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}
async function getHealth() {
  const r = await fetch('/health');
  show('healthOut', await r.json());
}
async function runGenerate() {
  const payload = {
    prompt: document.getElementById('prompt').value,
    max_length: Number(document.getElementById('maxLength').value),
    temperature: Number(document.getElementById('temperature').value),
    top_p: Number(document.getElementById('topP').value),
    top_k: Number(document.getElementById('topK').value)
  };
  const r = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  show('genOut', await r.json());
}
async function runEmbed() {
  const texts = document.getElementById('embedTexts').value.split('\\n').map(x=>x.trim()).filter(Boolean);
  const r = await fetch('/embed', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({texts})});
  show('embedOut', await r.json());
}
async function runStage() {
  const stage = document.getElementById('stage').value;
  const r = await fetch('/admin/run-stage', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({stage})});
  show('stageOut', await r.json());
}
async function runTrain() {
  const payload = {
    tokenizer_name: document.getElementById('tokenizerName').value,
    num_epochs: Number(document.getElementById('numEpochs').value),
    dataset_size: Number(document.getElementById('datasetSize').value),
    seq_length: Number(document.getElementById('seqLength').value),
    batch_size: Number(document.getElementById('batchSize').value),
  };
  const r = await fetch('/admin/run-train', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  show('trainOut', await r.json());
}
</script>
</body>
</html>
"""


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_text = inference_engine.generate(
            [request.prompt],
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )[0]
        
        # Ensure safe printable output for JSON/terminal clients.
        generated_text = generated_text.encode("utf-8", "replace").decode("utf-8")
        generated_text = "".join(ch for ch in generated_text if ch.isprintable() or ch in "\n\t ")

        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            tokens_generated=len(generated_text.split())
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """Get embeddings for texts."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        embeddings = inference_engine.embed(request.texts)
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist() if torch.is_tensor(embeddings) else embeddings,
            num_texts=len(request.texts),
            embedding_dim=embeddings.shape[-1] if hasattr(embeddings, 'shape') else len(embeddings[0])
        )
    
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/run-stage")
async def run_stage(request: StageRunRequest):
    allowed = {"collection", "cleaning", "filtering", "deduplication", "tokenization", "evaluation"}
    if request.stage not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid stage. Allowed: {sorted(allowed)}")
    cmd = [
        "python3",
        "scripts/run_pipeline.py",
        "--stage",
        request.stage,
        "--config",
        "pipeline_config.json",
        "--output-dir",
        "./outputs",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "stage": request.stage,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


@app.post("/admin/run-train")
async def run_train(request: TrainRunRequest):
    cmd = [
        "python3",
        "training/pretraining/train.py",
        "--num-epochs",
        str(request.num_epochs),
        "--dataset-size",
        str(request.dataset_size),
        "--seq-length",
        str(request.seq_length),
        "--batch-size",
        str(request.batch_size),
        "--tokenizer-name",
        request.tokenizer_name,
        "--output-dir",
        "./outputs",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


@app.get("/model/info")
async def model_info():
    """Get model information."""
    return {
        "name": "NPT Model",
        "version": "0.1.0",
        "type": "Language Model",
        "supported_tasks": ["generation", "embedding"],
        "max_sequence_length": 2048,
        "vocab_size": 50000
    }


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
