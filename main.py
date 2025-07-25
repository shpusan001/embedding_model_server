from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModel
import torch
import time

app = FastAPI()

MODEL_PATH = "./local_models/labse/models--sentence-transformers--LaBSE/snapshots/836121a0533e5664b21c7aacc5d22951f2b8b25b"  # ✅ 로컬 모델 경로

print("🟡 모델 로딩 시작...")

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"✅ Tokenizer 로딩 완료 ({time.time() - start:.2f}s)")

start = time.time()
model = AutoModel.from_pretrained(MODEL_PATH)
print(f"✅ Model 로딩 완료 ({time.time() - start:.2f}s)")

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.get("/embed")
async def embed(text: str = Query(..., min_length=1)):
    try:
        print(f"📩 요청 도착: {text}")
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1)
        print("✅ 임베딩 완료")
        return {"embedding": pooled[0].tolist()}
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return {"error": str(e)}
