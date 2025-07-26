from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
from typing import List
import torch
import time
import os

app = FastAPI()
security = HTTPBearer()

# API Key 환경변수에서 가져오기 (기본값 제공)
API_KEY = os.getenv("EMBEDDING_API_KEY", "default-embedding-key-2024")
if API_KEY == "default-embedding-key-2024":
    print("⚠️  기본 API 키를 사용중입니다. 보안을 위해 EMBEDDING_API_KEY 환경변수를 설정하세요.")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

MODEL_PATH = "./local_models/labse/models--sentence-transformers--LaBSE/snapshots/836121a0533e5664b21c7aacc5d22951f2b8b25b"  # ✅ 로컬 모델 경로

print("🟡 모델 로딩 시작...")

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"✅ Tokenizer 로딩 완료 ({time.time() - start:.2f}s)")

start = time.time()
model = AutoModel.from_pretrained(MODEL_PATH)
print(f"✅ Model 로딩 완료 ({time.time() - start:.2f}s)")

class BulkEmbedRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.get("/embed")
async def embed(text: str = Query(..., min_length=1), api_key: str = Depends(verify_api_key)):
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

@app.post("/embed/bulk")
async def embed_bulk(request: BulkEmbedRequest, api_key: str = Depends(verify_api_key)):
    try:
        print(f"📩 벌크 요청 도착: {len(request.texts)}개 텍스트")
        
        # 배치로 토크나이징 (효율적)
        inputs = tokenizer(request.texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 각 텍스트별로 평균 풀링
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        # 각 임베딩을 리스트로 변환
        embeddings = [embedding.tolist() for embedding in pooled]
        
        print(f"✅ 벌크 임베딩 완료: {len(embeddings)}개")
        return {"embeddings": embeddings}
    except Exception as e:
        print(f"❌ 벌크 처리 오류: {e}")
        return {"error": str(e)}
