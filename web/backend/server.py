
from re import template
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from tmh.text.text_generation import generate_text
import uvicorn

class GenerateRequest(BaseModel):
    text: str
    model: Optional[str] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8000/"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/generate")
async def generate_response(generateRequest: GenerateRequest):
    
    print("inside with reequest")
    print(generateRequest)

    if generateRequest.model:
        response = generate_text(
            model=generateRequest.model,
            prompt=generateRequest.text,
            max_length=generateRequest.max_length,
            temperature=generateRequest.temperature,
        )
    else:
        response = generate_text(model='birgermoell/swedish-gpt', prompt=generateRequest.text, max_length=250, temperature=0.9)

    print("response is", response)

    #print("the response is", response)
    return {
        "text": response }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)