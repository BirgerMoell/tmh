
from re import template
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tmh.text.text_generation import generate_text
import uvicorn





class TextRequest(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "KTH är ett universitet i Stockholm",
            }
        }


class SummaryRequest(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "Albert Einstein var son till Hermann och Pauline Einstein, vilka var icke-religiösa judar och tillhörde medelklassen. Fadern var försäljare och drev senare en elektroteknisk fabrik. Familjen bosatte sig 1880 i München där Einstein gick i en katolsk skola. Mängder av myter cirkulerar om Albert Einsteins person. En av dessa är att han som barn skulle ha haft svårigheter med matematik, vilket dock motsägs av hans utmärkta betyg i ämnet.[15] Han nämnde ofta senare att han inte trivdes i skolan på grund av dess pedagogik. Att Albert Einstein skulle vara släkt med musikvetaren Alfred Einstein är ett, ofta framfört, obevisat påstående. Alfred Einsteins dotter Eva har framhållit att något sådant släktskap inte existerar."
            }
        }



class ZeroShotRequest(BaseModel):
    sequence: str
    labels: list

    class Config:
        schema_extra = {
            "example": {
                "sequence": "one day I will see the world",
                "labels": ['travel', 'cooking', 'dancing']
            }
        }


class GenerateRequest(BaseModel):
    text: str
    model: Optional[str] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "Det var en gång",
                "model": "birgermoell/swedish-gpt",
                "max_length": 250,
                "temperature": 0.9
            }
        }


class QaRequest(BaseModel):
    question: str
    context: str

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the meaning of life?",
                "context": "The meaning of life is to be happy",
            }
        }


class PhonemeRequest(BaseModel):
    text: str
    language: str
    model: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Tal, Musik Hörsel är en underbar plats.",
                "language": "Swedish",
            }
        }


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

app.mount("/models", StaticFiles(directory="models"), name="models")

@app.get("/")
def read_root():
    return {"Hello": "World"}


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

@app.post("/ner")
async def qa_response(textRequest: TextRequest):
    
    from tmh.text.ner import named_entity_recognition

    ner = named_entity_recognition(textRequest.text)
    print(ner)
    ner = ner

    cleaned = []

    for item in ner:
        item['score'] = float(item['score'])
        item['start'] = int(item['start'])
        item['end'] = int(item['end'])
        cleaned.append(item)

    return {
      "ner":  cleaned
    }

@app.post("/qa")
async def qa_response(qaRequest: QaRequest):
    
    from tmh.text.question_answering import get_answer

    answer = get_answer({'question': qaRequest.question, 'context': qaRequest.context})

    print("the answer response is", answer)

    return answer

@app.post("/zero_shot")
async def qa_response(zeroShotRequest: ZeroShotRequest):
    
    from tmh.text.zero_shot import get_zero_shot_classification

    classified_label = get_zero_shot_classification(zeroShotRequest.sequence, zeroShotRequest.labels)

    print("the classified label response is", classified_label)

    return classified_label

@app.post("/translate_and_summarize")
async def qa_response(summaryRequest: SummaryRequest):
    
    from tmh.text.summarization import translate_and_summarize

    swedish_summary = translate_and_summarize(summaryRequest.text)
    print(swedish_summary)

    return swedish_summary

@app.post("/phonemize")
async def qa_response(phonemeRequest: PhonemeRequest):
    print("inside phonemize with", phonemeRequest)
    
    from tmh.phonemes import get_phonemes
    # import pdb
    # pdb.set_trace()

    phoneme_models = {
        "English": "models/en_us_cmudict_ipa_forward.pt",
        "Swedish": "models/best_model_no_optim.pt"
    }

    phonemes = get_phonemes(phonemeRequest.text, phoneme_models[phonemeRequest.language], language=phonemeRequest.language)
    
    print(phonemes)

    return phonemes

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)