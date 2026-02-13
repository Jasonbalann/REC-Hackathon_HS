from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mock_db = {
    "profile": {"name": "Alex", "income": 5000},
    "accounts": {"checking": 1400, "savings": 12000, "debt": 500},
    "history": [
        "2023-10-01: Salary +$5000",
        "2023-10-05: Rent -$2000",
        "2023-10-10: Dining Out -$150",
        "2023-10-12: Uber -$45"
    ],
    "goals": ["Buy House", "Clear Debt"]
}

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

@app.get("/")
def health():
    return {"status": "ok"}

