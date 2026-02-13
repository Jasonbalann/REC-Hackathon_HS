import os
from dotenv import load_dotenv 
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

app = FastAPI()

# --- 1. PYDANTIC MODELS (To catch data from HTML/JavaScript) ---
class Transaction(BaseModel):
    date: str
    desc: str
    cat: str
    amount: float

class ChatContext(BaseModel):
    name: str
    income: float
    expenses: float
    savings: float
    goals: List[str]
    transactions: List[Transaction]

class ChatRequest(BaseModel):
    text: str
    context: ChatContext

class InsightRequest(BaseModel):
    context: dict
    transactions: List[Transaction]

class AlertRequest(BaseModel):
    type: str
    context: dict

# --- 2. SETUP & CORS ---
@app.get("/")
async def read_index():
    return FileResponse('index.html')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LLM SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Or gemini-2.5-flash depending on your key access
    temperature=0.7,
    convert_system_message_to_human=True 
)

@app.get("/")
def health():
    return {"status": "ok"}

# --- 4. CHAT ENDPOINT (Now reads live HTML data) ---
system_template = """
You are a Financial Assistant for a user in India. 
All financial figures and calculations MUST be presented in Indian Rupees (₹).
Use this live user dashboard data to answer:
Profile: {profile}
Accounts: {accounts}
Recent Transactions: {history}
Goals: {goals}

Keep answers concise (max 3 sentences).
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{input}")
])

chat_chain = chat_prompt | llm | StrOutputParser()

@app.post("/chat")
async def chat(request: ChatRequest):
    # Convert the dynamic transactions list into a readable string for the AI
    txn_str = "\n".join([f"{t.date}: {t.desc} ({t.cat}) - ₹{t.amount}" for t in request.context.transactions])
    if not txn_str:
        txn_str = "No recent transactions."

    accounts_str = f"Income: ₹{request.context.income}, Expenses: ₹{request.context.expenses}, Savings: ₹{request.context.savings}"

    response = chat_chain.invoke({
        "input": request.text,
        "profile": f"Name: {request.context.name}",
        "accounts": accounts_str,
        "history": txn_str,
        "goals": ", ".join(request.context.goals)
    })
    return {"reply": response}

# --- 5. NEW: INSIGHT ENDPOINT (Fixes the "Analyzing patterns..." bug) ---
insight_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial AI. Give a single, brief sentence (max 10 words) of financial advice based on this current month data: Income ₹{income}, Expenses ₹{expenses}."),
    ("user", "Give me a quick dashboard insight.")
])
insight_chain = insight_prompt | llm | StrOutputParser()

@app.post("/get-insight")
async def get_insight(request: InsightRequest):
    insight = insight_chain.invoke({
        "income": request.context.get("income", 0),
        "expenses": request.context.get("expenses", 0)
    })
    return {"insight": insight}

# --- 6. ALERTS ENDPOINT (Now uses live HTML data) ---
@app.post("/generate-alert")
async def generate_alert(request: AlertRequest):
    alert_type = request.type
    income = request.context.get('income', 0)
    expenses = request.context.get('expenses', 0)
    savings = request.context.get('savings', 0)
    
    scenarios = {
        "overspending": f"User's current expenses are ₹{expenses} against an income of ₹{income}.",
        "bill": f"Reminder to pay bills. Current savings available: ₹{savings}.",
        "summary": f"Dashboard update. Total savings: ₹{savings}."
    }
    context = scenarios.get(alert_type, "General update")

    notify_prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a single, urgent, 10-15 word push notification for an Indian banking app (use ₹ symbol)."),
        ("user", f"Context: {context}")
    ])
    
    alert_chain = notify_prompt | llm | StrOutputParser()
    alert_text = alert_chain.invoke({})
    
    return {"message": alert_text}