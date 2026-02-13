import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
# --- CHANGED THESE TWO LINES BELOW ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# -------------------------------------
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

system_template = """
You are a Financial Assistant. Use this user data to answer:
Profile: {profile}
Accounts: {accounts}
Recent History: {history}
Goals: {goals}

Keep answers concise (max 3 sentences).
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{input}")
])

chat_chain = chat_prompt | llm | StrOutputParser()

@app.post("/chat")
async def chat(text: str = Body(..., embed=True)):

    response = chat_chain.invoke({
        "input": text,
        "profile": mock_db['profile'],
        "accounts": mock_db['accounts'],
        "history": "\n".join(mock_db['history']),
        "goals": mock_db['goals']
    })
    return {"reply": response}

