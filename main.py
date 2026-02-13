import os
from dotenv import load_dotenv 
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

@app.get("/")
async def read_index():
    return FileResponse('index.html')

# --- 2. CORS SETUP (Allows Frontend to connect) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. MOCK DATABASE (Your simulated history) ---
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


# --- 4. LLM SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    convert_system_message_to_human=True 
)

@app.get("/")
def health():
    return {"status": "ok"}

# --- 5. CHAT ENDPOINT ---
system_template = """
You are a Financial Assistant for a user in India. 
All financial figures and calculations MUST be presented in Indian Rupees (₹).
Use this user data to answer:
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

# --- 6. NEW: NOTIFICATION ENDPOINT (For the Bonus) ---
@app.post("/generate-alert")
async def generate_alert(data: dict = Body(...)):
    alert_type = data.get("type")
    
    # Define scenarios based on DB data
    scenarios = {
        "overspending": f"User spent $500 on 'Luxury'. Checking balance is now ${mock_db['accounts']['checking'] - 500}.",
        "bill": f"Rent of $2000 is due. Current Checking: ${mock_db['accounts']['checking']}.",
        "summary": f"End of month. Saved: ${mock_db['accounts']['savings']}. Debt: ${mock_db['accounts']['debt']}."
    }
    
    context = scenarios.get(alert_type, "General update")

    # Specialized Prompt for Notifications
    notify_prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a single, urgent, 10-15 word push notification for a banking app."),
        ("user", f"Context: {context}")
    ])
    
    alert_chain = notify_prompt | llm | StrOutputParser()
    alert_text = alert_chain.invoke({})
    
    return {"message": alert_text}

 