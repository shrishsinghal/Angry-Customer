from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from bs4 import BeautifulSoup


async def fetch_policy_from_url(url: str) -> str:
    """Fetch and extract text from a policy URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0, follow_redirects=True)
            
            if response.status_code != 200:
                return ""
            
            # Parse HTML and extract text
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit to 2000 chars to avoid huge prompts
            return text[:2000]
    except:
        return ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    angry_message: str
    platform: str | None = None
    company_policy: str | None = None
    policy_url: str | None = None  # NEW

SYSTEM_PROMPT = """You are an expert customer dispute response writer.

Your job:
- De-escalate angry customers
- Avoid admitting legal or financial liability
- Keep responses compliant with platform policies
- Sound calm, professional, and human
- Do not raigbait the customer

Rules:
- Never admit fault unless explicitly instructed
- Never promise refunds unless policy allows it
- Never sound robotic or threatening
- Acknowledge emotions without agreeing with accusations
- Reference company policy ONLY if provided
- If policy is unclear, keep response neutral and procedural

Output:
- One concise reply (2-4 short paragraphs max)
- Clear next steps
- Professional tone"""

@app.post("/generate")
async def generate_reply(req: GenerateRequest):
    if not req.angry_message or len(req.angry_message.strip()) < 10:
        raise HTTPException(status_code=400, detail="Message too short")
    
    # Handle policy - either from text or URL
    policy_text = req.company_policy or ""
    
    if req.policy_url and not policy_text:
        fetched = await fetch_policy_from_url(req.policy_url)
        if fetched:
            policy_text = fetched
    
    user_prompt = f"Customer message:\n{req.angry_message}\n\n"
    
    if req.platform:
        user_prompt += f"Platform: {req.platform}\n\n"
    
    if policy_text.strip():
        user_prompt += f"Company Policy:\n{policy_text}\n\n"
    
    user_prompt += "Write a professional, de-escalating reply:"
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"API error: {response.text}")
            
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            
            return {"reply": reply}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Customer reply generator API"}