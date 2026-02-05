from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

print("🔥 MAIN.PY LOADED 🔥")

app = FastAPI()

# CORS - allow frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GenerateRequest(BaseModel):
    angry_message: str
    platform: str | None = None
    company_policy: str | None = None

SYSTEM_PROMPT = """You are an expert customer dispute response writer.

Your job:
- De-escalate angry customers
- Avoid admitting legal or financial liability
- Keep responses compliant with platform policies
- Sound calm, professional, and human

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
    
    # Build user prompt
    user_prompt = f"Customer message:\n{req.angry_message}\n\n"
    
    if req.platform:
        user_prompt += f"Platform: {req.platform}\n\n"
    
    if req.company_policy and req.company_policy.strip():
        user_prompt += f"Company Policy:\n{req.company_policy}\n\n"
    
    user_prompt += "Write a professional, de-escalating reply:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        reply = response.choices[0].message.content.strip()
        
        return {"reply": reply}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Customer reply generator API"}
