from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Doit Helper",
    description="FastAPI app for Doit automation helper with Claude Haiku integration via OpenRouter",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DoitQuestion(BaseModel):
    question: str
    context: Optional[str] = None

class DoitResponse(BaseModel):
    answer: str
    suggestions: Optional[List[str]] = None
    helpful: bool = True

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not found in environment variables")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Doit Helper API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "doit-helper",
        "version": "1.0.0",
        "openrouter_configured": bool(OPENROUTER_API_KEY)
    }

@app.post("/doit-helper", response_model=DoitResponse)
async def doit_helper(request: DoitQuestion):
    """
    Main endpoint to help with Doit questions
    Sends questions to Claude Haiku via OpenRouter and returns helpful technical responses
    """
    
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OpenRouter API key not configured"
        )
    
    try:
        # System prompt for Doit expertise
        system_prompt = """Você é um especialista em Doit (Python Task Management & Automation Tool).

Contexto técnico do Doit:
- Doit é uma ferramenta Python para automação de tarefas e build systems
- Usa arquivos dodo.py para definir tarefas
- Comandos principais: doit list, doit run, doit clean, doit forget
- Suporta dependências entre tarefas, cache inteligente e execução paralela
- Ideal para pipelines de dados, builds automatizados e workflows
- Configuração via YAML, Python ou linha de comando
- Integração com Make, CMake, SCons e outras ferramentas

Exemplos comuns:
1. Instalação: pip install doit
2. Criar dodo.py com def task_hello(): return {'actions': ['echo Hello']}
3. Executar: doit run
4. Listar tarefas: doit list

Responda de forma clara, amigável e com passo a passo detalhado.
Se possível, inclua comandos prontos e links úteis.
Sempre forneça exemplos práticos e soluções completas.
Use tom técnico mas acessível, em português brasileiro."""

        # Prepare the user message
        user_message = request.question
        if request.context:
            user_message = f"Contexto: {request.context}\n\nPergunta: {request.question}"

        # Call OpenRouter API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://doit-helper.railway.app",
                    "X-Title": "Doit Helper"
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail="Error communicating with AI service"
                )
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # Generate intelligent suggestions
            suggestions = []
            question_lower = request.question.lower()
            
            if "install" in question_lower:
                suggestions.append("Teste a instalação com 'doit --version'")
                suggestions.append("Consulte https://pydoit.org/install.html")
            
            if "yaml" in question_lower:
                suggestions.append("Valide seu YAML com ferramentas online antes de usar")
                suggestions.append("Use doit.tools.config_changed para detectar mudanças")
            
            if "erro" in question_lower or "error" in question_lower:
                suggestions.append("Execute 'doit clean' para limpar cache")
                suggestions.append("Use 'doit -v 2' para debug detalhado")
            
            if not suggestions:
                suggestions.append("Consulte a documentação oficial do Doit em https://pydoit.org/")
            
            return DoitResponse(
                answer=ai_response,
                suggestions=suggestions,
                helpful=True
            )
            
    except httpx.TimeoutException:
        logger.error("Timeout calling OpenRouter API")
        raise HTTPException(
            status_code=504,
            detail="AI service timeout - please try again"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
