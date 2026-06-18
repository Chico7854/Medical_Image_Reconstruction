import numpy as np
import time
import asyncio
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Preload system matrices
H_map = {
    'H-1.csv': np.loadtxt('H/H-1.csv', delimiter=','),
    'H-2.csv': np.loadtxt('H/H-2.csv', delimiter=','),
}

# In-memory storage for results indexed by a session or client run
# In production, use a unique session ID per client
resultados_storage: List[dict] = []

class Requisicao(BaseModel):
    sinal: list
    h: str
    nome: str
    algoritmo: str
    e_ultimo: bool  # Flag indicating the last payload

# (cgne and cgnr functions remain exactly the same as your original code)
def cgne(H, g, max_iter=100, epsilon=1e-4):
    inicio = time.time()
    lambd = np.max(np.abs(H.T @ g)) * 0.10
    f = np.zeros(H.shape[1])
    r = g.copy()
    p = H.T @ r
    for i in range(max_iter):       
        alpha = (r.T @ r) / ((p.T @ p) + lambd * (r.T @ r))
        f = f + alpha * p
        r_novo = r - alpha * ((H @ p) + lambd * r)
        beta = (r_novo.T @ r_novo) / (r.T @ r)
        p = (H.T @ r_novo) + (beta * p)
        erro = abs(np.linalg.norm(r_novo) - np.linalg.norm(r))
        r = r_novo
        iteracoes = i + 1
        if erro < epsilon: break
    tempo = round(time.time() - inicio, 4)
    return np.abs(f), iteracoes, tempo

def cgnr(H, g, max_iter=100, epsilon=1e-4):
    inicio = time.time()
    lambd = np.max(np.abs(H.T @ g)) * 0.10
    f = np.zeros(H.shape[1])
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    for i in range(max_iter):
        w = H @ p
        alpha = (z @ z) / ((w @ w) + lambd * (p @ p))
        f = f + alpha * p
        r_novo = r - alpha * w
        z_novo = H.T @ r_novo - lambd * f
        beta = (z_novo @ z_novo) / (z @ z)
        p = z_novo + beta * p
        erro = abs(np.linalg.norm(r_novo) - np.linalg.norm(r))
        r = r_novo
        z = z_novo
        iteracoes = i + 1
        if erro < epsilon: break
    tempo = round(time.time() - inicio, 4)
    return np.abs(f), iteracoes, tempo

# Global counter to track active background tasks
processando_count = 0
processando_lock = asyncio.Lock()

def processar_reconstrucao_background(req: Requisicao):
    global processando_count
    inicio_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    H = H_map[req.h]
    g = np.array(req.sinal)
    tamanho = int(np.sqrt(H.shape[1]))
    
    if req.algoritmo == 'cgne':
        f, iteracoes, tempo = cgne(H, g)
    else:    
        f, iteracoes, tempo = cgnr(H, g)

    imagem = f.reshape(tamanho, tamanho).T.tolist()
    fim_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    resultado = {
        'nome': req.nome,
        'imagem': imagem,
        'iteracoes': iteracoes,
        'tempo': tempo,
        'tamanho': f"{tamanho}x{tamanho}",
        'inicio': inicio_str,
        'fim': fim_str,
        'algoritmo': req.algoritmo
    }
    
    resultados_storage.append(resultado)
    print(f"[BACKGROUND] Processado {req.nome}")

    # Decrement active task counter safely
    async def sub_counter():
        global processando_count
        async with processando_lock:
            processando_count -= 1
            
    asyncio.run(sub_counter())

@app.post("/reconstruir")
async def reconstruir(req: Requisicao, background_tasks: BackgroundTasks):
    global processando_count
    async with processando_lock:
        processando_count += 1
        
    # Trigger execution in background thread without waiting
    background_tasks.add_task(processar_reconstrucao_background, req)
    
    return {"status": "recebido", "e_ultimo": req.e_ultimo}

@app.get("/resultados")
async def obter_resultados():
    global processando_count
    # Long poll loop: Wait until all processing is complete and we have results
    while True:
        async with processando_lock:
            if processando_count == 0 and len(resultados_storage) > 0:
                break
        await asyncio.sleep(0.5)
        
    # Clear and extract results
    resposta = list(resultados_storage)
    resultados_storage.clear()
    return {"status": "sucesso", "dados": resposta}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)