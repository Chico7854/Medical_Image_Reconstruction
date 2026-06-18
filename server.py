import numpy as np
import time
import asyncio
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Preload system matrices
H_map = {
    'H-1.csv': np.loadtxt('H/H-1.csv', delimiter=','),
    'H-2.csv': np.loadtxt('H/H-2.csv', delimiter=','),
}

class Requisicao(BaseModel):
    sinal: list
    h: str
    nome: str
    algoritmo: str

def cgne(H, g, max_iter=10, epsilon=1e-4):
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

        if erro < epsilon:
            break

    tempo = round(time.time() - inicio, 4)
    f = np.abs(f)
    return f, iteracoes, tempo

def cgnr(H, g, max_iter=10, epsilon=1e-4):
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
        z_novo = H.T @ r_novo - lambd * f # Corrected regularized residual step
        
        beta = (z_novo @ z_novo) / (z @ z)
        p = z_novo + beta * p
        
        erro = abs(np.linalg.norm(r_novo) - np.linalg.norm(r))
        
        r = r_novo
        z = z_novo
        iteracoes = i + 1
        
        if erro < epsilon:
            break

    tempo = round(time.time() - inicio, 4)
    f = np.log1p(np.abs(f))
    return f, iteracoes, tempo

@app.post("/reconstruir")
async def reconstruir(req: Requisicao):
    inicio_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    H = H_map[req.h]
    g = np.array(req.sinal)
    tamanho = int(np.sqrt(H.shape[1]))
    algoritmo = req.algoritmo

    # Offload CPU-bound matrix execution to avoid blocking the main event thread
    if algoritmo == 'cgne':
        f, iteracoes, tempo = await asyncio.to_thread(cgne, H, g)
    else:    
        f, iteracoes, tempo = await asyncio.to_thread(cgnr, H, g)

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
        'algoritmo': algoritmo
    }

    print(f"[{fim_str}] {req.nome} — {iteracoes} iterações — {tempo}s")
    return resultado

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)