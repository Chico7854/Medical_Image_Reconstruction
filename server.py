import numpy as np
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()
executor = ProcessPoolExecutor(max_workers=8)

# Carrega os dois H na memória ao iniciar o servidor
H_map = {
    'H-1.csv': np.loadtxt('H/H-1.csv', delimiter=','),
    'H-2.csv': np.loadtxt('H/H-2.csv', delimiter=','),
}

class Requisicao(BaseModel):
    sinal: list
    h: str
    nome: str

def cgnr(H, g, max_iter=10, epsilon=1e-4):
    inicio = time.time()
    
    f = np.zeros(H.shape[1])
    r = g - H @ f
    z = H.T @ r
    p = z.copy()

    iteracoes = 0
    for i in range(max_iter):
        w = H @ p
        alpha = (z @ z) / (w @ w)
        
        f = f + alpha * p
        r_novo = r - alpha * w
        z_novo = H.T @ r_novo
        
        beta = (z_novo @ z_novo) / (z @ z)
        p = z_novo + beta * p
        
        erro = abs((r_novo @ r_novo) - (r @ r)) / (r @ r)
        
        r = r_novo
        z = z_novo
        iteracoes = i + 1
        
        if erro < epsilon:
            break

    tempo = round(time.time() - inicio, 4)
    return f, iteracoes, tempo

def processar(h_nome, sinal, nome):
    H = np.loadtxt(f'H/{h_nome}', delimiter=',')
    g = np.array(sinal)
    tamanho = int(np.sqrt(H.shape[1]))

    inicio_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f, iteracoes, tempo = cgnr(H, g)
    fim_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    imagem = f.reshape(tamanho, tamanho).T.tolist()

    return {
        'nome': nome,
        'imagem': imagem,
        'iteracoes': iteracoes,
        'tempo': tempo,
        'tamanho': f"{tamanho}x{tamanho}",
        'inicio': inicio_str,
        'fim': fim_str,
    }

@app.post("/reconstruir")
async def reconstruir(req: Requisicao):
    import asyncio
    loop = asyncio.get_event_loop()
    resultado = await loop.run_in_executor(executor, processar, req.h, req.sinal, req.nome)
    
    print(f"[{resultado['fim']}] {req.nome} — {resultado['iteracoes']} iterações — {resultado['tempo']}s")
    
    return resultado