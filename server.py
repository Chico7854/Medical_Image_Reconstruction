import numpy as np
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

H_map = {
    'H-1.csv': np.loadtxt('H/H-1.csv', delimiter=','),
    'H-2.csv': np.loadtxt('H/H-2.csv', delimiter=','),
}

class Requisicao(BaseModel):
    sinal: list
    h: str
    nome: str

def cgne(H, g, max_iter=10, epsilon=1e-4):
    inicio = time.time()

    f = np.zeros(H.shape[1])
    r = g
    p = H.T @ r

    for i in range(max_iter):       
        alpha = (r.T @ r) / (p.T @ p)
        
        f = f + alpha * p
        r_novo = r - alpha * (H @ p)
        
        beta = (r_novo.T @ r_novo) / (r.T @ r)
        
        p = (H.T @ r_novo) + (beta * p)
        
        erro = np.linalg.norm(r_novo) - np.linalg.norm(r)
        r = r_novo
        iteracoes = i + 1

        if erro < epsilon:
            break

    tempo = round(time.time() - inicio, 4)
    return f, iteracoes, tempo

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
    f = np.log(np.abs(f))
    return f, iteracoes, tempo

@app.post("/reconstruir")
def reconstruir(req: Requisicao):
    inicio_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    H = H_map[req.h]
    g = np.array(req.sinal)
    tamanho = int(np.sqrt(H.shape[1]))

    f, iteracoes, tempo = cgne(H, g)

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
    }

    print(f"[{fim_str}] {req.nome} — {iteracoes} iterações — {tempo}s")
    return resultado