import numpy as np
import time
import asyncio
import threading
import psutil
import csv
import os
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

resultados_storage: Dict[str, List[dict]] = {}
processando_count: Dict[str, int] = {}
storage_lock = asyncio.Lock()

class Requisicao(BaseModel):
    sinal: list
    h: str
    nome: str
    algoritmo: str
    client_id: str

# --- BACKGROUND MONITORING ---
METRICS_FILE = "metrics.csv"

def monitorar_recursos():
    """Background thread that samples CPU/Memory at a fast interval."""
    pid = os.getpid()
    processo = psutil.Process(pid)
    
    # Initialize CSV file with headers
    with open(METRICS_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", 
            "System_Total_CPU_Percent", 
            "System_Used_Memory_GB"
        ])
    
    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Global system metrics
            sys_cpu = psutil.cpu_percent()
            sys_mem = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # Convert to GB
            
            with open(METRICS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, sys_cpu, round(sys_mem, 2)])
                
            time.sleep(0.5)  # High resolution sampling
        except Exception as e:
            print(f"[MONITOR ERROR] {e}")
            time.sleep(1)

# Start the monitoring thread immediately on server startup
monitor_thread = threading.Thread(target=monitorar_recursos, daemon=True)
monitor_thread.start()
# -----------------------------

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

async def processar_reconstrucao_background(req: Requisicao):
    inicio_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    H = H_map[req.h]
    g = np.array(req.sinal)
    tamanho = int(np.sqrt(H.shape[1]))
    
    if req.algoritmo == 'cgne':
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
        'algoritmo': req.algoritmo
    }
    
    async with storage_lock:
        if req.client_id not in resultados_storage:
            resultados_storage[req.client_id] = []
        resultados_storage[req.client_id].append(resultado)
        processando_count[req.client_id] -= 1
        
    print(f"[BACKGROUND] Processado {req.nome} para {req.client_id}")

@app.post("/reconstruir")
async def reconstruir(req: Requisicao, background_tasks: BackgroundTasks):
    async with storage_lock:
        if req.client_id not in processando_count:
            processando_count[req.client_id] = 0
        processando_count[req.client_id] += 1
        
    background_tasks.add_task(processar_reconstrucao_background, req)
    return {"status": "recebido"}

@app.get("/resultados/{client_id}")
async def obter_resultados(client_id: str):
    while True:
        async with storage_lock:
            if client_id in processando_count and processando_count[client_id] == 0:
                if client_id in resultados_storage and len(resultados_storage[client_id]) > 0:
                    break
        await asyncio.sleep(0.5)
        
    async with storage_lock:
        resposta = list(resultados_storage[client_id])
        del resultados_storage[client_id]
        del processando_count[client_id]
        
    return {"status": "sucesso", "dados": resposta}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)