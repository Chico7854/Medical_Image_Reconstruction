import numpy as np
import time
import asyncio
import threading
import psutil
import csv
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

H_map = {
    'H-1.csv': np.loadtxt('H/H-1.csv', delimiter=','),
    'H-2.csv': np.loadtxt('H/H-2.csv', delimiter=','),
}

resultados_storage: Dict[str, List[dict]] = {}
processando_count: Dict[str, int] = {}
# Track if the client has declared it is done sending
finalizado_flags: Dict[str, bool] = {}
storage_lock = asyncio.Lock()

class Requisicao(BaseModel):
    sinal: list
    h: str
    nome: str
    algoritmo: str
    client_id: str

METRICS_FILE = "metrics_py.csv"
tempo_inicial = None
cronometro_iniciado = threading.Event()

def monitorar_recursos():
    global tempo_inicial
    cronometro_iniciado.wait()
    tempo_inicial = time.perf_counter()
    
    with open(METRICS_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Tempo_Decorrido_S", "System_Total_CPU_Percent", "System_Used_Memory_GB"])
    
    while True:
        try:
            tempo_decorrido = round(time.perf_counter() - tempo_inicial, 2)
            sys_cpu = psutil.cpu_percent()
            sys_mem = psutil.virtual_memory().used / (1024 * 1024 * 1024)
            with open(METRICS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([tempo_decorrido, sys_cpu, round(sys_mem, 2)])
            time.sleep(0.5)
        except Exception as e:
            time.sleep(1)

monitor_thread = threading.Thread(target=monitorar_recursos, daemon=True)
monitor_thread.start()

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

@app.post("/reconstruir")
async def reconstruir(req: Requisicao, background_tasks: BackgroundTasks):
    if not cronometro_iniciado.is_set():
        cronometro_iniciado.set()

    async with storage_lock:
        if req.client_id not in processando_count:
            processando_count[req.client_id] = 0
            finalizado_flags[req.client_id] = False
        processando_count[req.client_id] += 1
        
    background_tasks.add_task(processar_reconstrucao_background, req)
    return {"status": "recebido"}

@app.post("/finalizar/{client_id}")
async def finalizar_cliente(client_id: str):
    async with storage_lock:
        finalizado_flags[client_id] = True
    return {"status": "sinalizado"}

@app.get("/resultados/{client_id}")
async def obter_resultados(client_id: str):
    async with storage_lock:
        terminar = False
        # Only true if sender is done AND background worker thread counts are empty
        if finalizado_flags.get(client_id, False) and processando_count.get(client_id, 0) == 0:
            if client_id not in resultados_storage or len(resultados_storage[client_id]) == 0:
                terminar = True
            
        dados = resultados_storage.pop(client_id, [])
        if terminar:
            processando_count.pop(client_id, None)
            finalizado_flags.pop(client_id, None)
            
        return {"status": "sucesso", "concluido": terminar, "dados": dados}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)