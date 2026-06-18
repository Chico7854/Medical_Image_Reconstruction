import numpy as np
import requests
import time
import json
import os
import threading
import random
import matplotlib.pyplot as plt
from datetime import datetime

SERVIDOR = 'http://localhost:8000'

def salvar_imagem(imagem_lista, path):
    imagem = np.array(imagem_lista)
    plt.imshow(imagem, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def enviar_sinal(nome_g, nome_h, algoritmo, client_id):
    g = np.loadtxt("sinais/" + nome_g)
    payload = {
        'sinal': g.tolist(),
        'h': nome_h,
        'nome': nome_g,
        'algoritmo': algoritmo,
        'client_id': client_id
    }
    resposta = requests.post(f"{SERVIDOR}/reconstruir", json=payload)
    return resposta.json()

def executar_pipeline_cliente(client_id, tasks):
    output_dir = f"images_{client_id}"
    os.makedirs(output_dir, exist_ok=True)
    relatorio = []
    total = len(tasks)
    
    print(f"[{client_id}] Starting thread. Total tasks: {total}\n")
    
    # 1. Fire-and-forget loop
    for idx, task in enumerate(tasks):
        nome_g = task["nome_g"]
        nome_h = task["nome_h"]
        delay = task["delay"]
        
        # Dynamically randomize the algorithm per request
        algoritmo = random.choice(['cgne', 'cgnr'])
        
        print(f"[{client_id}] (Progress {idx+1}/{total}) Sending {nome_g} via {algoritmo.upper()}...")
        enviar_sinal(nome_g, nome_h, algoritmo, client_id)
        time.sleep(delay)

    # 2. Long polling loop waiting for back-end processing
    print(f"\n[{client_id}] Queue processed. Waiting for server computations...")
    resposta_servidor = requests.get(f"{SERVIDOR}/resultados/{client_id}").json()

    # 3. Extract completed matrices
    if resposta_servidor.get("status") == "sucesso":
        for idx, resultado in enumerate(resposta_servidor["dados"]):
            path = f"{output_dir}/{idx:02d}_{resultado['nome'].replace('.csv', '')}.png"
            salvar_imagem(resultado['imagem'], path)

            relatorio.append({
                'nome': resultado['nome'],
                'imagem': path,
                'iteracoes': resultado['iteracoes'],
                'tempo': resultado['tempo'],
                'tamanho': resultado['tamanho'],
                'inicio': resultado['inicio'],
                'fim': resultado['fim'],
                'algoritmo': resultado['algoritmo']
            })
        print(f"[{client_id}] Finished completely. Outputs written to ./{output_dir}")

    # Write report file specific to this thread
    with open(f'relatorio_{client_id}.json', 'w') as f:
        json.dump(relatorio, f, indent=2)


if __name__ == "__main__":
    if not os.path.exists('config.json'):
        print("Error: config.json not found. Run generate_config.py first.")
        exit(1)
        
    with open('config.json', 'r') as f:
        config_total = json.load(f)

    threads = []
    
    for client_id, tasks in config_total.items():
        t = threading.Thread(target=executar_pipeline_cliente, args=(client_id, tasks))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n[MAIN] All 3 client worker threads completed execution.")