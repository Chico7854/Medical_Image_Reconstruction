import numpy as np
import requests
import time
import json
import os
import threading
import random
from datetime import datetime
from matplotlib.figure import Figure

SERVIDOR = 'http://localhost:8000'

def salvar_imagem(imagem_lista, path):
    imagem = np.array(imagem_lista)
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(imagem, cmap='gray')
    ax.axis('off')
    fig.savefig(path, bbox_inches='tight', pad_inches=0)

def aplicar_ganho_sinal(g, fator):
    S = len(g)
    for i in range(1, S + 1):
        gamma = 100 + ((1/20) * i * np.sqrt(i))
    return g * gamma * fator

def enviar_sinal(nome_g, nome_h, algoritmo, client_id, fator):
    g = np.loadtxt("sinais/" + nome_g)
    g_modificado = aplicar_ganho_sinal(g, fator)

    payload = {
        'sinal': g_modificado.tolist(),
        'h': nome_h,
        'nome': nome_g,
        'algoritmo': algoritmo,
        'client_id': client_id
    }
    try:
        requests.post(f"{SERVIDOR}/reconstruir", json=payload)
    except Exception as e:
        print(f"[{client_id} SENDER ERROR] Error sending payload: {e}")

def receber_resultados_loop(client_id, output_dir, relatorio, total_tasks):
    recebidos_count = 0
    print(f"[{client_id} RECEIVER] Receiver loop thread active.")
    
    while True:
        try:
            resposta = requests.get(f"{SERVIDOR}/resultados/{client_id}").json()
            if resposta.get("status") == "sucesso":
                for resultado in resposta["dados"]:
                    path = f"{output_dir}/{recebidos_count:02d}_{resultado['nome'].replace('.csv', '')}.png"
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
                    recebidos_count += 1
                    print(f"[{client_id} RECEIVER] (Progress {recebidos_count}/{total_tasks}) Processed and saved {resultado['nome']}")

                if resposta.get("concluido") is True:
                    break
        except Exception as e:
            print(f"[{client_id} RECEIVER ERROR] {e}")
            
        time.sleep(0.5)
        
    print(f"[{client_id} RECEIVER] Done fetching all calculations. Writing manifest.")

def executar_pipeline_cliente(client_id, tasks):
    output_dir = f"images_{client_id}"
    os.makedirs(output_dir, exist_ok=True)
    relatorio = []
    total = len(tasks)
    
    receiver_thread = threading.Thread(
        target=receber_resultados_loop, 
        args=(client_id, output_dir, relatorio, total)
    )
    receiver_thread.start()
    
    print(f"[{client_id} SENDER] Starting submission line. Total tasks: {total}")
    
    for idx, task in enumerate(tasks):
        nome_g = task["nome_g"]
        nome_h = task["nome_h"]
        delay = task["delay"]
        fator = task["fator"]
        algoritmo = random.choice(['cgne', 'cgnr'])
        
        # Log active upload tracking progress
        print(f"[{client_id} SENDER] (Progress {idx+1}/{total}) Sending {nome_g} via {algoritmo.upper()} (Gain Factor: {fator})...")
        enviar_sinal(nome_g, nome_h, algoritmo, client_id, fator)
        time.sleep(delay)

    print(f"\n[{client_id} SENDER] All inputs sent. Informing server...")
    try:
        requests.post(f"{SERVIDOR}/finalizar/{client_id}")
    except Exception as e:
        print(f"[{client_id} SENDER ERROR] Finalize error: {e}")
    
    receiver_thread.join()

    with open(f'relatorio_{client_id}.json', 'w') as f:
        json.dump(relatorio, f, indent=2)
    print(f"[{client_id}] Complete. Pipeline clean.\n")


if __name__ == "__main__":
    if not os.path.exists('config.json'):
        print("Error: config.json not found.")
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

    print("\n[MAIN] All client streams shut down smoothly.")