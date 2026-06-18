import numpy as np
import requests
import random
import time
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

sinais = {
    'G-1.csv':       'H-1.csv',
    'G-2.csv':       'H-1.csv',
    'A-60x60-1.csv': 'H-1.csv',
    'g-30x30-1.csv': 'H-2.csv',
    'g-30x30-2.csv': 'H-2.csv',
    'A-30x30-1.csv': 'H-2.csv',
}

SERVIDOR = 'http://localhost:8000'
os.makedirs('images', exist_ok=True)
relatorio = []

def salvar_imagem(imagem_lista, path):
    imagem = np.array(imagem_lista)
    plt.imshow(imagem, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def enviar_sinal(nome_g, nome_h, algoritmo, e_ultimo):
    g = np.loadtxt("sinais/" + nome_g)
    payload = {
        'sinal': g.tolist(),
        'h': nome_h,
        'nome': nome_g,
        'algoritmo': algoritmo,
        'e_ultimo': e_ultimo
    }
    print(f"[{datetime.now()}] Enviando {nome_g}...")
    resposta = requests.post(f"{SERVIDOR}/reconstruir", json=payload)
    return resposta.json()

try:
    itens = list(sinais.items())
    total = len(itens)

    # 1. Send all data immediately
    for idx, (nome_g, nome_h) in enumerate(itens):
        e_ultimo = (idx == total - 1)
        enviar_sinal(nome_g, nome_h, 'cgnr', e_ultimo)
        
        if not e_ultimo:
            intervalo = random.uniform(0.5, 1.5)
            time.sleep(intervalo)

    # 2. Start listening for the processing pipeline to complete
    print("\n[CLIENT] Todos os sinais enviados. Aguardando servidor finalizar o processamento...")
    resposta_servidor = requests.get(f"{SERVIDOR}/resultados").json()

    # 3. Process the downloaded payload bundle
    if resposta_servidor.get("status") == "sucesso":
        for resultado in resposta_servidor["dados"]:
            path = f"images/{resultado['nome'].replace('.csv', '')}.png"
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
            print(f"  → Processado remoto: {resultado['nome']} ({resultado['tempo']}s) | Salvo em {path}")

finally:
    with open('relatorio.json', 'w') as f:
        json.dump(relatorio, f, indent=2)
    print(f"\n[CLIENT] Relatório final criado com {len(relatorio)} itens.")