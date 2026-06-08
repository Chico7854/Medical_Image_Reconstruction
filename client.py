import numpy as np
import requests
import random
import time
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Mapeamento de sinais para o H correspondente
sinais = {
    'G-1.csv':       'H-1.csv',
    'G-2.csv':       'H-1.csv',
    'A-60x60-1.csv': 'H-1.csv',
    'g-30x30-1.csv': 'H-2.csv',
    'g-30x30-2.csv': 'H-2.csv',
    'A-30x30-1.csv': 'H-2.csv',
}

SERVIDOR = 'http://localhost:8000'
relatorio = []

os.makedirs('images', exist_ok=True)

def aplicar_ganho(g):
    S = len(g)
    g_com_ganho = g.copy()
    for l in range(1, S + 1):
        gamma = 100 + (1/20) * np.sqrt(l * l)
        g_com_ganho[l-1] = g_com_ganho[l-1] * gamma
    return g_com_ganho

def salvar_imagem(imagem_lista, path):
    imagem = np.array(imagem_lista)
    imagem_norm = (imagem - imagem.min()) / (imagem.max() - imagem.min())
    imagem_real = np.exp(imagem_norm)
    plt.imshow(imagem_real, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def enviar_sinal(nome_g, nome_h):
    g = np.loadtxt("sinais/" + nome_g)
    g = aplicar_ganho(g)

    payload = {
        'sinal': g.tolist(),
        'h': nome_h,
        'nome': nome_g,
    }
    
    print(f"[{datetime.now()}] Enviando {nome_g} com {nome_h}...")
    resposta = requests.post(f"{SERVIDOR}/reconstruir", json=payload)
    
    return resposta.json()

# Loop principal — envia sinais em ordem
try:
    for nome_g, nome_h in sinais.items():
        resultado = enviar_sinal(nome_g, nome_h)
        
        path = f"images/{nome_g.replace('.csv', '')}.png"
        salvar_imagem(resultado['imagem'], path)

        relatorio.append({
            'nome': resultado['nome'],
            'imagem': path,
            'iteracoes': resultado['iteracoes'],
            'tempo': resultado['tempo'],
            'tamanho': resultado['tamanho'],
            'inicio': resultado['inicio'],
            'fim': resultado['fim'],
            'h_usado': nome_h,
        })
        
        print(f"  → H usado: {nome_h}")
        print(f"  → Imagem salva em {path}")
        print(f"  → Iterações: {resultado['iteracoes']}")
        print(f"  → Tempo: {resultado['tempo']}s")
        
        intervalo = random.uniform(1, 5)
        print(f"  → Próximo envio em {intervalo:.1f}s\n")
        time.sleep(intervalo)

    print("\nTodos os sinais enviados!")

finally:
    with open('relatorio.json', 'w') as f:
        json.dump(relatorio, f, indent=2)
    print(f"Relatório salvo com {len(relatorio)} reconstruções.")