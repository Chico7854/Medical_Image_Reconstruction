# Medical Image Reconstruction

Aplicação para reconstrução de imagens médicas a partir de sinais, utilizando o algoritmo CGNR (Conjugate Gradient for Normal Equations). O servidor recebe um sinal e uma matriz H, executa a reconstrução e retorna a imagem resultante.

## Requisitos

- Python 3.10+
- As matrizes H (`H-1.csv`, `H-2.csv`) na pasta `H/` — **não incluídas no repositório** por serem arquivos grandes. Você precisa obtê-las separadamente e colocá-las em `H/` antes de iniciar o servidor.
- Os sinais G (`.csv`) na pasta `sinais/`

## Instalação

```bash
pip install fastapi uvicorn numpy matplotlib requests
```

## Como rodar

### 1. Iniciar o servidor

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

O servidor ficará disponível em `http://localhost:8000`.

### 2. Executar o cliente

Em outro terminal, na mesma pasta do projeto:

```bash
python3 client.py
```

O cliente enviará cada sinal ao servidor com o H correspondente, salvará as imagens reconstruídas em `images/` e gerará um `relatorio.json` ao final.

## Mapeamento de sinais

| Sinal | H utilizado |
|---|---|
| G-1.csv | H-1.csv |
| G-2.csv | H-1.csv |
| A-60x60-1.csv | H-1.csv |
| g-30x30-1.csv | H-2.csv |
| g-30x30-2.csv | H-2.csv |
| A-30x30-1.csv | H-2.csv |

## Saídas

- `images/` — imagens reconstruídas em `.png`
- `relatorio.json` — registro de cada reconstrução com nome, iterações, tempo e H utilizado