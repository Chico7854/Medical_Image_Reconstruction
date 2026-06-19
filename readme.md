# Medical Image Reconstruction

Aplicação para reconstrução de imagens médicas a partir de sinais, utilizando o algoritmo CGNR (Conjugate Gradient for Normal Equations). O projeto possui implementações de servidores de alta performance tanto em Python quanto em C++. O servidor recebe um sinal e uma matriz H, executa a reconstrução e retorna a imagem resultante.

## Requisitos do Sistema

- **Python 3.10+**
- **Compilador C++** com suporte a C++17 ou superior (ex: `g++`)
- As matrizes H (`H-1.csv`, `H-2.csv`) na pasta `H/` — **não incluídas no repositório** por serem arquivos grandes. Você precisa obtê-las separadamente e colocá-las em `H/` antes de iniciar qualquer servidor.
- Os sinais G (arquivos `.csv`) na pasta `sinais/`

## Instalação e Dependências

### 1. Dependências do Python (Servidor Python e Cliente)
Instale os pacotes necessários utilizando o gerenciador de pacotes `pip`:

```bash
pip install fastapi uvicorn numpy matplotlib requests psutil pydantic

```

### 2. Dependências do C++ (Servidor C++)

Para compilar e rodar o servidor em C++, é necessário garantir a presença das seguintes bibliotecas no seu ambiente:

* **Eigen 3** (Biblioteca de álgebra linear de alta performance)
* **Crow Framework** (Microframework web e roteador para C++)
* **Pthread** (Biblioteca nativa do sistema para suporte a multi-threading)

#### No Linux (Ubuntu/Debian):

Execute o comando abaixo no terminal para instalar o compilador, as ferramentas essenciais de build e os cabeçalhos da biblioteca Eigen:

```bash
sudo apt update
sudo apt install build-essential libeigen3-dev

```

*Nota: A instalação via `apt` extrai os arquivos exatamente no diretório `/usr/include/eigen3`, respeitando a flag de inclusão definida no seu Makefile (`-I/usr/include/eigen3`).*

#### Configurando o Crow Framework:

Como o Crow é uma biblioteca baseada inteiramente em arquivos de cabeçalho (*header-only*):

1. Baixe a versão ou os arquivos correspondentes do repositório oficial do Crow.
2. Certifique-se de que o arquivo `crow.h` (e as suas dependências internas) esteja localizado na **mesma pasta** do arquivo `server.cpp`, conforme a diretiva de inclusão local `#include "crow.h"`.

---

## Como Executar a Aplicação

### Opção A: Utilizando o Servidor C++ (Alta Performance)

1. **Compile o servidor** utilizando o Makefile contido na raiz do projeto:
```bash
make

```


2. **Inicie o binário gerado:**
```bash
./server

```


3. **Limpar arquivos compilados (Opcional):**
Caso precise limpar o binário e os relatórios gerados pelo C++, execute:
```bash
make clean

```



---

### Opção B: Utilizando o Servidor Python

1. **Inicie o servidor com o Uvicorn:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000

```



Qualquer um dos servidores selecionados acima estará escutando requisições no endereço `http://localhost:8000`.

---

### Executar o Cliente

Com o servidor ativo (seja a versão C++ ou Python), abra um novo terminal na mesma pasta e execute o script cliente:

```bash
python3 client.py

```

O cliente lerá sequencialmente os sinais configurados, enviará cada um ao servidor HTTP juntamente com os parâmetros necessários, salvará as imagens reconstruídas em `images/` e gerará um `relatorio.json` ao final.

## Mapeamento de Sinais e Matrizes

| Sinal (Arquivo) | Matriz H Utilizada |
| --- | --- |
| G-1.csv | H-1.csv |
| G-2.csv | H-1.csv |
| A-60x60-1.csv | H-1.csv |
| g-30x30-1.csv | H-2.csv |
| g-30x30-2.csv | H-2.csv |
| A-30x30-1.csv | H-2.csv |

## Diretórios e Arquivos de Saída

* `images/` — Contém as imagens médicas reconstruídas salvas em formato gráfico `.png`.
* `relatorio.json` — Registro estruturado de cada processo de reconstrução contendo: identificação do sinal, número de iterações executadas, tempo gasto e matriz H utilizada.
* `metrics_py.csv` / `metrics_cpp.csv` — Arquivos de logs de métricas operacionais e de consumo gerados dinamicamente pelos servidores.

```

```