import json
import random

CONFIG_FILE = "config.json"
CLIENTS = ["client_1", "client_2", "client_3"]

SINAIS_MAP = {
    'G-1.csv':       'H-1.csv',
    'G-2.csv':       'H-1.csv',
    'A-60x60-1.csv': 'H-1.csv',
    'g-30x30-1.csv': 'H-2.csv',
    'g-30x30-2.csv': 'H-2.csv',
    'A-30x30-1.csv': 'H-2.csv',
}

sinais_lista = list(SINAIS_MAP.keys())
config_data = {}

for client in CLIENTS:
    total_requests = random.randint(30, 50)
    client_tasks = []
    
    for i in range(total_requests):
        nome_g = random.choice(sinais_lista)
        nome_h = SINAIS_MAP[nome_g]
        delay = round(random.uniform(0.1, 0.3), 2)
        
        # Randomize the amplification factor for the gain formula
        fator = round(random.uniform(1.0, 3.0), 4)
        
        client_tasks.append({
            "nome_g": nome_g,
            "nome_h": nome_h,
            "delay": delay,
            "fator": fator  # Added parameter
        })
        
    config_data[client] = client_tasks

with open(CONFIG_FILE, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"Successfully generated {CONFIG_FILE} with gain parameters!")