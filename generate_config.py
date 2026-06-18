import json
import random

# Configuration settings
CONFIG_FILE = "config.json"
CLIENTS = ["client_1", "client_2", "client_3"]

# Available signals and their corresponding H matrices
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
    # Randomize the total number of requests for this client (between 30 and 50)
    total_requests = random.randint(30, 50)
    client_tasks = []
    
    for i in range(total_requests):
        # Randomize the signal choice
        nome_g = random.choice(sinais_lista)
        nome_h = SINAIS_MAP[nome_g]
        
        # Randomize a unique delay value for this specific request
        delay = round(random.uniform(0.5, 3.0), 2)
        
        client_tasks.append({
            "nome_g": nome_g,
            "nome_h": nome_h,
            "delay": delay
        })
        
    config_data[client] = client_tasks

# Save to config.json
with open(CONFIG_FILE, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"Successfully generated {CONFIG_FILE}!")
for client, tasks in config_data.items():
    print(f"  → {client}: {len(tasks)} requests configured.")