import requests
import json
import numpy as np
import os


url = "https://api.runpod.ai/v2/8uqz9rklt6dkt6/runsync"
url = "https://api.runpod.ai/v2/rifqrrc90o57rk/runsync"
headers = {
    "accept": "application/json",
    "authorization": os.getenv("RUNPOD_API_KEY"),
    "content-type": "application/json"
}

for i in range(100):
    matrix = np.random.rand(8,8) * np.random.rand() * 897
    summ = np.sum(matrix)
    matrix = matrix.tolist()
    data = {
        "input": {
            "matrix": f"{matrix}"
        }
    }


    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        print(summ, response.json()['output'])
    except:
        print('error: ', response.json())
