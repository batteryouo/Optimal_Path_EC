import yaml
import json

def readYaml(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data



def readJson(path):
    # Open the file in read mode
    with open(path, 'r', encoding='utf-8') as f:
        # Parse the JSON file into a Python dictionary or list
        data = json.load(f)
    return data