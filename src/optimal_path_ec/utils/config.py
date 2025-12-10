import yaml

def readYaml(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data