import yaml
import os

def justify(mode, text, n=15):
    if mode == "l":
        return text.ljust(n)
    if mode == "c":
        return text.center(n)
    if mode == "r":
        return text.rjust(n)

def log_config(config):
    for key, value in config.items():
        key = justify("l", str(key))
        value = justify("c", str(value))
        print(key+":"+value)

def read_yaml(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist.")
    config = yaml.safe_load(open(config_path, "r"))
    log_config(config)
    return config
