import torch
import yaml
from Dataset import BootDataset


option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)