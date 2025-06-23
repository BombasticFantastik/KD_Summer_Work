import yaml
from Dataset import BootDataset
import os

option_path=fr'D:/Code/KD_Summer_Work/config.yml'
with open(option_path,'r') as file_option:
    print(file_option)
    option=yaml.safe_load(file_option)

#print(option)
dataset=BootDataset(option['data_path'])


    