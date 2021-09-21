import os, shutil
import importlib


# Base paths
base_path = '/home/ken/nelf_data/' # TODO Change to your dataset path

# Import config
config_path = f'{os.path.dirname(os.path.abspath(__file__))}/config_{os.sys.argv[1]}.py'
spec = importlib.util.spec_from_file_location('config', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


# Add attribute to config; Create path is needed
names = [x for x in config.__dict__ if not x.startswith("_")]
for name in names:
    if name.endswith('path'):
        path = f'{base_path}/{getattr(config, name)}'
        globals()[name] = path
        os.makedirs(path, exist_ok=True)
    else:
        globals()[name] = getattr(config, name)

shutil.copyfile(config_path, f'{globals()["exp_path"]}/config.py')