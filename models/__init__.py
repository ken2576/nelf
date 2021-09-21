from .nelf import Nelf
from .nelf_ft import NelfFt
from .nelf_direct import NelfDirect
from .nelf_ft_direct import NelfFtDirect
from .ibr import IBR
from .sipr import SIPR

def get_model(name):
    if name == 'nelf':
        return Nelf
    elif name == 'nelf_direct':
        return NelfDirect
    elif name == 'nelf_ft':
        return NelfFt
    elif name == 'nelf_ft_direct':
        return NelfFtDirect
    elif name == 'ibr':
        return IBR
    elif name == 'sipr':
        return SIPR
    else:
        raise NotImplementedError