from loader.caltech256 import get_caltech
from config import cfg
def get_loader():
    pair = {
        'caltech256' : get_caltech
    }

    return pair[cfg.data.type]()
