# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

from .ecc import ECC
from .orb import ORB
from .sift import SIFT
from .sof import SOF


def get_cmc_method(cmc_method):
    if cmc_method == 'ecc':
        return ECC
    elif cmc_method == 'orb':
        return ORB
    elif cmc_method == 'sof':
        return SOF
    elif cmc_method == 'sift':
        return SIFT
    else:
        return None
