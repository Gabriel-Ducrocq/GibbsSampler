import numpy as np
import healpy as hp
import qcinv
import utils
import matplotlib.pyplot as plt
from classy import Class
import config
cosmo = Class()


map = hp.ud_grade(hp.read_map("wmap_temperature_kq85_analysis_mask_r9_9yr_v5(1).fits",0), 512)
print(map)


