import numpy as np
import vaex
import healpy as hp
import matplotlib.pyplot as plt
from complete_mask import mask_gen
from complete_mask import sixd_prob_funct_aio
from complete_mask import read_mask
from complete_mask import plot_mask_hpx

if __name__ == "__main__":
    in_path = "../../small_gaia.hdf5"
    out_path = "out_mask.fits"
    mask_gen(in_path, out_path)
