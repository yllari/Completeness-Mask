import numpy as np
import vaex
import healpy as hp
import matplotlib.pyplot as plt
from complete_mask import mask_gen
from complete_mask import sixd_prob_funct_aio
from complete_mask import sixd_prob_funct
from complete_mask import read_mask
from complete_mask import plot_mask_hpx

if __name__ == "__main__":
    in_path = "../../small_gaia.hdf5"
    out_path = "out_mask.fits"
    mask_gen(in_path, out_path)

    out_path = "complete_mask.fits"
    prob, nside, lims, bins = read_mask(out_path)

    for i in range(0, 13):
        plot_mask_hpx(prob, lims, i)

    div = bins[0]*2
    bb = np.linspace(70,76, div)
    ll = np.linspace(340,350, div)

    bb = np.linspace(0,10, div)
    ll = np.linspace(180,182, div)
    bb, ll = np.meshgrid(bb,ll)
    bp_rp = np.linspace(-7.5,10.5,div)
    g = np.linspace(1.5,23.0,div)
    bp_rp, g = np.meshgrid(bp_rp, g)

    probs = sixd_prob_funct_aio(
        ll,
        bb,
        g,
        bp_rp,
        out_path)
    hpx = hp.ang2pix(
        nside, (90. - bb)*np.pi/180.,
        ll*np.pi/180.,
        nest=False
    )  # Start from 0
    print(np.unique(hpx))
    plt.scatter(bp_rp, g, c = probs, s=5.5)
    plt.gca().invert_yaxis()
    plt.savefig("example.png")

