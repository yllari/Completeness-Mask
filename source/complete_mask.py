import vaex
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import time
from astropy.io import fits


def mask_gen(infile, outfile):
    '''Completeness 6D sample mask generation for Gaia DR3 database.

    The filename with gaia db is introduced and a fits file with
    a probability matrix is provided in the color/magnitude/healpx
    space.

    The quotient 5D/6D stars in the 3D space color/magntiude/healpx
    is computed and returned as a 400x100x13 matrix (by default)
    In the last position (400x100x-1) of the matrix the
    mask without spatial (all sky) position dependance is saved.


    :param infile: filename for the input, gaia database
    :param outfile: filename for the output, fits extension
    :return: None
    '''

    t_0 = time.time()
    # -+-+-+-+-+-+-+-+-+-PREPROCESSING--+-+-+-+-+-+-+-+-+
    columns_use = [
        'l',
        'b',
        'parallax',
        'radial_velocity',
        'phot_g_mean_mag',
        'phot_bp_mean_mag',
        'phot_rp_mean_mag',
    ]

    gaia_df = vaex.open(infile)
    print("Total of stars: ", gaia_df.count())


    # Creating new columns (virtual)

    gaia_df = gaia_df[columns_use]
    gaia_df['distance'] = 1.0/gaia_df['parallax']
    bp_rp = gaia_df['phot_bp_mean_mag'] - gaia_df['phot_rp_mean_mag']
    gaia_df['bp_rp'] = bp_rp

    # boolean expressions for different parameter cuts
    # - distance cut: d \in [0, 3.5] kpc for distance = 1/parallax
    # - radial velocity filter: belongs to 6D if rvs is known
    # Distance filter works as nan check as well
    dist_cut = (gaia_df['distance'] > 0.0) & (gaia_df['distance'] < 3.501)
    rvs = ~gaia_df['radial_velocity'].ismissing()

    ##-+-+-+--+-+-Uncomment if any other spatial filter is needed-+-+-+-+-+-
    ## Adding geometric columns and colour, magnitude, etc etc...

    #gaia.add_variable('Z0', 0.0196)   ;  Z0 = 0.0196    # kpc from a
    #conversation somewhere on the email, with references and everything.

    ## Adding Cartesian coordinates (tilde X, Y, Z), because they are
    #centred at the Sun
    #gaia.add_virtual_columns_spherical_to_cartesian(alpha='l', delta='b',
    #distance='distance', xname='tildex', yname='tildey', zname='tildez')

    ## Adding Cylindrical (polar) coordinates, with the Sun at the centre.
    #--> tilde Rho, tild Phi
    #gaia.add_virtual_columns_cartesian_to_polar(x='(tildex)',
    #y='(tildey)', radius_out='tildeRho', azimuth_out='tildePhi',
    #radians=True)

    ## Adding Cartesian coordinates centred at the galactic PLANE. --> xyz
    #gaia.add_column('x', gaia.evaluate('tildex'))
    #gaia.add_column('y', gaia.evaluate('tildey'))
    #gaia.add_column('z', gaia.evaluate('tildez+Z0'))



    # gaia_df is just gaia_5d (all data)
    gaia_df = gaia_df[dist_cut].extract()
    print("Stars meeting cut criteria", gaia_df.count())

    # -+-+-+-+-+-+-+-+-+-COMPLETENESS-+-+-+-+-+-+-+-+-+
    # Healpix determination and classification on the
    # 3D grid color-magnitude-healpixel
    #
    # Parameters
    # - nside: determines number of healpixels, nside = 1 -> 12 healpx
    #          MUST be a power of 2
    # - coord_lims/bins: histogram parameters
    # coord lims are stablished via dataset exploration (approx max-min)
    nside = 1
    # Number of healpxs for given nside (- 1 for lims)
    n_hpx = hp.nside2npix(nside) - 1
    coords = ['bp_rp', 'phot_g_mean_mag', 'hpix']
    coord_lims = [[-7.5, 10.5], [1.5, 23.0], [0 - 0.5, n_hpx + 0.5]]
    # -0.5, +0.5 correctly centers bins in the integer values
    # healpix starts from 0
    coord_bins = [100, 400, int(n_hpx + 1)]


    bb, ll = gaia_df.evaluate('b'), gaia_df.evaluate('l')
    ipix = hp.ang2pix(
        nside,
        np.pi*(90. - bb)/180.,
        np.pi*ll/180., nest=False
    )
    print("Healpix calculated")
    gaia_df.add_column("hpix", ipix)

    # 6D selection
    gaia6D_df = gaia_df[rvs].extract()

    counts_5D = np.array(
        gaia_df.count(
            binby=coords,
            limits=coord_lims,
            shape=coord_bins)
    )

    counts_6D = np.array(
        gaia6D_df.count(
            binby=coords,
            limits=coord_lims,
            shape=coord_bins)
    )
    prob = counts_6D/counts_5D

    # Generating a mask without spatial dependance (last axis)
    all_sky_5D = np.sum(counts_5D, axis=-1)
    all_sky_6D = np.sum(counts_6D, axis=-1)
    all_sky_prob = all_sky_6D/all_sky_5D
    # Changing shape to concatenate
    all_sky_prob = np.expand_dims(all_sky_prob, -1)
    prob = np.concatenate([prob, all_sky_prob], axis=-1)

    print("Mask Completed! Elasped time (min): ", (time.time() - t_0)/60 )

    # Save mask and engrave all info on header
    primary_HDU = fits.PrimaryHDU(data=prob)
    hdr = primary_HDU.header
    hdr['NSIDE'] = nside
    hdr.comments['NSIDE'] = 'healpx division'
    hdr['LCOLOR1'] = coord_lims[0][0]
    hdr.comments['LCOLOR1'] = 'inferior limit of color'
    hdr['LCOLOR2'] = coord_lims[0][1]
    hdr.comments['LCOLOR2'] = 'superior limit of color'
    hdr['LMAG1'] = coord_lims[1][0]
    hdr.comments['LMAG1'] = 'inferior limit of apparent mag'
    hdr['LMAG2'] = coord_lims[1][1]
    hdr.comments['LMAG2'] = 'superior limit of apparent mag'
    hdr['BCOLOR'] = coord_bins[0]
    hdr.comments['BCOLOR'] = 'number of bins on color'
    hdr['BMAG'] = coord_bins[1]
    hdr.comments['BMAG'] = 'number of bins on apparent mag'

    primary_HDU.writeto(outfile, overwrite='True')

    return None


def read_mask(
        infile,
        showheader: bool=False
    ):
    '''Read completeness mask fits file and return prob matrix, params.

    :param infile: string or 'os' direction to fits file
    :return: probability matrix (ndarray),
             nside (int), limits (list) and bins (list)
    '''

    nside = 0
    lims = []
    bins = []
    with fits.open(infile) as fit_file:
        header = fit_file[0].header
        if showheader:
            print(repr(header))
        nside = header['NSIDE']
        lims.append([header['LCOLOR1'], header['LCOLOR2']])
        lims.append([header['LMAG1'], header['LMAG2']])
        bins.append(header['BCOLOR'])
        bins.append(header['BMAG'])
        prob_matrix = fit_file[0].data

    print("Read complete!")

    return prob_matrix, nside, lims, bins

def plot_mask_hpx(prob, extent, healpx):
    '''Plot probability matrix for given healpx as a PNG.
    Image is exported to "#healpx.png"

    :param prob: probability matrix as given by read_mask
    :param extent: limits of plot
    :param healpx: healpx to plot
    :return: None
    '''

    extent = np.array(extent).flatten()
    fig, ax = plt.subplots()
    im = ax.imshow(
        prob[:, :, healpx].T,
        extent=extent,
        origin='lower',
        aspect='auto',
        interpolation='none',
    )
    plt.gca().invert_yaxis()
    fig.colorbar(im, label="Probability")
    ax.set_xlabel(r"$G_{BP}-G_{RP}$")
    ax.set_ylabel(r"$G$")
    fig.savefig(f"#{healpx}.png", dpi=300)
    plt.close()
    return None


def sixd_prob_funct(
        ll,
        bb,
        g,
        bp_rp,
        mask_file):
    '''Given a position in color-magnitude-spatial coord a completeness
    probability is calculated (6D/5D).

    Values outside of range, both because of lack of information on
    mask or lims, are returned as zero.

    :param ll: galactic longitudes
    :param bb: galactic latitudes
    :param g: apparent magnitude G (as given by Gaia DR3 phot_g_mean_mag)
    :param bp_rp: color difference G_BP-G_RP
                  (phot_bp_mean_mag - phot_rp_mean_mag)
    :param mask_file: path to mask as generated by 'mask_gen'
    :return: np array of probabilities with the same size as input.
    '''

    bb = np.asarray(bb)
    ll = np.asarray(ll)
    g = np.asarray(g)
    bp_rp = np.asarray(bp_rp)
    prob_matrix, nside, lims, bins = read_mask(mask_file)
    # lims is a list containing [[bp_rp min, bp_rp max], [g min, g max]]

    # Defining behaviour for nan values
    # (those with no info in Gaia DR3), can be changed
    # On the -1 position, the all-sky mask
    prob_matrix[:, :, -1][np.isnan(prob_matrix[:, :, -1])] = 0
    # indx discarding out of limits coordinates. Strict > or not >=
    # does not really matter
    in_indx = (
        (bp_rp > lims[0][0]) & (bp_rp < lims[0][1])
        & (g > lims[1][0]) & (g < lims[1][1])
    )

    color_step = (lims[0][1] - lims[0][0])/bins[0]
    g_step = (lims[1][1] - lims[1][0])/bins[1]

    color_indx = (bp_rp - lims[0][0])/color_step
    g_indx = (g - lims[1][0])/g_step

    heal_indx = (
        hp.ang2pix(
            nside, (90. - bb)*np.pi/180.,
            ll*np.pi/180.,
            nest=False
        ) #Start from 0
    )

    # Outside of range values are assigned zero prob
    sixd_prob = np.zeros(ll.shape)

    color_in_indx = color_indx[in_indx].astype(int)
    g_in_indx = g_indx[in_indx].astype(int)
    heal_in_indx = heal_indx[in_indx].astype(int)

    sixd_prob[in_indx] = prob_matrix[
        color_in_indx,
        g_in_indx,
        heal_in_indx
    ]

    # Changing nan values for values with the mask
    # computed with all sky (-1 position of array)
    prob_isnan = np.isnan(sixd_prob[in_indx])
    sixd_prob[in_indx & np.isnan(sixd_prob)] = prob_matrix[
        color_in_indx[prob_isnan],
        g_in_indx[prob_isnan],
        -1
    ]

    return sixd_prob

# sixd_prob_funct_aio is the function to be copied if
# nothing else is going to be imported from this file (reads file and
# calculates probability)

def sixd_prob_funct_aio(
        ll,
        bb,
        g,
        bp_rp,
        mask_file):
    '''Given a position in color-magnitude-spatial coord a completeness
    probability is calculated (6D/5D).

    Values outside of range, either outside of range
    or because of lack of information on mask (nan),
    are returned as zero.

    :param ll: galactic longitudes
    :param bb: galactic latitudes
    :param g: apparent magnitude G (as given by Gaia DR3 phot_g_mean_mag)
    :param bp_rp: color difference G_BP-G_RP
                  (phot_bp_mean_mag - phot_rp_mean_mag)
    :param mask_file: path to mask as generated by 'mask_gen' function
    :return: np array of probabilities with the same size as input.
    '''

    bb = np.asarray(bb)
    ll = np.asarray(ll)
    g = np.asarray(g)
    bp_rp = np.asarray(bp_rp)


    # Reading first
    nside = 0
    lims = []
    bins = []
    with fits.open(mask_file) as fit_file:
        header = fit_file[0].header
        nside = header['NSIDE']
        lims.append([header['LCOLOR1'], header['LCOLOR2']])
        lims.append([header['LMAG1'], header['LMAG2']])
        bins.append(header['BCOLOR'])
        bins.append(header['BMAG'])
        prob_matrix = fit_file[0].data
    # lims is a list containing [[bp_rp min, bp_rp max], [g min, g max]]

    # Defining behaviour for nan values
    # (those with no info in Gaia DR3), can be changed
    # On the -1 position, the all-sky mask
    prob_matrix[:, :, -1][np.isnan(prob_matrix[:, :, -1])] = 0
    # indx discarding out of limits coordinates. Strict > or not >=
    # does not really matter
    in_indx = (
        (bp_rp > lims[0][0]) & (bp_rp < lims[0][1])
        & (g > lims[1][0]) & (g < lims[1][1])
    )

    color_step = (lims[0][1] - lims[0][0])/bins[0]
    g_step = (lims[1][1] - lims[1][0])/bins[1]

    color_indx = (bp_rp - lims[0][0])/color_step
    g_indx = (g - lims[1][0])/g_step

    heal_indx = (
        hp.ang2pix(
            nside, (90. - bb)*np.pi/180.,
            ll*np.pi/180.,
            nest=False
        ) #Start from 0
    )

    # Outside of range values are assigned zero prob
    sixd_prob = np.zeros(ll.shape)

    color_in_indx = color_indx[in_indx].astype(int)
    g_in_indx = g_indx[in_indx].astype(int)
    heal_in_indx = heal_indx[in_indx].astype(int)

    sixd_prob[in_indx] = prob_matrix[
        color_in_indx,
        g_in_indx,
        heal_in_indx
    ]

    # Changing nan values for values with the mask
    # computed with all sky (-1 position of array)
    prob_isnan = np.isnan(sixd_prob[in_indx])
    sixd_prob[in_indx & np.isnan(sixd_prob)] = prob_matrix[
        color_in_indx[prob_isnan],
        g_in_indx[prob_isnan],
        -1
    ]

    return sixd_prob
