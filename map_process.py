import healpy as hp
import numpy as np
import pymaster as nmt
from scipy.optimize import curve_fit


def coadd_maps(map_list, out_filename, unit_conv=1.0e-3):

    """

    Function for generating co-added maps.

    Inputs
    ------
    :map_list: List of the maps you want to co-add - list.
    :out_filename: Output filename for the coadded map - str.
    :unit_conv: Conversion factor to apply so output map is in K - str.

    Returns
    -------
    :coadd: The co-added map - numpy.ndarray.

    """

    single_I = np.array([hp.read_map(m, field=0) for m in map_list]) * unit_conv
    single_Q = np.array([hp.read_map(m, field=1) for m in map_list]) * unit_conv
    single_U = np.array([hp.read_map(m, field=2) for m in map_list]) * unit_conv
    single_hits = np.array([hp.read_map(m, field=3) for m in map_list])

    # Note, something odd seems to happen with WMAP I. Q and U are fine though.
    coadd_hits = np.sum(single_hits, axis=0)
    coadd_I = np.sum(single_I * single_hits, axis=0) / coadd_hits
    coadd_Q = np.sum(single_Q * single_hits, axis=0) / coadd_hits
    coadd_U = np.sum(single_U * single_hits, axis=0) / coadd_hits
    coadd = np.array([coadd_I, coadd_Q, coadd_U, coadd_hits])
    
    hp.write_map(out_filename, coadd, overwrite=True)

    return coadd


def nl_model(ell, A, B):

    return (A / (ell)) + B


class Map_Split_Noise(object):

    def	__init__(self, map_file1, map_file2, scaling, diff_filename=None, mask_filename=None, apo_scale=None,
                 apo_type=None, lmax=None, nlb=None, nl_filename=None, lmin=None, AI_init=None, BI_init=None,
                 alphaI_init=None, AE_init=None, BE_init=None, alphaE_init=None, AB_init=None, BB_init=None,
                 alphaB_init=None, params_filename=None):
    
        """
    
        Class for estimating noise power spectra from data splits.

        Attributes
        ----------
        :map_file1: First map filename - str.
        :map_file2: Second map filename - str.
        :scaling: Scaling to apply to difference map (not including Nobs scaling) - float.
        :diff_filename: Output filename for scaled difference map - str.
        :mask_filename: Mask filename for power spectrum estimation - str.
        :apo_scale: Apodization scale for power spectrum estimation - float.
        :apo_type: Apodization type for power spectrum estimation - str.
        :lmax: Maximum multipole to calculate power spectra to - int.
        :nlb: Bin width for power spectra - int.
        :nl_filename: Output filename for the noise power spectra - str.
        :lmin: Minimum multipole to inlude in model fit - int.
        :AI_init: Initial A parameter for II - float.
        :BI_init: Initial B parameter for II - float.
        :alphaI_init: Initial power law index for II - float.
        :AE_init: Initial A parameter for EE - float.
        :BE_init: Initial B parameter for EE - float.
        :alphaE_init: Initial power law index for EE - float.
        :AB_init: Initial A parameter for BB - float.
        :BB_init: Initial B parameter for BB - float.
        :alphaB_init: Initial power law index for BB - float.
        :params_filename: Output filename for fitted nl model parameters - str.

        """
    
        self.map_file1 = map_file1
        self.map_file2 = map_file2
        self.scaling = scaling
        self.diff_filename = diff_filename
        self.mask_filename = mask_filename
        self.apo_scale = apo_scale
        self.apo_type = apo_type
        self.lmax = lmax
        self.nlb = nlb
        self.nl_filename = nl_filename
        self.lmin = lmin
        self.AI_init = AI_init
        self.BI_init = BI_init
        self.alphaI_init = alphaI_init
        self.AE_init = AE_init
        self.BE_init = BE_init
        self.alphaE_init = alphaE_init
        self.AB_init = AB_init
        self.BB_init = BB_init
        self.alphaB_init = alphaB_init
        self.params_filename = params_filename
    
    def diff_maps(self):

        print('Estimating difference maps')
        m1 = hp.read_map(self.map_file1, field=None)
        m2 = hp.read_map(self.map_file2, field=None)
        
        diff_I = m1[0] - m2[0]
        diff_Q = m1[1] - m2[1]
        diff_U = m1[2] - m2[2]
        Nobs = m1[3] + m2[3]
        
        diff_I = diff_I * self.scaling * np.sqrt(Nobs)
        diff_Q = diff_Q * self.scaling * np.sqrt(Nobs)
        diff_U = diff_U * self.scaling * np.sqrt(Nobs)
        diff_map = np.array([diff_I, diff_Q, diff_U, m1[3], m2[3], Nobs])

        if self.diff_filename is not None:
            hp.write_map(self.diff_filename, diff_map, overwrite=True)
        
        return diff_map

    def noise_power(self):

        diff_map = self.diff_maps()
        print('Estimating noise power spectra for difference maps')
        nside = hp.get_nside(diff_map)
        mask = hp.read_map(self.mask_filename)
        mapI = diff_map[0]
        mapQ = diff_map[1]
        mapU = diff_map[2]
        mapI[mask == 0] = 0
        mapQ[mask == 0] = 0
        mapU[mask == 0] = 0
        
        mask_apo = nmt.mask_apodization(mask, aposize=self.apo_scale, apotype=self.apo_type)
        b = nmt.NmtBin(nside, nlb=self.nlb, lmax=self.lmax)
        leff = b.get_effective_ells()

        f0 = nmt.NmtField(mask, [mapI])
        f2 = nmt.NmtField(mask, [mapQ, mapU], purify_e=False, purify_b=False)
        
        nl0 = nmt.compute_full_master(f0, f0, b)
        nl2 = nmt.compute_full_master(f2, f2, b)
        
        nlI = nl0[0]
        nlE = nl2[0]
        nlB = nl2[3]
        
        if self.nl_filename is not None:
            np.savetxt(self.nl_filename, np.c_[leff, nlI, nlE, nlB])
            
        return leff, nlI, nlE, nlB

    def nl_model_fit(self):

        leff, nlI, nlE, nlB = self.noise_power()
        print('Fitting fiducial model to noise power spectra')
        nlI = nlI[leff >= self.lmin]
        nlE = nlE[leff >= self.lmin]
        nlB = nlB[leff >= self.lmin]
        leff = leff[leff >= self.lmin]
        
        poptI, pcovI = curve_fit(nl_model, leff, nlI, p0=[self.AI_init, self.BI_init])
        poptE, pcovE = curve_fit(nl_model, leff, nlE, p0=[self.AE_init, self.BE_init])
        poptB, pcovB = curve_fit(nl_model, leff, nlB, p0=[self.AB_init, self.BB_init])

        if self.params_filename is not None:
            np.savetxt(self.params_filename, np.c_[poptI[0], poptI[1], poptE[0], poptE[1],
                                                   poptB[0], poptB[1]])
            
        return poptI, poptE, poptB
        
        
