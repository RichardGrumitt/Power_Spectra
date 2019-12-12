import numpy as np
import healpy as hp
import configparser
import argparse
import pymaster as nmt


"""

This code estimates E and B power spectra using the NaMaster library. This performs pure-E and pure-B power spectrum
estimation, effectively eliminating issues arising from E to B leakage when dealing with cut-skies. 

"""

########################################################################################################################

print('Reading in basic parameters ...')

parser = argparse.ArgumentParser(description='Code for estimating pseudo-Cl given two sky maps with uncorrelated '
                                             'noise.')
parser.add_argument('param_file', help='Main parameters file.')

Config = configparser.ConfigParser()
Config.read(parser.parse_args().param_file)

input_map1 = Config.get('Cl Estimation Params', 'input_map1')
input_map2 = Config.get('Cl Estimation Params', 'input_map2')
input_cov1 = Config.get('Cl Estimation Params', 'input_cov1')
input_cov2 = Config.get('Cl Estimation Params', 'input_cov1')
use_noise_model1 = Config.getboolean('Cl Estimation Params', 'use_noise_model1')
use_noise_model2 = Config.getboolean('Cl Estimation Params', 'use_noise_model2')
nl_param_file1 = Config.get('Cl Estimation Params', 'nl_param_file1')
nl_param_file2 = Config.get('Cl Estimation Params', 'nl_param_file2')
hits_map1 = Config.get('Cl Estimation Params', 'hits_map1')
hits_map2 = Config.get('Cl Estimation Params', 'hits_map2')
diffuse_map1 = Config.get('Cl Estimation Params', 'diffuse_map1')
diffuse_map2 = Config.get('Cl Estimation Params', 'diffuse_map2')

mask_file = Config.get('Cl Estimation Params', 'mask_file')
beam1_file = Config.get('Cl Estimation Params', 'beam1_file')
beam2_file = Config.get('Cl Estimation Params', 'beam2_file')
fwhm1 = Config.getfloat('Cl Estimation Params', 'fwhm1')
fwhm2 =	Config.getfloat('Cl Estimation Params', 'fwhm2')

nside = Config.getint('Cl Estimation Params', 'nside')
lmax = Config.getint('Cl Estimation Params', 'lmax')
custom_bpws = Config.getboolean('Cl Estimation Params', 'custom_bpws')
bpws_file = Config.get('Cl Estimation Params', 'bpws_file')
nlb = Config.getint('Cl Estimation Params', 'nlb')
apodize_scale = Config.getfloat('Cl Estimation Params', 'apodize_scale')
apodize_type = Config.get('Cl Estimation Params', 'apodize_type')
num_monte_carlo = Config.getint('Cl Estimation Params', 'num_monte_carlo')

leff_out = Config.get('Cl Estimation Params', 'leff_out')
cl_mc_mean_out = Config.get('Cl Estimation Params', 'cl_mc_mean_out')
cl_mc_std_out = Config.get('Cl Estimation Params', 'cl_mc_std_out')
cl_mc_cov_EE_EE_out = Config.get('Cl Estimation Params', 'cl_mc_cov_EE_EE_out')
cl_mc_cov_BB_BB_out = Config.get('Cl Estimation Params', 'cl_mc_cov_BB_BB_out')
cl_mc_cov_EB_EB_out = Config.get('Cl Estimation Params', 'cl_mc_cov_EB_EB_out')

########################################################################################################################

print('Reading in the Q/U maps, associated covariance maps, mask, and beam file (spherical harmonic transform) ...')

Qmap1, Umap1 = hp.read_map(input_map1, field=(1,2)) * 1e6 # Converting input maps from K to uK.
Qmap2, Umap2 = hp.read_map(input_map2, field=(1,2)) * 1e6

Qcov1, Ucov1 = hp.read_map(input_cov1, field=(1,2)) * 1e12 # Converting cov maps from K^2 to uK^2.
Qcov2, Ucov2 = hp.read_map(input_cov2, field=(1,2)) * 1e12

mask = hp.read_map(mask_file)
beam1 = np.loadtxt(beam1_file)
beam2 = np.loadtxt(beam2_file)

Qmap1[mask == 0] = 0
Umap1[mask == 0] = 0
Qmap2[mask == 0] = 0
Umap2[mask == 0] = 0

########################################################################################################################

print('Estimating pure E and B pseudo-Cls using NaMaster ...')

# Apodize the mask. Key parameters are the apodization scale and type. See the NaMaster docs for details.
ell = np.arange(3 * nside, dtype='int32')
mask_apo = nmt.mask_apodization(mask, aposize=apodize_scale, apotype=apodize_type)

# Generate the ell binning scheme.
if custom_bpws:
    bpws = np.loadtxt('bpws_file', usecols=0)
    weights = np.loadtxt('bpws_file', usecols=1)
    b = nmt.NmtBin(nside, bpws=bpws, ells=ell, weights=weights)
    leff = b.get_effective_ells()
elif not custom_bpws:
    b = nmt.NmtBin(nside, nlb=nlb, lmax=lmax)
    leff = b.get_effective_ells()
    nb = nmt.NmtBin(nside, nlb=1)

# This function returns NaMaster field objects with our apodized mask and input Q/U maps. (Pure B and Pure E fields)
def get_pureB_fields(qmap1, umap1, qmap2, umap2):
    fnum1 = nmt.NmtField(mask_apo, [qmap1, umap1], purify_e=False, purify_b=True, beam=beam1)
    fnum2 = nmt.NmtField(mask_apo, [qmap2, umap2], purify_e=False, purify_b=True, beam=beam2)
    return fnum1, fnum2

def get_pureE_fields(qmap1, umap1, qmap2, umap2):
    fnum1 = nmt.NmtField(mask_apo, [qmap1, umap1], purify_e=True, purify_b=False, beam=beam1)
    fnum2 = nmt.NmtField(mask_apo, [qmap2, umap2], purify_e=True, purify_b=False, beam=beam2)
    return fnum1, fnum2

def get_pureEB_fields(qmap1, umap1, qmap2, umap2):
    fnum1 = nmt.NmtField(mask_apo, [qmap1, umap1], purify_e=True, purify_b=True, beam=beam1)
    fnum2 = nmt.NmtField(mask_apo, [qmap2, umap2], purify_e=True, purify_b=True, beam=beam2)
    return fnum1, fnum2

def get_impure_fields(qmap1, umap1, qmap2, umap2):
    fnum1 = nmt.NmtField(mask_apo, [qmap1, umap1], purify_e=False, purify_b=False, beam=beam1)
    fnum2 = nmt.NmtField(mask_apo, [qmap2, umap2], purify_e=False, purify_b=False, beam=beam2)
    return fnum1, fnum2

# Initialise the workspace for the fields (you only have to do this once).
field_pureB_num1, field_pureB_num2 = get_pureB_fields(Qmap1, Umap1, Qmap2, Umap2)
w_pureB = nmt.NmtWorkspace()
w_pureB.compute_coupling_matrix(field_pureB_num1, field_pureB_num2, b)
field_pureE_num1, field_pureE_num2 = get_pureE_fields(Qmap1, Umap1, Qmap2, Umap2)
w_pureE = nmt.NmtWorkspace()
w_pureE.compute_coupling_matrix(field_pureE_num1, field_pureE_num2, b)
field_pureEB_num1, field_pureEB_num2 = get_pureEB_fields(Qmap1, Umap1, Qmap2, Umap2)
w_pureEB = nmt.NmtWorkspace()
w_pureEB.compute_coupling_matrix(field_pureE_num1, field_pureE_num2, b)
field_impure_num1, field_impure_num2 = get_impure_fields(Qmap1, Umap1, Qmap2, Umap2)
w_impure = nmt.NmtWorkspace()
w_impure.compute_coupling_matrix(field_impure_num1, field_impure_num2, b)

# This function does the actual power spectrum calculation.
def compute_master(f_a, f_b, wsp, noise_bias=None):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    if noise_bias is None:
        cl_decoupled = wsp.decouple_cell(cl_coupled)
    elif noise_bias is not None:
        cl_decoupled = wsp.decouple_cell(cl_coupled, cl_noise=noise_bias)
    return cl_decoupled

# Now we're ready to run over our Monte Carlo simulations.
# Here we estimate the mean Cl, the standard deviation on these, and the full covariance matrix between bandpowers.
snl_pureB_power = []
snl_pureE_power = []
snl_pureEB_power = []
snl_impure_power = []

# Model noise power spectra - only used if use_noise_model{1/2} is True.
nl_params1 = np.loadtxt(nl_param_file1)
nl_params2 = np.loadtxt(nl_param_file2)
nlE1 = nl_params1[2] / ell + nl_params1[3]
nlB1 = nl_params1[4] / ell + nl_params1[5]
nlE2 = nl_params2[2] / ell + nl_params2[3]
nlB2 = nl_params2[4] / ell + nl_params2[5]
nlE1[0:2] = 0
nlB1[0:2] = 0
nlE2[0:2] = 0
nlB2[0:2] = 0
nl1 = (nlE1, nlE1, nlB1, np.zeros(len(nlB1)))
nl2 = (nlE2, nlE2, nlB2, np.zeros(len(nlB2)))
hits1 = hp.read_map(hits_map1)
hits2 = hp.read_map(hits_map2)

# Read in diffuse signal simulations - used in Monte Carlo covariance estimation.
diffuse_signal_Q1, diffuse_signal_U1 = hp.read_map(diffuse_map1, field=(1,2))
diffuse_signal_Q2, diffuse_signal_U2 = hp.read_map(diffuse_map2, field=(1,2))

field_pureB_num1, field_pureB_num2 = get_pureB_fields(Qmap1, Umap1, Qmap2, Umap2)
cl_pureB_mean = compute_master(field_pureB_num1, field_pureB_num2, w_pureB)
field_pureE_num1, field_pureE_num2 = get_pureE_fields(Qmap1, Umap1, Qmap2, Umap2)
cl_pureE_mean = compute_master(field_pureE_num1, field_pureE_num2, w_pureE)
field_pureEB_num1, field_pureEB_num2 = get_pureEB_fields(Qmap1, Umap1, Qmap2, Umap2)
cl_pureEB_mean = compute_master(field_pureEB_num1, field_pureEB_num2, w_pureEB)
field_impure_num1, field_impure_num2 = get_impure_fields(Qmap1, Umap1, Qmap2, Umap2)
cl_impure_mean = compute_master(field_impure_num1, field_impure_num2, w_impure)

fid_r0_lensed_filename = './fiducial_spectra/cmb_fiducial_r0_lensed.dat'
fid_r0p01_unlensed_filename = './fiducial_spectra/cmb_fiducial_r0p01_unlensed.dat'
theory_r = 0.0
theory_AL = 1.0

theory_ell = np.loadtxt(fid_r0_lensed_filename, usecols=0)
theory_dl_TT = np.loadtxt(fid_r0_lensed_filename, usecols=1)
theory_dl_EE = np.loadtxt(fid_r0_lensed_filename, usecols=2)
theory_dl_BB = np.loadtxt(fid_r0p01_unlensed_filename, usecols=3)
theory_dl_lens = np.loadtxt(fid_r0_lensed_filename, usecols=3)
theory_dl_TE = np.loadtxt(fid_r0_lensed_filename, usecols=4)

theory_cl_TT = 2 * np.pi * theory_dl_TT / (theory_ell * (theory_ell + 1))
theory_cl_EE = 2 * np.pi * theory_dl_EE / (theory_ell * (theory_ell + 1))
theory_cl_BB = 2 * np.pi * (theory_r * theory_dl_BB / 0.01 + theory_AL * theory_dl_lens) / (theory_ell * (theory_ell + 1))
theory_cl_TE = 2 * np.pi * theory_dl_TE / (theory_ell * (theory_ell + 1))

theory_cl_TT = np.insert(theory_cl_TT, 0, np.array([0, 0]))
theory_cl_EE = np.insert(theory_cl_EE, 0, np.array([0, 0]))
theory_cl_BB = np.insert(theory_cl_BB, 0, np.array([0, 0]))
theory_cl_TE = np.insert(theory_cl_TE, 0, np.array([0, 0]))

theory_spectra = (theory_cl_TT, theory_cl_EE, theory_cl_BB, theory_cl_TE)

for i in range(num_monte_carlo):

    print('Monte Carlo Sim {}/{}'.format(i + 1, num_monte_carlo))

    if not use_noise_model1:
        noise_realise_Q1 = np.random.normal(size=len(Qmap1)) * np.sqrt(Qcov1) * 1.05 # 5% calibration uncertainty for C-BASS
        noise_realise_U1 = np.random.normal(size=len(Umap1)) * np.sqrt(Ucov1) * 1.05
    elif use_noise_model1:
        noise_all1 = hp.synfast(nl1, nside, new=True)
        noise_realise_Q1 = noise_all1[1] / np.sqrt(hits1)
        noise_realise_U1 = noise_all1[2] / np.sqrt(hits1)
    if not use_noise_model2:
        noise_realise_Q2 = np.random.normal(size=len(Qmap2)) * np.sqrt(Qcov2) * 1.05
        noise_realise_U2 = np.random.normal(size=len(Umap2)) * np.sqrt(Ucov2) * 1.05
    elif use_noise_model2:
        noise_all2 = hp.synfast(nl2, nside, new=True)
        noise_realise_Q2 = noise_all2[1] / np.sqrt(hits2)
        noise_realise_U2 = noise_all2[2] / np.sqrt(hits2)
        
    cmb = hp.synfast(theory_spectra, nside, new=True) # Try maps in micro-K.
    cmbI1, cmbQ1, cmbU1 = hp.smoothing(cmb, fwhm=np.radians(fwhm1 / 60))
    cmbI2, cmbQ2, cmbU2 = hp.smoothing(cmb, fwhm=np.radians(fwhm2 / 60))
    sn_realise_Q1 = noise_realise_Q1 + cmbQ1 + diffuse_signal_Q1
    sn_realise_U1 = noise_realise_U1 + cmbU1 + diffuse_signal_U1
    sn_realise_Q2 = noise_realise_Q2 + cmbQ2 + diffuse_signal_Q2
    sn_realise_U2 = noise_realise_U2 + cmbU2 + diffuse_signal_U2
    sn_realise_Q1[mask == 0] = 0
    sn_realise_U1[mask == 0] = 0
    sn_realise_Q2[mask == 0] = 0
    sn_realise_U2[mask == 0] = 0

    sn_field_pureB_num1, sn_field_pureB_num2 = get_pureB_fields(sn_realise_Q1, sn_realise_U1, sn_realise_Q2, sn_realise_U2)
    snl_pureB = compute_master(sn_field_pureB_num1, sn_field_pureB_num2, w_pureB)
    snl_pureB_power.append(snl_pureB)
    sn_field_pureE_num1, sn_field_pureE_num2 = get_pureE_fields(sn_realise_Q1, sn_realise_U1, sn_realise_Q2, sn_realise_U2)
    snl_pureE = compute_master(sn_field_pureE_num1, sn_field_pureE_num2, w_pureE)
    snl_pureE_power.append(snl_pureE)
    sn_field_pureEB_num1, sn_field_pureEB_num2 = get_pureB_fields(sn_realise_Q1, sn_realise_U1, sn_realise_Q2, sn_realise_U2)
    snl_pureEB = compute_master(sn_field_pureEB_num1, sn_field_pureEB_num2, w_pureEB)
    snl_pureEB_power.append(snl_pureEB)
    sn_field_impure_num1, sn_field_impure_num2 = get_impure_fields(sn_realise_Q1, sn_realise_U1, sn_realise_Q2, sn_realise_U2)
    snl_impure = compute_master(sn_field_impure_num1, sn_field_impure_num2, w_impure)
    snl_impure_power.append(snl_impure)
    
snl_pureB_power = np.array(snl_pureB_power)
snl_pureE_power = np.array(snl_pureE_power)
snl_pureEB_power = np.array(snl_pureEB_power)
snl_impure_power = np.array(snl_impure_power)
cl_pureB_std = np.std(snl_pureB_power, axis=0)
cl_pureE_std = np.std(snl_pureE_power, axis=0)
cl_pureEB_std = np.std(snl_pureEB_power, axis=0)
cl_impure_std = np.std(snl_impure_power, axis=0)
nl_pureB_EE_EE_covar = np.cov(np.transpose(snl_pureB_power[:, 0, :]))
nl_pureE_EE_EE_covar = np.cov(np.transpose(snl_pureE_power[:, 0, :]))
nl_pureEB_EE_EE_covar = np.cov(np.transpose(snl_pureEB_power[:, 0, :]))
nl_impure_EE_EE_covar = np.cov(np.transpose(snl_impure_power[:, 0, :]))
nl_pureB_BB_BB_covar = np.cov(np.transpose(snl_pureB_power[:, 3, :]))
nl_pureE_BB_BB_covar = np.cov(np.transpose(snl_pureE_power[:, 3, :]))
nl_pureEB_BB_BB_covar = np.cov(np.transpose(snl_pureEB_power[:, 3, :]))
nl_impure_BB_BB_covar = np.cov(np.transpose(snl_impure_power[:, 3, :]))
nl_pureB_EB_EB_covar = np.cov(np.transpose(snl_pureB_power[:, 0, :]), np.transpose(snl_pureB_power[:, 3, :]))
nl_pureE_EB_EB_covar = np.cov(np.transpose(snl_pureE_power[:, 0, :]), np.transpose(snl_pureE_power[:, 3, :]))
nl_pureEB_EB_EB_covar = np.cov(np.transpose(snl_pureEB_power[:, 0, :]), np.transpose(snl_pureEB_power[:, 3, :]))
nl_impure_EB_EB_covar = np.cov(np.transpose(snl_impure_power[:, 0, :]), np.transpose(snl_impure_power[:, 3, :]))

print('Saving Monte Carlo output.')
np.savetxt(leff_out, leff)
np.savetxt(f'{cl_mc_mean_out}_pureB.txt', cl_pureB_mean)
np.savetxt(f'{cl_mc_mean_out}_pureE.txt', cl_pureE_mean)
np.savetxt(f'{cl_mc_mean_out}_pureEB.txt', cl_pureEB_mean)
np.savetxt(f'{cl_mc_mean_out}_impure.txt', cl_impure_mean)
np.savetxt(f'{cl_mc_std_out}_pureB.txt', cl_pureB_std)
np.savetxt(f'{cl_mc_std_out}_pureE.txt', cl_pureE_std)
np.savetxt(f'{cl_mc_std_out}_pureEB.txt', cl_pureEB_std)
np.savetxt(f'{cl_mc_std_out}_impure.txt', cl_impure_std)
np.savetxt(f'{cl_mc_cov_EE_EE_out}_pureB.txt', nl_pureB_EE_EE_covar)
np.savetxt(f'{cl_mc_cov_EE_EE_out}_pureE.txt', nl_pureE_EE_EE_covar)
np.savetxt(f'{cl_mc_cov_EE_EE_out}_pureEB.txt', nl_pureEB_EE_EE_covar)
np.savetxt(f'{cl_mc_cov_EE_EE_out}_impure.txt', nl_impure_EE_EE_covar)
np.savetxt(f'{cl_mc_cov_BB_BB_out}_pureB.txt', nl_pureB_BB_BB_covar)
np.savetxt(f'{cl_mc_cov_BB_BB_out}_pureE.txt', nl_pureE_BB_BB_covar)
np.savetxt(f'{cl_mc_cov_BB_BB_out}_pureEB.txt', nl_pureEB_BB_BB_covar)
np.savetxt(f'{cl_mc_cov_BB_BB_out}_impure.txt', nl_impure_BB_BB_covar)
np.savetxt(f'{cl_mc_cov_EB_EB_out}_pureB.txt', nl_pureB_EB_EB_covar)
np.savetxt(f'{cl_mc_cov_EB_EB_out}_pureE.txt', nl_pureE_EB_EB_covar)
np.savetxt(f'{cl_mc_cov_EB_EB_out}_pureEB.txt', nl_pureEB_EB_EB_covar)
np.savetxt(f'{cl_mc_cov_EB_EB_out}_impure.txt', nl_impure_EB_EB_covar)

print('Congratulations, you completed the power spectrum estimation!')
