import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import healpy as hp
import numpy as np


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'figure.figsize': [12.0, 10.4]})

class Power_Spectrum_Plots(object):
    
    def __init__(self, cl_filename, cl_err_filename, leff_filename, out_dir, out_prefix, lmin=5, lmax=150,
                 unit_conv=1e6):

        """

        Attributes
        ----------
        
        cl_filename: Filename of .txt file containing Cl spin-2 bandpowers (in NaMaster format) - str.
        cl_err_filename: Filename of .txt file containing Cl spin-2 bandpower errors (in NaMaster format) - str.
        leff_filename: Filename of .txt file containing the effective multipole values - str.
        out_dir: Output directory for figures - str.
        out_prefix: Prefix for output filenames - str.
        lmin: Minimum multipole to plot - int.
        lmax: Maximum multipole to plot - int.
        unit_conv: Factor to multiply power spectrum to get them in units of mK^2 - float.

        """

        self.cl_filename = cl_filename
        self.cl_err_filename = cl_err_filename
        self.leff_filename = leff_filename
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.lmin = lmin
        self.lmax = lmax
        self.unit_conv = unit_conv
        
    def extract_leff(self):
        
        leff = np.ndarray.flatten(np.loadtxt(self.leff_filename))

        return leff
    
    def extract_clBB(self):

        clBB = np.ndarray.flatten(np.loadtxt(self.cl_filename)[3]) * self.unit_conv
        clBB_err = np.ndarray.flatten(np.loadtxt(self.cl_err_filename)[3]) * self.unit_conv
        
        return clBB, clBB_err

    def extract_clEE(self):

        clEE = np.ndarray.flatten(np.loadtxt(self.cl_filename)[0]) * self.unit_conv
        clEE_err = np.ndarray.flatten(np.loadtxt(self.cl_err_filename)[0]) * self.unit_conv

        return clEE, clEE_err

    def extract_clEB(self):
        
        clEB = np.ndarray.flatten(np.loadtxt(self.cl_filename)[1]) * self.unit_conv
        clEB_err = np.ndarray.flatten(np.loadtxt(self.cl_err_filename)[1]) * self.unit_conv

        return clEB, clEB_err

    def clBB_plot(self):
        
        leff = self.extract_leff()
        clBB, clBB_err = self.extract_clBB()

        plt.figure()
        plt.errorbar(leff, clBB, yerr=clBB_err, color='k', fmt='o')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{BB}\quad [\mathrm{mK}^2]$')
        plt.yscale('log')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.xlim(self.lmin, self.lmax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_clBB.pdf', dpi=900)
        plt.close()
        
        return leff, clBB, clBB_err
    
    def clEE_plot(self):
        
        leff = self.extract_leff()
        clEE, clEE_err = self.extract_clEE()
        
        plt.figure()
        plt.errorbar(leff, clEE, yerr=clEE_err, color='k', fmt='o')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{EE}\quad [\mathrm{mK}^2]$')
        plt.yscale('log')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.xlim(self.lmin, self.lmax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_clEE.pdf', dpi=900)
        plt.close()
        
        return leff, clEE, clEE_err
    
    def clEB_plot(self):
        
        leff = self.extract_leff()
        clEB, clEB_err = self.extract_clEB()
        
        plt.figure()
        plt.errorbar(leff, clEB, yerr=clEB_err, color='k', fmt='o')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{EB}\quad [\mathrm{mK}^2]$')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.xlim(self.lmin, self.lmax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_clEB.pdf', dpi=900)
        plt.close()
        
        return leff, clEB, clEB_err
    
    def dlBB_plot(self):
        
        leff = self.extract_leff()
        clBB, clBB_err = self.extract_clBB()
        
        dlBB = leff * (leff + 1) * clBB / (2 * np.pi)
        dlBB_err = leff * (leff + 1) * clBB_err / (2 * np.pi)

        ymin = np.amin(dlBB[np.logical_and(leff > self.lmin, leff < self.lmax)]) / 2
        ymax = np.amax(dlBB[np.logical_and(leff > self.lmin, leff < self.lmax)]) * 2
        
        plt.figure()
        plt.errorbar(leff, dlBB, yerr=dlBB_err, color='k', fmt='o')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell^{BB}/2\pi\quad [\mathrm{mK}^2]$')
        plt.yscale('log')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.xlim(self.lmin, self.lmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlBB.pdf', dpi=900)
        plt.close()
        
        return leff, dlBB, dlBB_err
    
    def dlEE_plot(self):
        
        leff = self.extract_leff()
        clEE, clEE_err = self.extract_clEE()
        
        dlEE = leff * (leff + 1) * clEE / (2 * np.pi)
        dlEE_err = leff * (leff + 1) * clEE_err / (2 * np.pi)

        ymin = np.amin(dlEE[np.logical_and(leff > self.lmin, leff < self.lmax)]) / 2
        ymax = np.amax(dlEE[np.logical_and(leff > self.lmin, leff < self.lmax)]) * 2
        
        plt.figure()
        plt.errorbar(leff, dlEE, yerr=dlEE_err, color='k', fmt='o')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell^{EE}/2\pi\quad [\mathrm{mK}^2]$')
        plt.yscale('log')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.xlim(self.lmin, self.lmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlEE.pdf', dpi=900)
        plt.close()
        
        return leff, dlEE, dlEE_err
    
    def dlEB_plot(self):
        
        leff = self.extract_leff()
        clEB, clEB_err = self.extract_clEB()
        
        dlEB = leff * (leff + 1) * clEB / (2 * np.pi)
        dlEB_err = leff * (leff + 1) * clEB_err / (2 * np.pi)

        ymin = np.amin(dlEB[np.logical_and(leff > self.lmin, leff < self.lmax)]) * 1.5
        ymax = np.amax(dlEB[np.logical_and(leff > self.lmin, leff < self.lmax)]) * 1.5
        
        plt.figure()
        plt.errorbar(leff, dlEB, yerr=dlEB_err, color='k', fmt='o')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell^{EB}/2\pi\quad [\mathrm{mK}^2]$')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.xlim(self.lmin, self.lmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlEB.pdf', dpi=900)
        plt.close()
        
        return leff, dlEB, dlEB_err
    
    def dlBB_dlEE_plot(self):
        
        leff = self.extract_leff()
        clBB, clBB_err = self.extract_clBB()
        clEE, clEE_err = self.extract_clEE()
        
        dlBB = leff * (leff + 1) * clBB / (2 * np.pi)
        dlBB_err = leff * (leff + 1) * clBB_err / (2 * np.pi)
        dlEE = leff * (leff + 1) * clEE / (2 * np.pi)
        dlEE_err = leff * (leff + 1) * clEE_err / (2 * np.pi)

        ymin = np.amin(dlBB[np.logical_and(leff > self.lmin, leff < self.lmax)]) / 2
        ymax = np.amax(dlEE[np.logical_and(leff > self.lmin, leff < self.lmax)]) * 2
        
        plt.figure()
        plt.errorbar(leff, dlBB, yerr=dlBB_err, color='k', fmt='x', label=r'$BB$')
        plt.errorbar(leff, dlEE, yerr=dlEE_err, color='k', fmt='o', label=r'$EE$')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi\quad [\mathrm{mK}^2]$')
        plt.yscale('log')
        plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
        plt.legend(loc='upper right')
        plt.xlim(self.lmin, self.lmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlBB_dlEE.pdf', dpi=900)
        plt.close()
        
        return leff, dlBB, dlBB_err, dlEE, dlEE_err
