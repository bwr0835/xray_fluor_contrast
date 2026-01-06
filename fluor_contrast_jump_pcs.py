'''
File: fluor_contrast_jump_pcs_final.py
Author: B. Roter
Date: December 10, 2025

Using the xraylib (XRL) fundamental parameter database [1], this program calculates the minimum number of incident photons and radiation skin dose per pixel required to detect a user-defined
number of x-ray fluorescence (XRF) photons at a given excitaiton energy and limit of detection. For the dose, the model protein assumed is H48.6C32.9N8.9O8.9S0.6 [2].

The following xraylib (XRL) functions are utilized in this code:

Mass photoionization cross sections (cm^2/g)
    CS_Photo (jump ratio-approximated total photoionization cross section)
    CS_Photo_Total (excitation-dependent total mass photoionization cross section)
    CS_Photo_Total_CP (excitation-dependent total mass photoionization cross section for compounds)
    CS_Photo_Partial [excitation-dependent subshell mass photoionization partial cross section (PCS)]
    
Mass x-ray fluorescence (XRF) production cross sections (cm^2/g)
    CS_FluorLine_Kissel_Cascade (excitation-dependent mass XRF production cross section; Coster-Kronig (CK) transitions and cascade effects included)
    CS_FluorLine_Kissel_no_Cascade (excitation-dependent mass XRF production cross section; CK transitions included, cascade effects excluded)
    CS_FluorShell_Kissel_Cascade (excitation-dependent subshell mass XRF production cross section; CK transitions, cascade effects included)
    CS_FluorShell_Kissel_no_Cascade (excitation-dependent subshell mass XRF production cross section; CK transitions included, cascade effects excluded)

Absorption edge and XRF transition parameters (energies are in keV)
    CosKronTransProb (CK yield)
    EdgeEnergy (absorption edge energy)
    FluorYield (fluorescence yield)
    JumpFactor (absorption edge jump ratio)
    LineEnergy (XRF line energy)
    RadRate (XRF fractional yield/branching ratio)

Chemistry (masses are in g)
    AtomicWeight (elemental molar mass)
    GetCompoundDataNISTByIndex (information on XRL-defined NIST compounds)
    SymbolToAtomicNumber (symbol-to-atomic number conversion)
    
#################################################################################

References:

[1] T. Schoonjans, A. Brunetti, B. Golosio, et al., “The xraylib library for X-ray-matter interactions. Recent developments,” 
Spectrochimica Acta Part B: At. Spectrosc. 66, 776-784 (2011).

[2] R. A. London, M. D. Rosen, and J. E. Trebes, “Wavelength choice for soft x-ray laser holography of biological samples,” 
Appl. Opt. 28, 3397-3404 (1989).

'''

import numpy as np, xraylib as xrl, scipy.constants as cnsts

def mass_photo_pcs_jump(Z, subshell, E0): # Jump ratio-approximated mass photoionization PCSes
    try:
        mass_photo_cs_total = xrl.CS_Photo(Z, E0)
    
    except:
        return 0

    if subshell == xrl.K_SHELL:
        try:
            r_K = xrl.JumpFactor(Z, subshell)
        
        except:
            r_K = 0

        try:
            E_K = xrl.EdgeEnergy(Z, subshell)
        
        except:
            E_K = 0

        if E0 >= E_K and E_K > 0 and r_K > 0:
            J_K = 1 - 1/r_K

            return mass_photo_cs_total*J_K

        else:
            return 0
    
    elif subshell == xrl.L1_SHELL:
        try:
            r_L1 = xrl.JumpFactor(Z, subshell)
    
        except:
            r_L1 = 0
        
        try:
            r_K = xrl.JumpFactor(Z, xrl.K_SHELL)

        except:
            r_K = 0

        try:
            E_K = xrl.EdgeEnergy(Z, xrl.K_SHELL)
        
        except:
            E_K = 0
        
        try:
            E_L1 = xrl.EdgeEnergy(Z, subshell)
        
        except:
            E_L1 = 0
        
        if E0 >= E_K and E_K > 0 and r_K > 0 and r_L1 > 0:
            J_L1 = (1/r_K)*(1 - 1/r_L1)

            return mass_photo_cs_total*J_L1
        
        elif E0 >= E_L1 and E_L1 > 0 and r_L1 > 0:
            J_L1 = 1 - 1/r_L1
            
            return mass_photo_cs_total*J_L1
        
        else:
            return 0
        
    elif subshell == xrl.L2_SHELL:
        try:
            r_L2 = xrl.JumpFactor(Z, subshell)
    
        except:
            r_L2 = 0
        
        try:
            r_L1 = xrl.JumpFactor(Z, xrl.L1_SHELL)

        except:
            r_L1 = 0
        
        try:
            r_K = xrl.JumpFactor(Z, xrl.K_SHELL)
    
        except:
            r_K = 0
        
        try:
            E_K = xrl.EdgeEnergy(Z, xrl.K_SHELL)
        
        except:
            E_K = 0
        
        try:
            E_L1 = xrl.EdgeEnergy(Z, xrl.L1_SHELL)
        
        except:
            E_L1 = 0
        
        try:
            E_L2 = xrl.EdgeEnergy(Z, subshell)
        
        except:
            E_L2 = 0

        if E0 >= E_K and E_K > 0 and r_K > 0 and r_L1 > 0 and r_L2 > 0:
            J_L2 = (1/(r_K*r_L1))*(1 - 1/r_L2)

            return mass_photo_cs_total*J_L2
        
        elif E0 >= E_L1 and E_L1 > 0 and r_L1 > 0 and r_L2 > 0:
            J_L2 = (1/r_L1)*(1 - 1/r_L2)
        
            return mass_photo_cs_total*J_L2
        
        elif E0 >= E_L2 and E_L2 > 0 and r_L2 > 0:
            J_L2 = 1 - 1/r_L2

            return mass_photo_cs_total*J_L2
        
        else:
            return 0
        
    elif subshell == xrl.L3_SHELL:
        try:
            r_K = xrl.JumpFactor(Z, xrl.K_SHELL)
    
        except:
            r_K = 0
        
        try:
            r_L1 = xrl.JumpFactor(Z, xrl.L1_SHELL)

        except:
            r_L1 = 0
        
        try:
            r_L2 = xrl.JumpFactor(Z, xrl.L2_SHELL)
    
        except:
            r_L2 = 0
        
        try:
            r_L3 = xrl.JumpFactor(Z, subshell)

        except:
            r_L3 = 0
        
        try:
            E_K = xrl.EdgeEnergy(Z, xrl.K_SHELL)
        
        except:
            E_K = 0
        
        try:
            E_L1 = xrl.EdgeEnergy(Z, xrl.L1_SHELL)
        
        except:
            E_L1 = 0

        try:
            E_L2 = xrl.EdgeEnergy(Z, xrl.L2_SHELL)
        
        except:
            E_L2 = 0
        
        try:
            E_L3 = xrl.EdgeEnergy(Z, subshell)
        
        except:
            E_L3 = 0
        
        if E0 >= E_K and E_K > 0 and r_K > 0 and r_L1 > 0 and r_L2 > 0 and r_L3 > 0:
            J_L3 = (1/(r_K*r_L1*r_L2))*(1 - 1/r_L3)

            return mass_photo_cs_total*J_L3
        
        elif E0 >= E_L1 and E_L1 > 0 and r_L1 > 0 and r_L2 and r_L3 > 0:
            J_L3 = (1/(r_L1*r_L2))*(1 - 1/r_L3)
        
            return mass_photo_cs_total*J_L3
        
        elif E0 >= E_L2 and E_L2 > 0 and r_L2 > 0 and r_L3 > 0:
            J_L3 = (1/r_L2)*(1 - 1/r_L3)

            return mass_photo_cs_total*J_L3
        
        elif E0 >= E_L3 and E_L3 > 0 and r_L3 > 0:
            J_L3 = 1 - 1/r_L3

            return mass_photo_cs_total*J_L3

        else:
            return 0 

def mass_photo_pcs_jump_with_ck_no_ck_constraints(Z, subshell, E0): # Modified jump ratio-approximated mass photoionization PCSes when including CK transitions
                                                                    # (CS_FluorShell and CS_FluorLine ignore whole subshells even if at least one CK yield is nonzero, 
                                                                    # while CS_FluorShell_Kissel and other functions related to that one do not)
    if subshell == xrl.K_SHELL:
        mass_photo_pcs_K = mass_photo_pcs_jump(Z, subshell, E0)
        
        if mass_photo_pcs_K > 0:
            try:
                E_K = xrl.EdgeEnergy(Z, subshell)
            
            except:
                E_K = 0

            if E0 >= E_K and E_K > 0:
                return mass_photo_pcs_K
        
            else:
                return 0
        
        else:
            return 0
    
    elif subshell == xrl.L1_SHELL:
        mass_photo_pcs_L1 = mass_photo_pcs_jump(Z, subshell, E0)

        if mass_photo_pcs_L1 > 0:
            try:
                E_L1 = xrl.EdgeEnergy(Z, subshell)
            
            except:
                E_L1 = 0

            if E0 >= E_L1 and E_L1 > 0:
                return mass_photo_pcs_L1
            
            else:
                return 0
            
        else:
            return 0
    
    elif subshell == xrl.L2_SHELL:
        mass_photo_pcs_L2 = mass_photo_pcs_jump(Z, subshell, E0)
        mass_photo_pcs_L1 = mass_photo_pcs_jump(Z, xrl.L1_SHELL, E0)
        
        try:
            f_L1L2 = xrl.CosKronTransProb(Z, xrl.FL12_TRANS)
        
        except:
            f_L1L2 = 0

        if mass_photo_pcs_L2 > 0:
            try:
                E_L1 = xrl.EdgeEnergy(Z, xrl.L1_SHELL)
            
            except:
                E_L1 = 0
            
            try:
                E_L2 = xrl.EdgeEnergy(Z, subshell)

            except:
                E_L2 = 0
            
            if E0 >= E_L1 and E_L1 > 0:
                mass_photo_pcs_L2 += mass_photo_pcs_L1*f_L1L2

                return mass_photo_pcs_L2
        
            elif E0 >= E_L2 and E_L1 > 0:
                return mass_photo_pcs_L2
        
            else:
                return 0
        
        else:
            return 0
        
    elif subshell == xrl.L3_SHELL:
        mass_photo_pcs_L3 = mass_photo_pcs_jump(Z, subshell, E0)
        mass_photo_pcs_L2 = mass_photo_pcs_jump(Z, xrl.L2_SHELL, E0)
        mass_photo_pcs_L1 = mass_photo_pcs_jump(Z, xrl.L1_SHELL, E0)

        if mass_photo_pcs_L3 > 0:
            try:
                f_L1L2 = xrl.CosKronTransProb(Z, xrl.FL12_TRANS)
            
            except:
                f_L1L2 = 0
            
            try:
                f_L1L3 = xrl.CosKronTransProb(Z, xrl.FL13_TRANS)
        
            except:
                f_L1L3 = 0
            
            try:
                f_L1L3p = xrl.CosKronTransProb(Z, xrl.FLP13_TRANS)
        
            except:
                f_L1L3p = 0
            
            try:
                f_L2L3 = xrl.CosKronTransProb(Z, xrl.FL23_TRANS)
        
            except:
                f_L2L3 = 0

            try:
                E_L1 = xrl.EdgeEnergy(Z, xrl.L1_SHELL)
            
            except:
                E_L1 = 0
            
            try:
                E_L2 = xrl.EdgeEnergy(Z, xrl.L2_SHELL)

            except:
                E_L2 = 0
            
            try:
                E_L3 = xrl.EdgeEnergy(Z, subshell)
            
            except:
                E_L3 = 0
            
            if E0 >= E_L1:
                mass_photo_pcs_L3 += (mass_photo_pcs_L2*f_L2L3 + mass_photo_pcs_L1*(f_L1L3 + f_L1L3p + f_L1L2*f_L2L3))

                return mass_photo_pcs_L3

            elif E0 >= E_L2:
                mass_photo_pcs_L3 += (mass_photo_pcs_L2*f_L2L3)

                return mass_photo_pcs_L3
        
            elif E0 >= E_L3:
                return mass_photo_pcs_L3
            
            else:
                return 0
        
        else:
            return 0

# Inputs

E0_kev = np.linspace(4, 34, 151) # 200 eV spacing
dE0 = E0_kev[1] - E0_kev[0]

n_E0 = len(E0_kev)

rhop_ug_cm2 = 0.05 # Minimum detectable mass density (limit of detection/LOD) (ug/cm^2) (use with theoretical calculations only!)

Z_start = 10 # Neon
Z_end = 92 # Uranium
dZ = Z_end - Z_start

Z_idx_array = np.arange(dZ + 1)

n_Z = len(Z_idx_array)

t_be_um = 25 # Beryllium window thickness (microns)
rho_be_g_cm3 = 1.85 # Volume mass density of beryllium (g/cm^3)

rhop_be_g_cm2 = t_be_um*1e-4*rho_be_g_cm3 # Areal mass density of beryllium (g/cm^2)

omega = 1.35 # Solid angle
omega_frac = omega/(4*np.pi) # Solid angle fraction relative to a sphere

N_fluor = 5 # Theoretical number of XRF photons detected per pixel

# XRL-defined subshells

subshells = [xrl.K_SHELL,
             xrl.L1_SHELL, 
             xrl.L2_SHELL, 
             xrl.L3_SHELL]

n_subshells = len(subshells)

# XRL-defined fluorescence lines

k_fluor_lines = [xrl.KL1_LINE,
                 xrl.KL2_LINE,
                 xrl.KL3_LINE,
                 xrl.KM1_LINE,
                 xrl.KM2_LINE,
                 xrl.KM3_LINE,
                 xrl.KM4_LINE,
                 xrl.KM5_LINE,
                 xrl.KN1_LINE,
                 xrl.KN2_LINE,
                 xrl.KN3_LINE,
                 xrl.KN4_LINE,
                 xrl.KN5_LINE,
                 xrl.KN6_LINE,
                 xrl.KN7_LINE,
                 xrl.KO1_LINE,
                 xrl.KO2_LINE,
                 xrl.KO3_LINE,
                 xrl.KO4_LINE,
                 xrl.KO5_LINE,
                 xrl.KO6_LINE,
                 xrl.KO7_LINE,
                 xrl.KP1_LINE,
                 xrl.KP2_LINE,
                 xrl.KP3_LINE,
                 xrl.KP4_LINE,
                 xrl.KP5_LINE]

l1_fluor_lines = [xrl.L1L2_LINE,
                  xrl.L1L3_LINE,
                  xrl.L1M1_LINE,
                  xrl.L1M2_LINE,
                  xrl.L1M3_LINE,
                  xrl.L1M4_LINE,
                  xrl.L1M5_LINE,
                  xrl.L1N1_LINE,
                  xrl.L1N2_LINE,
                  xrl.L1N3_LINE,
                  xrl.L1N4_LINE,
                  xrl.L1N5_LINE,
                  xrl.L1N67_LINE,
                  xrl.L1O1_LINE,
                  xrl.L1O2_LINE,
                  xrl.L1O3_LINE,
                  xrl.L1O45_LINE,
                  xrl.L1O6_LINE,
                  xrl.L1O7_LINE,
                  xrl.L1P1_LINE,
                  xrl.L1P23_LINE,
                  xrl.L1P4_LINE,
                  xrl.L1P5_LINE]

l2_fluor_lines = [xrl.L2L3_LINE,
                  xrl.L2M1_LINE,
                  xrl.L2M2_LINE,
                  xrl.L2M3_LINE,
                  xrl.L2M4_LINE,
                  xrl.L2M5_LINE,
                  xrl.L2N1_LINE,
                  xrl.L2N2_LINE,
                  xrl.L2N3_LINE,
                  xrl.L2N4_LINE,
                  xrl.L2N5_LINE,
                  xrl.L2N6_LINE,
                  xrl.L2N7_LINE,
                  xrl.L2O1_LINE,
                  xrl.L2O2_LINE,
                  xrl.L2O3_LINE,
                  xrl.L2O4_LINE,
                  xrl.L2O5_LINE,
                  xrl.L2O6_LINE,
                  xrl.L2O7_LINE,
                  xrl.L2P1_LINE,
                  xrl.L2P23_LINE,
                  xrl.L2P4_LINE,
                  xrl.L2P5_LINE,
                  xrl.L2Q1_LINE]

l3_fluor_lines = [xrl.L3M1_LINE,
                  xrl.L3M2_LINE,
                  xrl.L3M3_LINE,
                  xrl.L3M4_LINE,
                  xrl.L3M5_LINE,
                  xrl.L3N1_LINE,
                  xrl.L3N2_LINE,
                  xrl.L3N3_LINE,
                  xrl.L3N4_LINE,
                  xrl.L3N5_LINE,
                  xrl.L3N6_LINE,
                  xrl.L3N7_LINE,
                  xrl.L3O1_LINE,
                  xrl.L3O2_LINE,
                  xrl.L3O3_LINE,
                  xrl.L3O45_LINE,
                  xrl.L3O6_LINE,
                  xrl.L3O7_LINE,
                  xrl.L3P23_LINE,
                  xrl.L3P45_LINE,
                  xrl.L3Q1_LINE]

fluor_lines = [k_fluor_lines,
               l1_fluor_lines, 
               l2_fluor_lines, 
               l3_fluor_lines]

# Cross section inputs [cm^2/g (unless otherwise stated)]

# Jump-ratio approximated mass XRF production cross sections

mass_fluor_pcs_total_jump_ck = np.zeros((n_Z, n_subshells, n_E0))
mass_fluor_pcs_total_jump_ck_windowless = np.zeros((n_Z, n_subshells, n_E0))

# Excitation-dependent mass XRF production cross sections

mass_fluor_pcs_total_no_ck_no_cascade = np.zeros((n_Z, n_subshells, n_E0))
mass_fluor_pcs_total_no_ck_no_cascade_windowless = np.zeros((n_Z, n_subshells, n_E0))
mass_fluor_pcs_total_ck_no_cascade = np.zeros((n_Z, n_subshells, n_E0))
mass_fluor_pcs_total_ck_no_cascade_windowless = np.zeros((n_Z, n_subshells, n_E0))
mass_fluor_pcs_total_ck_cascade = np.zeros((n_Z, n_subshells, n_E0))
mass_fluor_pcs_total_ck_cascade_windowless = np.zeros((n_Z, n_subshells, n_E0))

# Minimum number of incident photons per pixel (photons/pixel)

# Jump ratio-approximated

min_required_photons_total_jump_ck = np.zeros((n_Z, n_E0))
min_required_photons_total_jump_ck_windowless = np.zeros((n_Z, n_E0))

# Excitation-dependent PCSes considered

min_required_photons_total_pcs_no_ck_no_cascade = np.zeros((n_Z, n_E0))
min_required_photons_total_pcs_no_ck_no_cascade_windowless = np.zeros((n_Z, n_E0))
min_required_photons_total_pcs_ck_no_cascade = np.zeros((n_Z, n_E0))
min_required_photons_total_pcs_ck_no_cascade_windowless = np.zeros((n_Z, n_E0))
min_required_photons_total_pcs_ck_cascade = np.zeros((n_Z, n_E0))
min_required_photons_total_pcs_ck_cascade_windowless = np.zeros((n_Z, n_E0))

# Model protein elements

protein_matrix = 'H48.6C32.9N8.9O8.9S0.6' # See [2]

r_beam_um = 0.02 # Incident beam radius (um)

theta = 0 # Sample tilt relative to incident beam normal (degrees)

A_beam = np.pi*(r_beam_um**2)/np.cos(np.deg2rad(theta)) # Beam spot (assumed to be circular, but affected by sample tilt -- beam width in one direction scaled by factor of 1/cos(theta))

# Radiation skin dose per pixel (Gy/pixel)

# Jump ratio-approximated

dose_jump_ck = np.zeros((n_Z, n_subshells, n_E0))
dose_jump_ck_windowless = np.zeros((n_Z, n_subshells, n_E0))
dose_jump_ck_total = np.zeros((n_Z, n_E0))
dose_jump_ck_total_windowless = np.zeros((n_Z, n_E0))

# Excitation-dependent PCS consideration

dose_pcs_no_ck_no_cascade = np.zeros((n_Z, n_subshells, n_E0))
dose_pcs_no_ck_no_cascade_windowless = np.zeros((n_Z, n_subshells, n_E0))
dose_pcs_ck_no_cascade = np.zeros((n_Z, n_subshells, n_E0))
dose_pcs_ck_no_cascade_windowless = np.zeros((n_Z, n_subshells, n_E0))
dose_pcs_ck_cascade = np.zeros((n_Z, n_subshells, n_E0))
dose_pcs_ck_cascade_windowless = np.zeros((n_Z, n_subshells, n_E0))
dose_pcs_no_ck_no_cascade_total = np.zeros((n_Z, n_E0))
dose_pcs_no_ck_no_cascade_total_windowless = np.zeros((n_Z, n_E0))
dose_pcs_ck_no_cascade_total = np.zeros((n_Z, n_E0))
dose_pcs_ck_no_cascade_total_windowless = np.zeros((n_Z, n_E0))
dose_pcs_ck_cascade_total = np.zeros((n_Z, n_E0))
dose_pcs_ck_cascade_total_windowless = np.zeros((n_Z, n_E0))

A_protein = 0 # Total protein molar mass

for Z in Z_idx_array:
    for n in range(n_subshells):
        try: # Fluorescence yields
            fluor_yield = xrl.FluorYield(Z + Z_start, subshells[n]) 
        
        except:
            fluor_yield = 0

        for E in range(n_E0):
            mass_fluor_pcs_total_jump_ck_windowless[Z, n, E] = mass_photo_pcs_jump_with_ck_no_ck_constraints(Z + Z_start, subshells[n], E0_kev[E])*fluor_yield
            
            try:
                mass_fluor_pcs_total_no_ck_no_cascade_windowless[Z, n, E] = xrl.CS_Photo_Partial(Z + Z_start, subshells[n], E0_kev[E])*fluor_yield

            except:
                mass_fluor_pcs_total_no_ck_no_cascade_windowless[Z, n, E] = 0
            
            try:
                mass_fluor_pcs_total_ck_no_cascade_windowless[Z, n, E] = xrl.CS_FluorShell_Kissel_no_Cascade(Z + Z_start, subshells[n], E0_kev[E])
                
            except:
                mass_fluor_pcs_total_ck_no_cascade_windowless[Z, n, E] = 0
            
            try:
                mass_fluor_pcs_total_ck_cascade_windowless[Z, n, E] = xrl.CS_FluorShell_Kissel_Cascade(Z + Z_start, subshells[n], E0_kev[E])
            
            except:
                mass_fluor_pcs_total_ck_cascade_windowless[Z, n, E] = 0
                
            for line in fluor_lines[n]:
                try: # XRF line energy
                    fluor_energy = xrl.LineEnergy(Z + Z_start, line)
                
                except:
                    fluor_energy = 0
                
                if fluor_energy > 0:
                    try:
                        mass_photo_cs_total_be = xrl.CS_Photo_Total(4, fluor_energy)
                    
                    except:
                        mass_photo_cs_total_be = 0
                    
                    try:
                        mass_photo_cs_total_be_jump = xrl.CS_Photo(4, fluor_energy)
                    
                    except:
                        mass_photo_cs_total_be_jump = 0

                    if mass_photo_cs_total_be > 0:
                        try:
                            frac_emission = xrl.RadRate(Z + Z_start, line)

                            mass_fluor_pcs_total_no_ck_no_cascade[Z, n, E] += mass_fluor_pcs_total_no_ck_no_cascade_windowless[Z, n, E]*frac_emission*np.exp(-mass_photo_cs_total_be*rhop_be_g_cm2)
                
                        except:
                            pass
                
                        try:
                            mass_fluor_pcs_ck_no_cascade = xrl.CS_FluorLine_Kissel_no_Cascade(Z + Z_start, line, E0_kev[E])
                            
                            mass_fluor_pcs_total_ck_no_cascade[Z, n, E] += mass_fluor_pcs_ck_no_cascade*np.exp(-mass_photo_cs_total_be*rhop_be_g_cm2)
                
                        except:
                            pass
                
                        try: 
                            mass_fluor_pcs_ck_cascade = xrl.CS_FluorLine_Kissel_Cascade(Z + Z_start, line, E0_kev[E])

                            mass_fluor_pcs_total_ck_cascade[Z, n, E] += mass_fluor_pcs_ck_cascade*np.exp(-mass_photo_cs_total_be*rhop_be_g_cm2)
                
                        except:
                            pass
                    
                    if mass_photo_cs_total_be_jump > 0:
                        try:
                            frac_emission = xrl.RadRate(Z + Z_start, line)

                            mass_fluor_pcs_total_jump_ck[Z, n, E] += mass_fluor_pcs_total_jump_ck_windowless[Z, n, E]*frac_emission*np.exp(-mass_photo_cs_total_be_jump*rhop_be_g_cm2)

                        except:
                            continue

normalized_intensity_jump_ck = omega_frac*mass_fluor_pcs_total_jump_ck*rhop_ug_cm2*1e-6
normalized_intensity_jump_ck_windowless = omega_frac*mass_fluor_pcs_total_jump_ck_windowless*rhop_ug_cm2*1e-6
normalized_intensity_pcs_no_ck_no_cascade = omega_frac*mass_fluor_pcs_total_no_ck_no_cascade*rhop_ug_cm2*1e-6
normalized_intensity_pcs_no_ck_no_cascade_windowless = omega_frac*mass_fluor_pcs_total_no_ck_no_cascade_windowless*rhop_ug_cm2*1e-6
normalized_intensity_pcs_ck_no_cascade = omega_frac*mass_fluor_pcs_total_ck_no_cascade*rhop_ug_cm2*1e-6
normalized_intensity_pcs_ck_no_cascade_windowless = omega_frac*mass_fluor_pcs_total_ck_no_cascade_windowless*rhop_ug_cm2*1e-6
normalized_intensity_pcs_ck_cascade = omega_frac*mass_fluor_pcs_total_ck_cascade*rhop_ug_cm2*1e-6
normalized_intensity_pcs_ck_cascade_windowless = omega_frac*mass_fluor_pcs_total_ck_cascade_windowless*rhop_ug_cm2*1e-6

normalized_intensity_total_jump_ck = np.sum(normalized_intensity_jump_ck, axis = 1)
normalized_intensity_total_jump_ck_windowless = np.sum(normalized_intensity_jump_ck_windowless, axis = 1)
normalized_intensity_total_pcs_no_ck_no_cascade = np.sum(normalized_intensity_pcs_no_ck_no_cascade, axis = 1)
normalized_intensity_total_pcs_no_ck_no_cascade_windowless = np.sum(normalized_intensity_pcs_no_ck_no_cascade_windowless, axis = 1)
normalized_intensity_total_pcs_ck_no_cascade = np.sum(normalized_intensity_pcs_ck_no_cascade, axis = 1)
normalized_intensity_total_pcs_ck_no_cascade_windowless = np.sum(normalized_intensity_pcs_ck_no_cascade_windowless, axis = 1)
normalized_intensity_total_pcs_ck_cascade = np.sum(normalized_intensity_pcs_ck_cascade, axis = 1)
normalized_intensity_total_pcs_ck_cascade_windowless = np.sum(normalized_intensity_pcs_ck_cascade_windowless, axis = 1)

min_required_photons_total_jump_ck = N_fluor/normalized_intensity_total_jump_ck
min_required_photons_total_jump_ck_windowless = N_fluor/normalized_intensity_total_jump_ck_windowless
min_required_photons_total_pcs_no_ck_no_cascade = N_fluor/normalized_intensity_total_pcs_no_ck_no_cascade
min_required_photons_total_pcs_no_ck_no_cascade_windowless = N_fluor/normalized_intensity_total_pcs_no_ck_no_cascade_windowless
min_required_photons_total_pcs_ck_no_cascade = N_fluor/normalized_intensity_total_pcs_ck_no_cascade
min_required_photons_total_pcs_ck_no_cascade_windowless = N_fluor/normalized_intensity_total_pcs_ck_no_cascade_windowless
min_required_photons_total_pcs_ck_cascade = N_fluor/normalized_intensity_total_pcs_ck_cascade
min_required_photons_total_pcs_ck_cascade_windowless = N_fluor/normalized_intensity_total_pcs_ck_cascade_windowless     

min_required_photons_jump_ck = N_fluor/normalized_intensity_jump_ck
min_required_photons_jump_ck_windowless = N_fluor/normalized_intensity_jump_ck_windowless
min_required_photons_pcs_no_ck_no_cascade = N_fluor/normalized_intensity_pcs_no_ck_no_cascade
min_required_photons_pcs_no_ck_no_cascade_windowless = N_fluor/normalized_intensity_pcs_no_ck_no_cascade_windowless
min_required_photons_pcs_ck_no_cascade = N_fluor/normalized_intensity_pcs_ck_no_cascade
min_required_photons_pcs_ck_no_cascade_windowless = N_fluor/normalized_intensity_pcs_ck_no_cascade_windowless
min_required_photons_pcs_ck_cascade = N_fluor/normalized_intensity_pcs_ck_cascade
min_required_photons_pcs_ck_cascade_windowless = N_fluor/normalized_intensity_pcs_ck_cascade_windowless

fluence_jump_ck_photons_um2 = min_required_photons_jump_ck/A_beam
fluence_jump_ck_photons_um2_windowless = min_required_photons_jump_ck_windowless/A_beam
fluence_pcs_no_ck_no_cascade_photons_um2 = min_required_photons_pcs_no_ck_no_cascade/A_beam
fluence_pcs_no_ck_no_cascade_photons_um2_windowless = min_required_photons_pcs_no_ck_no_cascade_windowless/A_beam
fluence_pcs_ck_no_cascade_photons_um2 = min_required_photons_pcs_ck_no_cascade/A_beam
fluence_pcs_ck_no_cascade_photons_um2_windowless = min_required_photons_pcs_ck_no_cascade_windowless/A_beam
fluence_pcs_ck_cascade_photons_um2 = min_required_photons_pcs_ck_cascade/A_beam
fluence_pcs_ck_cascade_photons_um2_windowless = min_required_photons_pcs_ck_cascade_windowless/A_beam

fluence_jump_ck_photons_um2_total = min_required_photons_total_jump_ck/A_beam
fluence_jump_ck_photons_um2_total_windowless = min_required_photons_total_jump_ck_windowless/A_beam
fluence_pcs_no_ck_no_cascade_photons_um2_total = min_required_photons_total_pcs_no_ck_no_cascade/A_beam
fluence_pcs_no_ck_no_cascade_photons_um2_total_windowless = min_required_photons_total_pcs_no_ck_no_cascade_windowless/A_beam
fluence_pcs_ck_no_cascade_photons_um2_total = min_required_photons_total_pcs_ck_no_cascade/A_beam
fluence_pcs_ck_no_cascade_photons_um2_total_windowless = min_required_photons_total_pcs_ck_no_cascade_windowless/A_beam
fluence_pcs_ck_cascade_photons_um2_total = min_required_photons_total_pcs_ck_cascade/A_beam
fluence_pcs_ck_cascade_photons_um2_total_windowless = min_required_photons_total_pcs_ck_cascade_windowless/A_beam

for E in range(n_E0):
    try:
        mass_photo_cs_total_protein_pcs = xrl.CS_Photo_Total_CP(protein_matrix, E0_kev[E])
        
    except:
        pass
    
    try:
        mass_photo_cs_total_protein_jump = xrl.CS_Photo_CP(protein_matrix, E0_kev[E])
        
    except:
        continue
        
    dose_jump_ck[:, :, E] = 1e14*fluence_jump_ck_photons_um2[:, :, E]*mass_photo_cs_total_protein_jump*E0_kev[E]*cnsts.e
    dose_jump_ck_windowless[:, :, E] = 1e14*fluence_jump_ck_photons_um2_windowless[:, :, E]*mass_photo_cs_total_protein_jump*E0_kev[E]*cnsts.e
    dose_pcs_no_ck_no_cascade[:, :, E] = 1e14*fluence_pcs_no_ck_no_cascade_photons_um2[:, :, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_no_ck_no_cascade_windowless[:, :, E] = 1e14*fluence_pcs_no_ck_no_cascade_photons_um2_windowless[:, :, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_no_cascade[:, :, E] = 1e14*fluence_pcs_ck_no_cascade_photons_um2[:, :, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_no_cascade_windowless[:, :, E] = 1e14*fluence_pcs_ck_no_cascade_photons_um2_windowless[:, :, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_cascade[:, :, E] = 1e14*fluence_pcs_ck_cascade_photons_um2[:, :, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_cascade_windowless[:, :, E] = 1e14*fluence_pcs_ck_cascade_photons_um2_windowless[:, :, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e

    dose_jump_ck_total[:, E] = 1e14*fluence_jump_ck_photons_um2_total[:, E]*mass_photo_cs_total_protein_jump*E0_kev[E]*cnsts.e
    dose_jump_ck_total_windowless[:, E] = 1e14*fluence_jump_ck_photons_um2_total_windowless[:, E]*mass_photo_cs_total_protein_jump*E0_kev[E]*cnsts.e
    dose_pcs_no_ck_no_cascade_total[:, E] = 1e14*fluence_pcs_no_ck_no_cascade_photons_um2_total[:, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_no_ck_no_cascade_total_windowless[:, E] = 1e14*fluence_pcs_no_ck_no_cascade_photons_um2_total_windowless[:, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_no_cascade_total[:, E] = 1e14*fluence_pcs_ck_no_cascade_photons_um2_total[:, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_no_cascade_total_windowless[:, E] = 1e14*fluence_pcs_ck_no_cascade_photons_um2_total_windowless[:, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_cascade_total[:, E] = 1e14*fluence_pcs_ck_cascade_photons_um2_total[:, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e
    dose_pcs_ck_cascade_total_windowless[:, E] = 1e14*fluence_pcs_ck_cascade_photons_um2_total_windowless[:, E]*mass_photo_cs_total_protein_pcs*E0_kev[E]*cnsts.e