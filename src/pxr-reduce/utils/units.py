
import numpy as np
from scipy import constants

# Constants used for data reduction
sol = constants.c # [m/s] Speed of light in vacuum
planck_joule = constants.h # 6.6267015e-34  # [Joule s] # Plancks constant
elemcharge = constants.e # 1.602176634e-19  # [coulombs] # Charge of electron
planck = planck_joule / elemcharge  # [eV s]
metertoang = 10 ** 10 # Convert meters to Angstroms


# Functions used to convert between units used in XRR data

def energy_to_wavelength(val):
    """
    Converts energy to wavelength or wavelength to energy depending on input.
    
    Parameters
    -----------
    val : float
        Energy [eV] or Wavelength [Angstroms]
        
    Returns
    --------
    out : float
        Wavelength [Angstroms] or Energy [eV]                
    """
    
    return metertoang * planck * sol / val  # calculate wavelength [A] to Energy [eV]

def theta_to_q(theta, lam):
    """
    Converts a scattering angle to the momentum transfer vector.

    Parameters
    -----------
    theta : float
        Reflection angle [Radians]
    lam : float
        Wavelength [Angstroms] 
        
    Returns
    --------
    out : float
        q-vector (Angstrom^-1)
    """
    
    return 4 * np.pi * np.sin(theta)/lam
    
def q_to_theta(q, lam):
    """
    Converts the momentum transfer vector to the scattering angle.

    Parameters
    -----------
    q : float
        q-vector [Angstrom^-1]
    lam : float
        Wavelength [Angstroms] 
        
    Returns
    --------
    out : float
        Reflection Angle [Radians]
    """
    
    return np.rad2deg(np.arcsin(lam * q / (4 * np.pi)))