"""
Ice thickness coomputation for infrared camera data as implemented by Yu and Rothrock in 1996 (for satellite data)

This file defines many constants and some parametrizations to be used when computing the ice thickness from brightness temperature measurements.

The calc_h_i function is the main method. It computes all the appropriate fluxes and finds the necessary ice thickness so that the sum of fluxes is 0.
It computes the sensible heat flux, latent heat flux, the outgoing longwave radiation (gray body radiation) and the ingoing longwave and shortwave radiation.
The remaining energy is assumed to be conducted through the ice and the ice thickness is found.

A helper function calc_from_dship_pino reads in from a sequence of images and time series from dship and pinocchio, getting the appropriate values.

Written by Damien Ringeisen, Pascal Bourgault and Dmitrii Murashkin
2018, Lomonossov Ridge aboard the FS Polarstern
"""

import numpy as np
from copy import copy
import datetime as dt

## Constants 
sigma = 5.67e-8 # Stefan-Boltzman constant [W / m^2 K^4]

# In Maekynen et al. 2013
epsilon = 0.95 # Emissivity of ice and snow []
k_s= 0.3 # Heat conductivity snow [W / m K]
alpha = 0.8 # Sea ice albedo
rho_a = 1.03 # air density [kg / m^3]
c_p = 1.0044e3 # Specific heat of the air [J / kg m^3]
C_s = 0.003 # Bulk transfer coefficients for heat (0.003 for very thin ice, 0.00175 for thicker ice) []
L = 2.49e6 # Latent heat of vaporization [J / kg]
C_e = C_s # Bulk transfer coefficients for evaporation (same as C_s)

# in Yu and Rothrock 1996
beta=0.13 # parameter [W / m^2 kg ]
k_0=2.034 # Heat conductivity pure ice [W / m K]
S_i = 7.7 # Bulk salinity of ice [ppt] Value from Maekynen
p0=1012


# Used by Yu and Rothrock 1996
def e_s(T):
    """Parametrization of the saturation vapor pressure."""
    a = 2.7798202e-6
    b = -2.6913393e-3
    c = 0.97920849
    d = -158.63779
    e = 9653.1925
    return a * T**4 + b * T**3 + c * T**2 + d * T + e


# also in Yu and Rothrock 1996
def atm_F_ldn(temp, coverage=1):
    """Parametrization of the atmospheric emissivity and computation of the downwelling longwave radiation fron cloud temperature and coverage."""
    espilon_star = 0.7855 * (1 + 0.2232 * coverage**2.75)
    return espilon_star * sigma * (temp)**4.  # Longwave radiation down as parametrized from Yu and Rothrock 1996 (between 3 and 4)

    
# Taken from Yu and Rothrock, cited as from Doronin (1971)
def snow_thickness_doronin(h_i):
    b = np.zeros(np.shape(h_i))
    b[h_i < 0.05] = 0
    b[(h_i <= 0.2) * (h_i >= 0.05)] = 0.05 
    b[ h_i > 0.2 ] = 0.1
    return b * h_i
    

def calc_h_i(T_s, T_w=None, T_a=0, h_s=0, rh=0.9, u=0, S_w=0, F_ldn=0, F_sdn=0, C=0, nbr_it=5):
    """Calculation of the sea ice thickness.
    
    Args:
    T_s.  - Snow/ice surface temperature [degC] (Must be given)
    T_w   - Water temperature [degC] None if not known, then the freezing temperature for the given salinity is used.
    T_a   - Air temperature (2 meters above ground) [degC]
    u     - Surface wind 2 meters above ground, u=0.34*u_g (u_g geostrophic wind) [m/s]
    h_s   - Snow hickness [m], if None, uses Doronin (1971)'s parametrization like in Yu and Rothrock.
    rh    - Relative humidity (0.90) []
    S_w   - Salinity of water [ppt]. Used to get water freezing temperature if T_w is None.
    F_ldn - Longwave radiation down. If None, uses F_ldn = eps* sigma T_a^4 (Yu and Rothrock 1996). [W / m^2]
    F_sdn - Shortwave radiation down [W / m^2] 
    C     - Fractional cloud cover for the LW Down parametrization (Yu and Rothrock 1996) []
    nbr_it - If h_s is None, number of iterations to be done for Doronin's parametrization.
    All values default to 0, except for T_w (freezing) and rh (0.9).
    
    Returns:
        h_i  - Ice thickness array, with the same shape as T_s.
        
        If h_s was None, then Doronin's parametrization was used and the following is returned:
            * final snow thickness
            * list containing maximum residuals for each iteration.
        if it was a scalar, None is returned twice.
    """

    if T_w is None :
        T_w=-0.054 * S_w # Freezing Temperature of sea water [K]

    # Transform in Kelvin
    T_s=T_s+273.15 # [K]
    T_a=T_a+273.15 # [K]
    T_w=T_w+273.15 # [K]

    # Parameter-dependent variables form Yu and Rothrock 1996
    T_i = T_s # slab temperature [K]
    k_i= k_0 + beta * S_i / (T_i - 273) # Heat conductivity of sea ice (Yu and Rothrock 1996 (2.034 in Maykut 1982)

    F_lu = - sigma * T_s**4 # Longwave radiation up [W]

    if F_ldn is None:
        F_ldn = atm_F_ldn(T_a, C)

    F_s = rho_a * c_p * C_s * u * (T_a - T_s) # Turbulent sensible heat flux [W]
    
    e_sa = e_s(T_a) # Saturation vapor pressure in the air [mbar]
    e_s0 = e_s(T_s) # Saturation vapor pressure at the surface [mbar]
    F_e = rho_a * L * C_e * u * (rh * e_sa - e_s0)/p0 # Latent heat flux [W]


    F_sdn = (1 - alpha) * F_sdn # Absorbed shortwave radiation [W]

    # Sum of fluxes
    F_t = F_s + F_e + F_lu + F_ldn + F_sdn
        
    if h_s is None:
        # We use Doronin (1971)'s parametrization for snow thickness
        h_i = k_i * (T_s - T_w) / F_t  # Ice thickness with no snow
        if not isinstance(h_i, np.ndarray):
            h_i = np.array(h_i)
        
        h_res = np.zeros((nbr_it,))
        for i in range(nbr_it):
            # We compute the parametrized snow thickness from the previous ice thickness and update h_i
            h_old = h_i.copy()
            snow_thickness_approximation = snow_thickness_doronin(h_i)
            h_i = k_i * ((T_s - T_w) / F_t - (snow_thickness_approximation / k_s))
            h_res[i] = np.max(np.abs(h_i - h_old))
            if h_res[i] < 1e-3:
                break
        return h_i, snow_thickness_approximation, h_res

    return k_i * ((T_s - T_w) / F_t - (h_s / k_s)), None, None # Ice Thickness [m]


def calc_from_dship_pino(temp, dship, pino, frame_num, times, h_s=0, time_range=10):
    """Calculate the ice thickness from a temperature sequence, dship and pinocchio data.
    
    Args:
    temp       - The 3D (t, y, x) sequence of temperature images.
    dship      - A panda Dataframe read by utils.read_dship() with the needed measurements:
                    air_temperature, water_temperature, global_radiation, rel_humidity and true_wind_velocity
    pino       - A panda Dataframe read by utils.read_pino(), with the needed measurements:
                    coverage and temperature
    frame_num  - A integer, the index of the frame of temp to use.
    times      - A list of the datetime of each frame
    time_range - A integer. The time range to use when averaging.
    
    All values from  dship and pino will be averaged around time[frame_num] +/- time_range.
    Uses the atm_F_ldn parametrization with pinocchio data for the downwelling longwave.
    Returns the icethickness in meters as computed by calc_h_i
    """
    time_delta = dt.timedelta(minutes=time_range)
    
    actd = dship.loc[times[frame_num] - time_delta : times[frame_num] + time_delta].mean()
    actp = pino.loc[times[frame_num] - time_delta : times[frame_num] + time_delta].mean()
    return calc_h_i(temp[frame_num], T_a=actd.air_temperature, T_w=actd.water_temperature,
                    F_ldn=atm_F_ldn(actp.temperature, actp.coverage),  F_sdn=actd.global_radiation,
                    rh=actd.rel_humidity / 100, u=actd.true_wind_velocity, h_s=h_s)


if __name__=='__main__':
    
    u = 2 # Surface wind 2 meters above ground, u=0.34*u_g (u_g geostrophic wind) [m/s]
#    F_ldn= 235 # Longwave radiation down [W / m^2]
    F_ldn = None
    F_sdn = 0 # Shortwave radiation down [W / m^2] 
    h_s= None  # Thickness snow [m]
    f = 0.90 # relative humidity (90%) []
    T_a = -3 # Air temperature [degC]
    S_w = 31 # Salinity of water [ppt]
    T_w = -1.8  # Water temperature [degC] None if not known
    T_s= -6 # np.arange(1.3*T_a,T_w,0.01)  # Snow/ice surface temperature [degC]
    
    h_i, snow, residuals = calc_h_i(T_s, u=u, F_ldn=F_ldn, F_sdn=F_sdn, h_s=h_s, rh=f, T_a=T_a, S_w=S_w, T_w=T_w)

    print(h_i)










