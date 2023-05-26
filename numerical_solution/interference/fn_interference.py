import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.integrate import quad


#*******************************************************--POLARISATION--*************************************************************************************
#general polarisation state
def polarisation(theta, phi_x, phi_y):
    return np.cos(theta)*np.exp((1j)*phi_x) + np.sin(theta)*np.exp((1j)*phi_y)

#general polarisation state with ellipticity
def etat_polarisation_ellipse(theta, epsilon):
    J_x = (np.cos(theta)*np.cos(epsilon)+1j*np.sin(theta)*np.sin(epsilon))
    J_y = np.sin(theta)*np.cos(epsilon)-1j*np.cos(theta)*np.sin(epsilon)
    phi = np.angle(J_x)
    return (J_x + J_y)*np.exp(-1j*phi)

def gaussien_pointer(t, z, sigma):
    return (np.sqrt(1/(np.sqrt(2*np.pi)*sigma)))*np.exp(-np.square(t - z/c)/(4*np.square(sigma)))

def electric_field_wavefn(t, z, sigma, wavelength):
    freq = c/wavelength
    w_0 = 2*np.pi*freq
    k = 1/wavelength
    return gaussien_pointer(t, z, sigma)*np.exp(1j*(k*z - w_0*t))

def intensity(champ):
    return np.conjugate(champ)*champ

def g_1(E_0, E_1):
    G_1_tau = np.conjugate(E_0)*E_1
    G_1_0 = np.conjugate(E_0)*E_0
    return G_1_tau/G_1_0

def michelson(state, tau, sigma, wavelength, t, z):
    
    freq = c/wavelength
    freq_ang = 2*np.pi*freq

    def beamsplit(state):
        E_1 = (1/np.sqrt(2))*state
        E_2 = (1/np.sqrt(2))*state
        return E_1, E_2
    
    E_1, E_2 = beamsplit(state)

    def weak_measurement(split_state, tau, sigma, w, t, z):
        delta = (tau*(4*w*c*sigma**2 - (1j)*c*(tau + 2*t) + 2*(1j)*z))/(4*c*sigma**2)
        U = np.exp((1j)*delta)
        return np.abs(U*split_state)
    
    E_2_t = weak_measurement(E_2, tau, sigma, freq_ang, t, z)

    E_t = E_1 + E_2_t

    I_t = intensity(E_t)

    return I_t, E_1, E_2_t

#*******************************************************--PRESELECTION--*************************************************************************************
#***********************************************************************************************************************************************************

#Represents the preselected state as a gaussian envelope
def preselection(theta, phi_x, phi_y, t, z, sigma, wavelength):
    return polarisation(theta, phi_x, phi_y)*electric_field_wavefn(t,z,sigma,wavelength)



            