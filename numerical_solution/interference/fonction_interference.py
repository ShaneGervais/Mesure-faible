import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import fonction_polarisation as fp

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

#*******************************************************--PRESELECTION--*************************************************************************************
#***********************************************************************************************************************************************************

#Represents the preselected state as a gaussian envelope
def preselection(theta, phi_x, phi_y, t, z, sigma):
    def gaussien_pointer(t, z, sigma):
        return (np.sqrt(1/(np.sqrt(2*np.pi)*sigma)))*np.exp(-np.square(t - z/c)/(4*np.square(sigma)))
    
    return gaussien_pointer(t, z, sigma)*polarisation(theta, phi_x, phi_y)

#Represents the preselected state as a gaussian function
def preselection_N(theta, phi_x, phi_y, sigma, t, z, N):

    def gaussien_pointer_N(sigma, t, z, N):
        A_t = np.zeros(N)

        for i in range(len(A_t)):
            A_t[i] = (np.sqrt(1/(np.sqrt(2*np.pi)*sigma)))*np.exp(-np.square(t[i] - z/c)/(4*np.square(sigma)))
        
        return A_t

    A = gaussien_pointer_N(sigma, t, z, N)
    #champ électrique
    E = np.zeros(len(t))
    
    for i in range(len(t)):
        E[i] = A[i]*polarisation(theta, phi_x, phi_y)#*np.exp(1j*w*t[i])
    return E

#*******************************************************--PRESELECTION IN 2D--*************************************************************************************
#***********************************************************************************************************************************************************
def preselection_2D(theta, phi_x, phi_y, sigma, t, z):
    def gaussien_pointer(t, z, sigma):
        return (np.sqrt(1/(np.sqrt(2*np.pi)*sigma)))*np.exp(-np.square(t - z/c)/(4*np.square(sigma)))
    
    J = fp.j_theta_phi(theta, phi_x, phi_y)
    E = np.zeros([len(J), len(t)])
    for i in range(len(J)):
        for j in range(len(t)):
            E[i,j] = J[i]*gaussien_pointer(t, z, sigma)

    return E
    #return gaussien_pointer(t, z, sigma)*polarisation(theta, phi_x, phi_y)


#Preselected state based as a vector in N steps
def preselection_N_2D(theta, phi_x, phi_y, w, sigma, t, t_0, N):
    
    def fonction_gaussien(sigma, t, t_0, N):
        A_t = np.zeros(N)

        for i in range(len(A_t)):
            A_t[i] = (np.sqrt(1/((np.sqrt(2*np.pi))*sigma)))*np.exp(-((t[i] -t_0)**2)/(4*sigma**2))
        
        return A_t
    
    #vecteur de jones
    J = fp.j_theta_phi(theta, phi_x, phi_y)
    A = fonction_gaussien(sigma, t, t_0, N)
    #champ électrique
    E = np.zeros([len(J), len(t)])
    
    for i in range(len(J)):
        for j in range(len(t)):
            E[i,j] = A[j]*J[i]*np.exp(1j*w*t[j])

    return E

#***************************************--MICHELSON INTERFEROMETRY USING WEAK MEASUREMENTS--****************************************************************
#***********************************************************************************************************************************************************

def michelson(state, tau, sigma, w, t, z):

    def beamsplit(state):
        E_1 = (1/np.sqrt(2))*state
        E_2 = (1/np.sqrt(2))*state
        return E_1, E_2
    
    E_1, E_2 = beamsplit(state)

    def weak_measurement(split_state, tau, sigma, w, t, z):
        delta = (tau*(4*w*c*sigma**2 + (1j)*c*(tau+2*t) - 2*(1j)*z))/(4*c*sigma**2)
        U = np.exp((1j)*delta)
        return np.abs(U*split_state)
    
    E_2_t = weak_measurement(E_2, tau, sigma, w, t, z)

    plt.figure(2)
    plt.plot(t, E_1)
    plt.plot(t, E_2_t)
    plt.show()

    return E_1 + E_2_t

#*******************************************************--POSTSELECTION--*************************************************************************************
#***********************************************************************************************************************************************************


def postselection(weakened_state, theta, phi_x, phi_y):
    post_sel_state = polarisation(theta, phi_x, phi_y)

    return post_sel_state*weakened_state

#***********************************************************************************************************************************************************

def intensity(fonction):
    I = np.zeros(shape=(len(fonction)))
    
    for i in range(len(fonction)):
        I[i] = np.abs(np.dot(np.conjugate(fonction[i]), fonction[i]))
    
    return I

def intensity_2D(fonction, m):
    I = np.zeros(m)

    for i in range(m):
        I[i] = np.abs(np.dot(np.conjugate(fonction[0, i]), fonction[0, i]))

    return I
