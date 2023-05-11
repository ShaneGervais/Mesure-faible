import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.integrate import quad

wavelength = 640e-9 #nm
largeur = 10e-9 #ns
MIN  = -5*largeur
MAX = 5*largeur
N = 1000
angle_pre_selection = np.pi/4
ellipse_pre = 0
phi_x = 0
phi_y = 0
frequence = c/wavelength
print(frequence)
freq_angulaire = 2*np.pi*frequence
print(freq_angulaire)
d = 0.003 #différence de distance parcouru des deux faisceau (m)

def polarisation(theta, phi_x, phi_y):
    return np.cos(theta)*np.exp((1j)*phi_x) + np.sin(theta)*np.exp((1j)*phi_y)

def preselection(theta, phi_x, phi_y, t, z, sigma):
    def gaussien_pointer(t, z, sigma):
        return (np.sqrt(1/(np.sqrt(2*np.pi)*sigma)))*np.exp(-np.square(t - z/c)/(4*np.square(sigma)))
    
    return gaussien_pointer(t, z, sigma)*polarisation(theta, phi_x, phi_y)


time = np.linspace(MIN, MAX, N)

pre_select_state = preselection(angle_pre_selection, phi_x, phi_y, time, d, largeur)

plt.figure(1)
plt.plot(time, pre_select_state)
plt.show()

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

    def intensity_time(E_1, E_2_t):
        return (1/2)*np.conjugate(E_1)*E_1*(1+ np.real(np.conjugate(E_1)*E_2_t))
    
    #temporary***********************************************************************************************************************************************************
    MIN_FREQ = -600002/(largeur) 
    MAX_FREQ = -100000/(largeur)
    f = np.linspace(MIN_FREQ, MAX_FREQ, N)
    def intensity_freq(E_1, E_2_t, f, tau):

        E_f_1 = np.fft.fft(E_1)*np.exp(-1j*2*np.pi*f*tau)
        E_f_2 = np.fft.fft(E_2_t)*np.exp(-1j*2*np.pi*f*tau)

        return (1/2)*np.conjugate(E_f_1)*E_f_1*(1+ np.real(np.conjugate(E_f_1)*E_f_2))
    #*******************************************************************************************************************************************************************
    
    plt.figure(3)
    plt.plot(t, intensity_time(E_1, E_2_t))
    plt.show()

    plt.figure(4)
    plt.plot(f, intensity_freq(E_1, E_2_t, f, tau))
    plt.show()

    return E_1 + E_2_t

delay = d/c
print(delay)

E_t = michelson(pre_select_state, delay*1, largeur, freq_angulaire, time, 0)
plt.figure(5)
plt.plot(time, E_t)
plt.show()

def postselection(weakened_state, theta, phi_x, phi_y):
    post_sel_state = polarisation(theta, phi_x, phi_y)

    return post_sel_state*weakened_state

angle_post_selection = np.pi/4
ellipse_post = 0
phi_x_post = 0
phi_y_post = np.pi/2

E_t = postselection(E_t, angle_post_selection, phi_x_post, phi_y_post)

plt.figure(6)
plt.plot(time, E_t)
plt.show()

def weak_value_real(E_t, t):
    return np.real(np.trapz(np.conjugate(E_t)*t*E_t, t))

t_moy = weak_value_real(E_t, time)
print(t_moy)

MIN_WAVE = 405e-9
MAX_WAVE = 980e-9

MIN_FREQ = 1/(MIN)
MAX_FREQ = 1/(MAX)

f = np.linspace(MIN_FREQ, MAX_FREQ, N)

def weak_value_imag(E_t: list, f, tau):
    E_f = np.fft.fft(E_t)*np.exp(-1j*2*np.pi*f*tau)
    return np.imag(np.trapz(np.conjugate(E_f)*(f)*E_f, f))

f_moy = weak_value_imag(E_t, f, delay)

print(2*np.pi/largeur)
print(f_moy)

#**********************************************************************************************************************************************************************
def frequency_domain(E_t, f, tau): #needs work
    return np.fft.fft(E_t)*np.exp(-1j*2*np.pi*f*tau)

plt.figure(5)
plt.plot(f, frequency_domain(E_t, f, delay))
plt.show()
#**********************************************************************************************************************************************************************


#plt.figure(5)
#plt.plot(time, t_moy)
#plt.show()

#weak_value_real = np.trapz(t_moy, time)
#print(weak_value_real)
#I = quad(t_moy, -np.inf, np.inf)
#weak_value_real = I
#print(weak_value_real)
