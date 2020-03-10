#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 5                                     #
# 2/25/2020                                 #    
#                                           #
#                                           #
#############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig # we want scipy.signal

# Declare values and variables
f0 = 0.25 # in Hz
w0 = 2*np.pi*f0
R = 1e3
L = 27e-3
C = 100e-9

#############################################################
# User Defined Funtions

# Step Function
def u(t):
    y = np.zeros(t.shape) #t.shape of whatever is inputted in
    for i in range(len(t)): # run the loop once for each index of t 
        if t[i] >= 0: 
            y[i] = 1 
        else:
            y[i] = 0
    return y #send back the output stored in an array

#############################################################

steps = 1e-5
t = np.arange(0,1.2e-3 + steps, steps)

num = [1/(R*C),0] # numerator
den = [1,1/(R*C),(1/np.sqrt(L*C))**2] # denomerator


tout, yout = sig.impulse((num,den), T = t)

def sine_method(R,L,C,t):
    y = np.zeros(t.shape) #t.shape of whatever is inputted in
    alpha = -1/(2*R*C)
    omega = 0.5*np.sqrt(1/(R*C))**2 - 4*(1/(np.sqrt(L*C)))**2 + (0 * 1j)
    p = alpha + omega
    g = 1/(R*C)*p
    g_mag = np.abs(g)
    g_rad = np.angle(g)
    g_deg = g_rad*(180/np.pi)
    y = ((g_mag/np.abs(omega))*np.exp(alpha*t) 
        *np.sin(np.abs(omega)*t + g_rad))*u(t)
    return y

h_t = sine_method(R,L,C,t)

plt.figure(figsize = (10,7))
plt.plot(t,h_t)
plt.xlabel('t')
plt.grid()
plt.title('h(t)')

plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('sig.impulse')

tout, yout = sig.step((num,den), T = t) # same as sig.impulse

plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('Sig.step')

#final value theroem
#








