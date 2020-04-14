#############################################
# University of Idaho                       #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 10                                    #
# 4/8/2020                                  #    
#                                           #
#                                           #
#############################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

# RLC Circuit Parameters
R = 1000
L = 27e-3
C = 100e-9


#%% 3.3.1, 3.3.2, 3.3.3
steps = 1e3
omega = np.arange(1e3,1e6+steps,steps)

mag_H = 20*np.log10((omega/(R*C))/(np.sqrt(((1/(L*C))-omega**2)**2 + (omega/(R*C))**2)))
phase_H = ((np.pi/2)-np.arctan((omega/(R*C))/(-omega**2 + 1/(L*C)))) * 180/np.pi # phase is in degrees

for i in range(len(phase_H)):
    if (phase_H[i] > 90):
        phase_H[i] = phase_H[i] - 180



plt.figure(figsize=(10,7))
plt.subplot(2,1,1) #Top Figure
plt.ylabel("Mag")
plt.grid()
plt.semilogx(omega, mag_H)
plt.title("hand solved")
plt.subplot(2,1,2) # Bottom Figure
plt.xlabel("w")
plt.ylabel("Phase")
plt.grid()
plt.semilogx(omega, phase_H)
plt.show()

num_H = [1/(R*C),0]
den_H = [1,1/(R*C),1/(L*C)]
sys = sig.TransferFunction(num_H,den_H)
ang,mag,phase = sig.bode(sys,omega)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1) #Top Figure
plt.ylabel("Mag")
plt.grid()
plt.semilogx(ang, mag)
plt.title("sig.bode()")
plt.subplot(2,1,2) # Bottom  Figure
plt.xlabel("w")
plt.ylabel("Phase")
plt.grid()
plt.semilogx(ang, phase)
plt.show()

sys = con.TransferFunction(num_H,den_H)
_ = con.bode(sys, omega, dB=True,Hz=True, deg=True, Plot=True)

#%% 4.4.1, 4.4.2, 4.4.3, 4.4.4


steps = 1e-9
t = np.arange(0, 0.01+steps, steps)
x = (np.cos(2*np.pi*100*t)+ np.cos(2*np.pi*3024*t) + np.sin(2*np.pi* 50e3*t))

num_Z, den_Z = sig.bilinear(num_H, den_H, 1/steps)
xFiltered = sig.lfilter(num_Z, den_Z, x)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1) #Top Figure
plt.grid()
plt.plot(t,x)
plt.title("Unfiltered Signal")
plt.subplot(2,1,2)
plt.plot(t,xFiltered)
plt.title("Filtered Signal")
plt.grid()
plt.show()

