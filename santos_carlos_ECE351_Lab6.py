#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 6                                     #
# 3/3/2020                                 #    
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

# transfer funtion
num = [1,6,12] # numerator
den = [1,10,24] # denomerator


#############################################################
# Part 3.3.1 3.3.2 3.3.3  

steps = 1e-3
t = np.arange(0,2 + steps, steps)
y_t = (0.5 + np.exp(-6*t) - 0.5*np.exp(-4*t))*u(t)
tout, yout = sig.step((num,den), T = t)

plt.figure(figsize = (10,7))
plt.plot(t,y_t)
plt.xlabel('t')
plt.grid()
plt.title('Transfer Function: H(S)')

plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('Step Response: f(t)')

# H(S) * 1/s
num = [1,6,12] # numerator
den = [1,10,24,0] # denomerator

r,p,k = sig.residue(num,den)
print('\n PreLab Equation \n')
print('r:',r)
print('p:',p)
print('k:',k)

#############################################################
# Part 4.3.1 4.3.2 4.3.3

num = [25250]
den = [1,18,218,2036,9085,25250,0] # step respsonse

r,p,k = sig.residue(num,den)
print('\n Equation 1 \n')
print('r:',r)
print('p:',p)
print('k:',k)

t = np.arange(0,4.5 + steps, steps)

def cos_method(r,p,t):
    y = 0
    
    for i in range(len(r)):
        alpha = np.real(p[i]) # poles
        omega = np.imag(p[i])
        
        
        
        k_mag = np.abs(r[i]) #res
        k_rad = np.angle(r[i])
        #k_deg = k_rad*(180/np.pi)
        
        y = y + (k_mag*np.exp(alpha*t)*np.cos(omega*t + k_rad))*u(t) #Also, the factor of 2 accounts for both terms in a complex conjugate pair.

        
    return y

y_cos = cos_method(r,p,t)

plt.figure(figsize = (10,7))
plt.plot(t,y_cos)
plt.xlabel('t')
plt.grid()
plt.title('Cos Method: Equation 1')

num = [25250]
den = [1,18,218,2036,9085,25250] # step respsonse
tout, yout = sig.step((num,den), T = t)

plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('sig.response: Equation 1')






