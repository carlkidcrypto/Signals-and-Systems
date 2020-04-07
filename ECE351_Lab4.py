#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 4                                     #
# 2/18/2020                                 #    
#                                           #
#                                           #
#############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig

#############################################################
# User Defined Funtions

# Ramp funtion
def r(t):
    y = np.zeros(t.shape) #t.shape of whatever is inputted in
    for i in range(len(t)): # run the loop once for each index of t 
        if t[i] >= 0: 
            y[i] = t[i] 
        else:
            y[i] = 0
    return y #send back the output stored in an array

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
#%% Section 3:3.3
f0 = 0.25 # in Hz
w0 = 2*np.pi*f0 

def h1(t):
    y = np.exp(2*t)*u(1-t)
    return y

def h2(t):
    y = u(t-2) - u(t-6)
    return y

def h3(t):
    y = np.cos(w0*t)*u(t)
    return y


steps = 1e-2
t = np.arange(-10,10 + steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, h1(t))
plt.title('User Defined Funtions')
plt.grid() 
plt.ylabel('h1(t)') 

plt.subplot(3, 1, 2)
plt.plot(t, h2(t))
plt.grid() 
plt.ylabel('h2(t)')

plt.subplot(3, 1, 3)
plt.plot(t, h3(t)) 
plt.grid() 
plt.xlabel('t')
plt.ylabel('h3(t)')
plt.show()

#############################################################
#%% Section 4:4.3

def convo(t1,t2):
    
    t1_len = len(t1)
    t2_len = len(t2)
    
    Et1 = np.append(t1,np.zeros((1,t2_len - 1))) # extend the first function
    Et2 = np.append(t2,np.zeros((1,t1_len - 1)))# extend the first function
    
    res = np.zeros(Et1.shape) # works when we start at zero
    
    for i in range((t1_len + t2_len) - 2 ): # First Loop
        res[i] = 0
        
        for j in range(t1_len): # Second Loop
            if(i-j+1  > 0):
                try: 
                    res[i] += (Et1[j] * Et2[i-j+1])
                except:
                    print(i,j)
            
    return res

steps = 1e-2
t = np.arange(-10,10 + steps, steps) 


NN = len(t)

# works with functions that start less than 0
tE = np.arange(2*t[0],2*t[NN-1]+steps,steps)

con_h1_u = convo(h1(t),u(t))*steps
con_h2_u = convo(h2(t),u(t))*steps
con_h3_u = convo(h3(t),u(t))*steps

con_h1_ucheck = sig.convolve(h1(t),u(t))*steps
con_h2_ucheck = sig.convolve(h2(t),u(t))*steps
con_h3_ucheck = sig.convolve(h3(t),u(t))*steps

con_h1_uhand = ((0.5*np.exp(2*t))*u(1-t)) + (0.5*np.exp(2)*u(t-1))
con_h2_uhand = (r(t-2) - r(t-6))*u(t)
con_h3_uhand = ((1/w0)*(np.sin(w0*t)))*u(t)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(tE,con_h1_u, label='User-defined')
plt.plot(tE,con_h1_ucheck, '--', label='Built-in')
plt.ylabel('h1(t) * u(t)')
plt.grid()
plt.legend()
plt.title('Check with sig.convolve')

plt.subplot(3,1,2)
plt.plot(tE,con_h2_u, label='User-defined')
plt.plot(tE,con_h2_ucheck,'--', label='Built-in')
plt.ylabel('h2(t) * u(t)')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(tE,con_h3_u, label='User-defined')
plt.plot(tE,con_h3_ucheck,'--', label='Built-in')
plt.ylabel('h2(t) * u(t)')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,con_h1_uhand, 'r--', label='Hand Calculation')
plt.ylabel('h1(t) * u(t)')
plt.grid()
plt.legend()
plt.title('Hand Calculations')

plt.subplot(3,1,2)
plt.plot(t,con_h2_uhand, 'r--', label='Hand Calculation')
plt.ylabel('h2(t) * u(t)')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(t,con_h3_uhand, 'r--', label='Hand Calculation')
plt.ylabel('h2(t) * u(t)')
plt.grid()
plt.legend()
plt.show()








































