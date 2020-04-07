#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 3                                     #
# 2/11/2020                                 #    
#                                           #
#                                           #
#############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

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
#Section 3.1, 3.2, 3.3

steps = 1e-2
t = np.arange(0,20 + steps, steps)    

def f1(t):
    y = u(t-2) - u(t-9)
    return y

def f2(t):
    y = (np.exp(-t))*u(t)
    return y

def f3(t):
    y = (r(t-2)*(u(t-2) -u(t-3))) + (r(4-t)*(u(t-3) -u(t-4)))
    return y

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, f1(t))
plt.title('User Defined Funtions')
plt.grid() 
plt.ylabel('f1(t)') 

plt.subplot(3, 1, 2)
plt.plot(t, f2(t))
plt.grid() 
plt.ylabel('f2(t)')

plt.subplot(3, 1, 3)
plt.plot(t, f3(t)) 
plt.grid() 
plt.xlabel('t')
plt.ylabel('f3(t)')
plt.show()

#############################################################
#Section 4.1, 4.2, 4.3

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
                    res[i] += (Et1[i] * Et2[i-j+1])
                except:
                    print(i,j)
            
    return res

steps = 1e-2
t = np.arange(0,20 + steps, steps) 


NN = len(t)
tE = np.arange(0,2*t[NN-1],steps)
con12 = convo(f1(t),f2(t))*steps
con23 = convo(f2(t),f3(t))*steps
con13 = convo(f1(t),f3(t))*steps

con12check = sig.convolve(f1(t),f2(t))*steps
con23check = sig.convolve(f2(t),f3(t))*steps
con13check = sig.convolve(f1(t),f3(t))*steps

plt.figure(figsize = (10,15))
plt.subplot(3,1,1)
plt.plot(tE,con12, label='User-defined')
plt.plot(tE,con12check, '--', label='Built-in')
plt.ylabel('f1 * f2')
plt.xlim([0, 15])
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.title('Check with sig.convolve')

plt.subplot(3,1,2)
plt.plot(tE,con23, label='User-defined')
plt.plot(tE,con23check,'--', label='Built-in')
plt.ylabel('f2 * f3')
plt.xlim([0, 15])
plt.ylim([0, 1.2])
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(tE,con13, label='User-defined')
plt.plot(tE,con13check,'--', label='Built-in')
plt.ylabel('f1 * f3')
plt.grid()
plt.xlim([0, 15])
plt.ylim([0, 1.2])
plt.legend()
plt.show()


            
            
            
            
    
    






































