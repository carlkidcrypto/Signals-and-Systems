#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 8                                     #
# 3/24/2020                                 #    
#                                           #
#                                           #
#############################################


import numpy as np
import matplotlib.pyplot as plt

#%% Task 1
def a(k):
    a = (2*np.sin(np.pi*k)-np.sin(2*np.pi*k))/(np.pi*k)
    return np.round(a,2)

def b(k):
    b = (-2*np.cos(np.pi*k) + np.cos(2*np.pi*k) + 1)/(np.pi*k)
    return b

def Fourier_approx(n,t,T):
    x=0
    for i in range(1,n+1):
        x = x + b(i)*np.sin(i*(2*np.pi/T)*t)
    return x

n = np.arange(0,5)
ak = a(n)
bk = b(n)

print("a_0 = ", ak[0])
print("a_1 = ", ak[1])
print("b_1 = ", bk[1])
print("b_2 = ", bk[2])
print("b_3 = ", bk[3])

#%% Task 2

steps = 1e-3
t = np.arange(0,20+steps,steps)


f1 = Fourier_approx(1,t,8)
f3 = Fourier_approx(3,t,8)
f15 = Fourier_approx(15,t,8)
f50 = Fourier_approx(50,t,8)
f150 = Fourier_approx(150,t,8)
f1500 = Fourier_approx(1500,t,8)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, f1)
plt.title('Fourier Series Approximations of Square Wave')
plt.grid() 
plt.ylabel('N = 1') 

plt.subplot(3, 1, 2)
plt.plot(t, f3)
plt.grid() 
plt.ylabel('N = 3')

plt.subplot(3, 1, 3)
plt.plot(t, f15) 
plt.grid() 
plt.xlabel('t')
plt.ylabel('N = 15')
plt.show()

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, f50)
plt.title('Fourier Series Approximations of Square Wave')
plt.grid() 
plt.ylabel('N = 50') 

plt.subplot(3, 1, 2)
plt.plot(t, f150)
plt.grid() 
plt.ylabel('N = 150')

plt.subplot(3, 1, 3)
plt.plot(t, f1500) 
plt.grid() 
plt.xlabel('t')
plt.ylabel('N = 1500')
plt.show()

