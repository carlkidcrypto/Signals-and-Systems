#############################################
# University of Idaho                       #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 8                                     #
# 3/24/2020                                 #    
#                                           #
#                                           #
#############################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

#%% Lab 8 Functions
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


#%% User Defined Function
def my_fft(x, fs):
    N = len(x)
    X_fft = fft.fft(x)
    X_fft_shifted = fft.fftshift(X_fft)
    
    freq = np.arange(-N/2,N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return X_mag, X_phi, freq # return all three items

def my_fft_edited(x,fs):
    N = len(x)
    X_fft = fft.fft(x)
    X_fft_shifted = fft.fftshift(X_fft)
    
    freq = np.arange(-N/2,N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(N): # lets loop through it and find what we are looking for
        if (np.abs(X_mag[i]) < 1e-10):
            X_phi[i] = 0
            
    return X_mag, X_phi, freq # return all three items
    
    
#%% Task 1,2,3 Functions
    
def f1(t):
    return np.cos(2*np.pi*t)

def f2(t):
    return 5*np.sin(2*np.pi*t)

def f3(t):
    x=2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t)+3))**2
    return x

#%% Plots for Task 1, 2, 3
steps = 1e-2
t = np.arange(0,2,steps) 
    
x = f1(t)
lower = -2
upper = 2
fs = 1000000

X_mag, X_phi, freq = my_fft(x, fs)
plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,x)
plt.title('FFT of  f1(t)')
plt.ylabel('f1(t)')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

x = f2(t)
lower = -2
upper = 2

X_mag, X_phi, freq = my_fft(x, fs)
plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,x)
plt.title('FFT of  f2(t)')
plt.ylabel('f2(t)')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

x = f3(t)
lower = -15
upper = 15

X_mag, X_phi, freq = my_fft(x, fs)
plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,x)
plt.title('FFT of  f3(t)')
plt.ylabel('f3(t)')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

#%% Task 4

x = f1(t)
lower = -2
upper = 2

X_mag, X_phi, freq = my_fft_edited(x, fs)
plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,x)
plt.title('FFT Edited of  f1(t)')
plt.ylabel('f1(t)')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

x = f2(t)
lower = -2
upper = 2

X_mag, X_phi, freq = my_fft_edited(x, fs)
plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,x)
plt.title('FFT Edited of  f2(t)')
plt.ylabel('f2(t)')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

x = f3(t)
lower = -15
upper = 15

X_mag, X_phi, freq = my_fft_edited(x, fs)
plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,x)
plt.title('FFT Edited of  f3(t)')
plt.ylabel('f3(t)')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

#%% Task 5

steps = 1e-2
t = np.arange(0,16,steps) 
y = f15 = Fourier_approx(15,t,8)

X_mag, X_phi, freq = my_fft_edited(y, fs)
lower = -2
upper = 2

plt.figure(figsize=(10,7))
plt.subplot(3,1,1) #Top Figure
plt.plot(t,y)
plt.title('N = 15:  Square Wave')
plt.ylabel('N = 15')
plt.xlabel("t (s)")

plt.subplot(3,2,3) #Mag 1
plt.stem(freq, X_mag)
plt.ylabel("Magnitude")

plt.subplot(3,2,4)#Mag 2
plt.stem(freq, X_mag)
plt.xlim([lower,upper])

plt.subplot(3,2,5)#Phase 1
plt.stem(freq, X_phi)
plt.ylabel("Phase")
plt.xlabel("Freq. (Hz)")

plt.subplot(3,2,6)#Phase 2
plt.stem(freq, X_phi)
plt.xlim([lower,upper])
plt.xlabel("Freq. (Hz)")
plt.show()

 

