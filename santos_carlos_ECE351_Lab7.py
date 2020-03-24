#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 7                                     #
# 03/10/2020                                 #    
#                                           #
#                                           #
#############################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% 3.3.1 , 3.3.2, 3.3.3, 3.3.4, 3.3.5
# Equation G(s) Transfer Function
Gnum = [1,9] # numerator
Gden = [1,-2,-40,-64] # denomerator
# Equation A(s)
Anum = [1,4] # numerator
Aden = [1,4,3] # denomerator
# Equation B(s)
Bnum = [1,26,168] # numerator

# Equation G(s) Transfer Function
rG,pG,kG = sig.tf2zpk(Gnum,Gden)
print('Equation G(s)')
print('rG:',rG)
print('pG:',pG)
print('kG:',kG)
print('\n')

# Equation A(s)
rA,pA,kA = sig.tf2zpk(Anum,Aden)
print('Equation A(s)')
print('rA:',rA)
print('pA:',pA)
print('kA:',kA)
print('\n')

# Equation B(s)
rB = np.roots(Bnum)
print('Equation B(s)')
print('rB:',rB)
print('\n')

#open loop function H(s) = A(s) * G(s)
print('Equation H(s) = A(s) * G(s)')
Hnum = sig.convolve(Anum,Gnum)
print('Hnum = ',Hnum)
Hden = sig.convolve(Aden,Gden)
print('Hden = ',Hden)
print('\n')

#open loop step response X(s) = H(s) * (1/s)
print('Open Loop Equation X(s) = H(s) * 1/s')
Xnum = sig.convolve(Hnum,[1])
print('Xnum = ',Xnum)
Xden = sig.convolve(Hden,[1,0])
print('Xden = ',Xden)
print('\n')

#plot step response
steps = 1e-3
t = np.arange(0,2 + steps, steps)
tout, yout = sig.step((Xnum,Xden), T = t)

plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('Open Loop Step Response:')

#%% 4.3.1, 4.3.2, 4.3.3, 4.3.4, 4.3.5

#closed loop function H(s) = G(s)A(s) / 1 + G(s)B(s)
print('Closed Loop Equation X(s) = G(s)*A(s) / 1 + G(s)*B(s)')


Hnum = sig.convolve(Gnum,Anum)
print('Hnum = ',Hnum)
Hden = sig.convolve(Gden + sig.convolve(Bnum,Gnum),Aden)
print('Hden = ',Hden)

tout, yout = sig.step((Hnum,Hden), T = t)
plt.figure(figsize = (10,7))
plt.plot(tout,yout)
plt.xlabel('t')
plt.grid()
plt.title('Closed Loop Step Response')


  








 






