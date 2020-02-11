#############################################
#                                           #
# Carlos Santos                             #    
# ECE 351-51                                #
# Lab 2                                     #
# 2/4/2020                                 #    
#                                           #
#                                           #
#############################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # Set font size in plots
steps = 1e-2 # Define step size
t = np.arange(0,5 + steps, steps) # Add a step size to make sure the 
# plot includes 5.0. Since np.arange() only 9
# goes up to, but doesn’t include the 10 
# value of the second argument

print('Number of elements: len(t) = ' , len(t), '\nFirst Element: t[0] = ', 
      t[0], ' \nLast Element: t[len(t)-1] = ', t[len(t)-1])

# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the 
# first and last index of the array. Notice the array goes from 0 to len() - 1

# --- User - Defined Function --

# Create output y(t) using a for loop and if/else statements 
def example1(t): # The only variable sent to the function is t 
    y = np.zeros(t.shape)  # initialze y(t) as an array of zeros
    
    for i in range(len(t)): # run the loop once for each index of t 
        if i < (len(t) + 1)/3: 
            y[i] = t[i]**2
        else:
            y[i] = np.sin(5*t[i]) + 2
    return y #send back the output stored in an array
y = example1(t) # call the function we just created

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y) 
plt.grid()
plt.ylabel('y(t) with Good Resolution') 
plt.title('Background - Illustration of for Loops and if/else Statements')

t = np.arange(0, 5 + 0.25, 0.25) # redefine t with poor resolution
y = example1(t)

plt.subplot(2, 1, 2)
plt.plot(t, y)  
plt.grid() 
plt.ylabel('y(t) with Poor Resolution') 
plt.xlabel('t') 
plt.show()


#########################################################################
# Part 4.3.1 and 4.3.2
steps_func1 = .01
t_func1 = np.arange(0,10 + steps_func1, steps_func1)

print('Number of elements: len(t_func1) = ' , len(t_func1), '\nFirst Element: t_func1[0] = ', 
      t_func1[0], ' \nLast Element: t_func1[len(t_func1)-1] = ', t_func1[len(t_func1)-1])

def func1(t_func1):
    x = np.cos(t_func1)
    return x

x = func1(t_func1)
    
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_func1, x)
plt.title('Cosine Function')  
plt.grid() 
plt.xlabel('t')
plt.show()

#########################################################################
# part 5.3.1 and 5.3.2
# Funtion to plot: y(t) r(t) - r(t-3) + 5u(t-3) - 2u(t-6) - 2r(t-6)
steps_u = 1e-4
t_u = np.arange(-1,1 + steps_u, steps_u)

def u(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)): # run the loop once for each index of t 
        if t[i] >= 0: 
            y[i] = 1 
        else:
            y[i] = 0
    return y #send back the output stored in an array


x = u(t_u)
    
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_u, x)
plt.title('Step Function') 
plt.grid() 
plt.xlabel('t')
plt.show()



steps_r = 1e-4
t_r = np.arange(-1,1 + steps_r, steps_r)

def r(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)): # run the loop once for each index of t 
        if t[i] >= 0: 
            y[i] = t[i] 
        else:
            y[i] = 0
    return y #send back the output stored in an array

x = r(t_r)
    
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_r, x)
plt.title('Ramp Function') 
plt.grid() 
plt.xlabel('t')
plt.show()

steps_eqt = 1e-4
t_eqt = np.arange(-4,10 + steps_eqt, steps_eqt)
x = r(t_eqt) - r(t_eqt-3) + 5*u(t_eqt-3) - 2*u(t_eqt-6) - 2*r(t_eqt-6)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_eqt, x) 
plt.grid() 
plt.title('Plotted Function')
plt.xlabel('t')
plt.show()



#r(t) - r(t-3) + 5u(t-3) - 2u(t-6) - 2r(t-6)
   
#########################################################################
# Part 6.3.1, 6.3.2, 6.3.3, 6.3.4, and 6.3.5

steps_func2 = 1e-3
t_func2 = np.arange(-5,10 + steps_func2, steps_func2)

def func2(t):
    y = (r(t) - r(t-3) + 5*u(t-3) - 2*u(t-6) - 2*r(t-6))
    return y

y = func2(t_func2)
dt = np.diff(t_func2)
dy = np.diff(y, axis = 0)/dt

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t_func2, y, '--', label = 'y(t)')
plt.plot(t_func2[range(len(dy))], dy[:,0], label = 'dy/dt') # dy[:,0] might not work. Take out if needed
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Plotted Derivative Function')
plt.legend()
plt.xlim([0,15])
plt.ylim([-2,10])
plt.show()



y = func2(-t_func2)

plt.figure(figsize = (10,7))
plt.plot(t_func2, y, '--', label = 'f(-t)')
plt.grid()
plt.title('time reversal')
plt.show()

y = func2(t_func2 - 4)

plt.figure(figsize = (10,7))
plt.plot(t_func2, y, '--', label = 'f(t-4)')
plt.grid()
plt.title('time shift f(t-4)')
plt.show()

y = func2(-t_func2-4)

plt.figure(figsize = (10,7))
plt.plot(t_func2, y, '--', label = 'f(-t-4)')
plt.grid()
plt.title('time shift f(-t-4)')
plt.show()

y = func2(t_func2/2)

plt.figure(figsize = (10,7))
plt.plot(t_func2, y, '--', label = 'f(t/2)')
plt.grid()
plt.title('time scale f(t/2)')
plt.show()

y = func2(2*t_func2)

plt.figure(figsize = (10,7))
plt.plot(t_func2, y, '--', label = 'f(2t)')
plt.grid()
plt.title('time scale f(2t)')
plt.show()











