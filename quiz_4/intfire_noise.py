from __future__ import print_function
"""
Created on Wed Apr 22 16:02:53 2015

Basic integrate-and-fire neuron 
R Rao 2007

translated to Python by rkp 2015
"""

import numpy as np
import matplotlib.pyplot as plt


# input current
I = 1 # nA

# capacitance and leak resistance
C = 1 # nF
R = 40 # M ohms

# I & F implementation dV/dt = - V/RC + I/C
# Using h = 1 ms step size, Euler method

differences_array = [] 
for nise in np.linspace(0, 5, 1000):
	V = 0
	tstop = 1000
	abs_ref = 5 # absolute refractory period 
	ref = 0 # absolute refractory period counter
	V_trace = []  # voltage trace for plotting
	V_th = 10 # spike threshold
	spiketimes = [] # list of spike times

# input current
	noiseamp = nise # amplitude of added noise
	I += noiseamp*np.random.normal(0, 1, (tstop,)) # nA; Gaussian noise

	for t in range(tstop):
	  
	   if not ref:
	       V = V - (V/(R*C)) + (I[t]/C)
	   else:
	       ref -= 1
	       V = 0.2 * V_th # reset voltage
	   
	   if V > V_th:
	       V = 50 # emit spike
	       ref = abs_ref # set refractory counter

	   V_trace += [V]

	for i in range(len(V_trace)):
		if V_trace[i]>40:
			V_trace[i] = 1
		else:
			V_trace[i] = 0
	new_trace = np.array(V_trace, dtype=np.int)
	differences_array.append(sum(np.diff(new_trace.nonzero()[0])))

plt.plot(differences_array)
plt.show()