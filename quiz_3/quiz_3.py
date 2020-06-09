import math 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 

def entropy(dist):
	H = -sum([prob_i*math.log2(prob_i) for prob_i in list(dist)])
	return H


### Question_1 

bern = [0.9, 0.1]

print("Entrop of bern = ", entropy(bern))
## Question_2 

F_S0 = np.array([17/18, 1/18])
F_S1 = np.array([0.5, 0.5])

F = bern[0]*F_S0 + bern[1]*F_S1

H_F_S0 = entropy(dist = F_S0)
H_F_S1 = entropy(dist = F_S1)

H_F = entropy(F)

I_S_F = H_F - (bern[0]*H_F_S0 + bern[1]*H_F_S1) 

print("Mutual info = ", I_S_F)


## Question_7 


import pickle

with open('tuning_3.4.pickle', 'rb') as f:
    data = pickle.load(f)

neuron_1 = data['neuron1']
neuron_2 = data['neuron2']
neuron_3 = data['neuron3']
neuron_4 = data['neuron4']

neurons = [neuron_1, neuron_2, neuron_3, neuron_4]

per_trail_neural_responses = [np.mean(neuron_i, axis = 0) for neuron_i in neurons]

def get_Fano_factor(neuron):
	counts = np.sum(neuron, axis = 1)
	counts = counts*10/24
	return np.std(counts)**2/np.mean(counts) 

for neuron_i in neurons:
	print("Fano factor = ", get_Fano_factor(neuron_i))


## Question 9 

with open('pop_coding_3.4.pickle', 'rb') as f:
    coding_data = pickle.load(f)

r1 = np.mean(coding_data['r1']/np.max(neuron_1))
r2 = np.mean(coding_data['r2']/np.max(neuron_2))
r3 = np.mean(coding_data['r3']/np.max(neuron_3))
r4 = np.mean(coding_data['r4']/np.max(neuron_4)) 

rs =  [r1, r2, r3, r4]

c1 =  coding_data['c1']
c2 =  coding_data['c2']
c3 =  coding_data['c3']
c4 =  coding_data['c4']

cs = [c1, c2, c3, c4]

def get_angle(c):
	angle = (np.arctan2(c[1], c[0])*180/np.pi)
	return angle if angle>0  else 360+angle 

direction = sum(i for i in [r * c for r,c in zip(rs, cs)])

direction = get_angle(direction)

print("Angle of stimulus = ", direction)
