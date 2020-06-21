import pickle
import matplotlib.pyplot as plt 
import numpy as np 

with open('c10p1.pickle', 'rb') as f:
    data = pickle.load(f)

points = data['c10p1']

# plt.axvline(x=0, color = 'r')
# plt.axhline(y=0, color = 'r')

# plt.scatter(points[:,0], points[:,1])

means = np.mean(points, axis = 0)
print(f"The means are: {means}")
points = points - means 

#points = points + np.array([6, 8])
# plt.scatter(points[:,0], points[:,1])
# plt.show()

def step(w, point):
	alpha = 1
	delt_t = 0.01
	eta = 1 

	v = float(np.dot(w, point))
	w = w + delt_t*eta*(v*point )
	return w 

w = np.random.rand(2)
print(f"initial_w: {w}")



for i in range (100000//len(points)):
	for point in points:
		w = step(w, point)

print(f"final_w: {w}")

print(f"Unit w is: {w/np.linalg.norm(w)}")
values, vectors = np.linalg.eig((points.T@points)/len(points))
print(values)
print(vectors)