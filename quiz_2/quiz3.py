import numpy as np 
from scipy.stats import  multivariate_normal
import matplotlib.pyplot as plt 



## Question _1 

s = np.linspace(-10,10, 199)
pdf_1 = multivariate_normal.pdf(s, 5, 0.5)
pdf_2 = multivariate_normal.pdf(s, 7, 1)


plt.plot(s, pdf_1)
plt.plot(s, pdf_2)

def expected_loss(point, gaussian, loss):
	return multivariate_normal.pdf(point, *gaussian) * loss 


values = [5.667, 2.69, 5.830, 5.978]

results = []
for value in values:
	l1 = expected_loss(value, (5, 0.5), loss = 1)
	l2 = expected_loss(value, (7, 1), loss = 2)

	results.append((value, l1/l2))
	print("ration of Loss of point {} = ".format(value), l1/l2)

## By plotting the results best value was found to be 5.667 

best = min(results, key = lambda x: x[1])[0]
print(best)
plt.axvline(x=best, color = 'r')
plt.show()