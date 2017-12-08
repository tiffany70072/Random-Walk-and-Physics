import numpy as np
import random
import matplotlib.pyplot as plt
import math


size = 10
space = np.zeros([size], dtype = float)
pos_initial = 0

def direction(x):
	if x <= 0: return 1
	else: 	   return -1

def one_step(x, v):
	next_x = x + v
	if next_x < 0 or next_x >= size: next_x = x
	return next_x

count = []
sample_time = 1000
for sample in range(sample_time):
	a = random.randint(0, size-1)
	
	for t in range(1000):
		velocity = direction(random.randint(0, 2))
		a = one_step(a, velocity)
		space[a] += 1
		
	if sample % 100 == 0: print space
	
space = space/np.float(sample_time*t)
print "space = "
for i in range(size): print "%4.2f" %(space[i]*100),
print ""

def func1(size):
	g = [math.exp(-x*2/3.0) for x in range(size)]
	#g = [1-x*3/4.0 for x in range(size)]
	print g
	#g = [1 for x in range(size)]
	total = float(sum(g))
	g = [g[i]/total for i in range(size)]
	print "g =", g
	return g

def func2(size):
	g = [(1/2.0)**x for x in range(size)]
	total = float(sum(g))
	g = [g[i]/total for i in range(size)]
	return g

# output the fitting figure
plt.plot(func1(size), 'b-')

plt.axis([0, size, 0, 0.5])
plt.xlabel("x, with size = " + str(size))
plt.ylabel("Prob")
plt.title("P(+1) = 1/3")
plt.plot(space.tolist(), 'r-')
plt.plot(func2(size), 'g-')
plt.show()

