import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

size = 23
space = np.zeros([size, size], dtype = int) # boundary
pos_initial = [size/2, size/2] 				# initial position
count_collision = 0
print "pos_initial =", pos_initial

def direction(x):
	if   x == 0: return [1, 0]
	elif x == 1: return [-1, 0]
	elif x == 2: return [0, 1]
	else: return [0, -1]

def one_step_wall(x, v, count_collision = 0):
	next_x = x + v
	# avoid collision
	if next_x < 0 or next_x >= size:
		next_x = x
		count_collision += 1
	return next_x, count_collision

def one_step(x, v):
	next_x = [x[0] + v[0], x[1] + v[1]]
	# avoid collision
	if next_x[0] < 0 or next_x[0] >= size or next_x[1] < 0 or next_x[1] >= size:
		next_x = x
	return next_x

def brown(x, y):
	r = math.sqrt(x**2 + y**2)
	return 1/math.sqrt(2*math.pi*10.0)*math.exp(-(r**2)/2.0/10.0)

sample_time = 1
for sample in range(sample_time):
	a = pos_initial
	# smooth the discrete case to continuous case
	if sample < sample_time/4: step = 9
	elif sample < sample_time/2: step = 11
	else: step = 10
	step = 1000000
	
	for t in range(step):
		velocity = direction(random.randint(0, 3))
		a = one_step(a, velocity)
		#a, count_collision = one_step_wall(a, velocity, count_collision)
		space[a[0], a[1]] += 1

#count = np.array(count)
#print "count_collision =", count_collision, count_collision/float(np.sum(count))
#space = space/float(sample_time)
space = space/float(step)
#ave = np.mean(count)
#std = np.std(count)
#print "ave =", ave, ", std =", std

print ""
for i in range(size):
	print "%4.2f" %(space[size/2][i]*100),
print ""

'''plt.plot([i-size/2 for i in range(size)], [brown(i-11) for i in range(size)], 'r-')
plt.plot([i-size/2 for i in range(size)], space[size/2], 'b-')

#plt.plot([i-size/2 for i in range(size)], space[size/2, :], 'g-')
plt.ylabel('Prob')
plt.xlabel('Position. (Size = ' + str(size) + ')')
plt.title('t = 9(1/4), 10(1/2), 11(1/4), sample = 100000')
#plt.hist(count, facecolor = 'r', alpha = 0.75)
plt.show()'''

# output the figure
def random_walk(x, y, space):
	return space[x+size/2][y+size/2]

fig = plt.figure(figsize=(10, 6), edgecolor = 'k')
p = fig.add_subplot(111, projection = '3d')

p.set_xlabel('X')
p.set_ylabel('Y')
p.set_zlabel('Prob')
plt.title("Random walk (2D), t = 1000000")
x = y = np.arange(-size/2, size/2, 1)
X, Y  = np.meshgrid(x, y)
zs    = np.array([random_walk(x, y, space) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z     = zs.reshape(X.shape)
#Z_brown = np.array([brown(x, y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#Z_brown = Z_brown.reshape(X.shape)
#surf = p.plot_surface(X, Y, Z, alpha = 0.3, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
surf = p.plot_surface(X, Y, Z, alpha = 0.7, linewidth = 0, antialiased = False)
#brown = p.plot_surface(X, Y, Z_brown, alpha = 0.3, color = 'g', linewidth = 0, antialiased = False)
plt.show()
#p.scatter(X, Y, Z, c = c, marker = 'o', s = 5)
