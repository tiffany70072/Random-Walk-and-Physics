import numpy as np
import random
import math
import matplotlib.pyplot as plt

size  = 23
space = np.zeros([size], dtype = int) 	# boundary
pos_initial = size/2 					# initial position
count_collision = 0 					# number of collision
print "pos_initial =", pos_initial

def direction(x):
	if x == 0: return 1
	else: 	   return -1

def one_step_wall(x, v, count_collision = 0):
	next_x = x + v
	if next_x < 0 or next_x >= size: # avoid collision
		next_x = x
		count_collision += 1
	return next_x, count_collision

def one_step(x, v): # move one step and avoid collision
	next_x = x + v
	if next_x < 0 or next_x >= size: next_x = x
	return next_x

def brown(x):
	return 1/math.sqrt(2*math.pi*10.0)*math.exp(-(x**2)/2.0/10.0)

sample_time = 100000
for sample in range(sample_time):
	pos = pos_initial
	# smooth the discrete case to continuous case
	if   sample < sample_time/4: step = 9
	elif sample < sample_time/2: step = 11
	else: step = 10
	# start moving
	for t in range(step):
		velocity = direction(random.randint(0, 1))
		pos = one_step(pos, velocity)
		#pos, count_collision = one_step_wall(pos, velocity, count_collision)
	space[pos] += 1

#count = np.array(count)
#print "count_collision =", count_collision, count_collision/float(np.sum(count))
space = space/float(sample_time)
#ave = np.mean(count)
#std = np.std(count)
#print "ave =", ave, ", std =", std

print ""
for i in range(size): print "%4.2f" %(space[i]*100),
print ""

plt.plot([i-size/2 for i in range(size)], [brown(i-11) for i in range(size)], 'r-')
plt.plot([i-size/2 for i in range(size)], space, 'b-')

plt.ylabel('Prob')
plt.xlabel('Position. (Size = ' + str(size) + ')')
plt.title('t = 9(1/4), 10(1/2), 11(1/4), sample = 100000')
plt.show()


	
