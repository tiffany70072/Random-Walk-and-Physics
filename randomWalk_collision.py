import numpy as np
import random
import matplotlib.pyplot as plt

size  = 15
space = np.zeros([size, size], dtype = int)
a_initial = [8, 8]
b_move = 0
count_collision = 0
print "size =", size
print "b_move =", b_move

def direction(x):
	if   x == 0: return [1, 0]
	elif x == 1: return [-1, 0]
	elif x == 2: return [0, 1]
	else: return [0, -1]

def init_others():
	b_initial = []
	temp = [random.randint(0, size-1), random.randint(0, size-1)]
	b_initial.append(temp)
	return b_initial

def one_step_wall(x, v, count_collision = 0):
	temp = [x[0] + v[0], x[1] + v[1]]
	# avoid collision
	if temp[0] < 0 or temp[0] >= size or temp[1] < 0 or temp[1] >= size:
		temp = x
		count_collision += 1
	return temp, count_collision

def one_step(x, v, count_collision = 0):
	temp = [x[0] + v[0], x[1] + v[1]]
	# avoid collision
	if temp[0] < 0 or temp[0] >= size or temp[1] < 0 or temp[1] >= size:
		temp = x
	return temp

def relative(a, b):
	return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(0.5)

count_list = []
vr = []
sample_time = 400
for sample in range(sample_time):
	a = [random.randint(0, size-1), random.randint(0, size-1)]
	#if b_move == True: 
	b = init_others()
	t = 0
	count= 0
	
	for t in range(10000):
		a_v = direction(random.randint(0, 3))
		#a, count_collision = one_step_wall(a, a_v, count_collision)
		a = one_step(a, a_v)
		
		if b_move == True: 
			for i in range(len(b)):
				b_v = direction(random.randint(0, 3))
				b[i] = one_step(b[i], b_v)
			#b, count_collision = one_step_wall(b, b_v, count_collision)
		#t += 1
		#vr.append(relative(a_v, b_v))
		#space[b[0], b[1]] += 1
		for i in range(len(b)):
			if a == b[i]: count += 1
			#if sample == 0 and t > 100 and t < 105: print a, b[i]
	if sample % 50 == 0 and sample != 0: 
		print "sample num =", sample, ", ave(count) =", np.mean(np.array(count_list))
	count_list.append(count)

count_list = np.array(count_list)
print count_list[:50]
#print "count_collision =", count_collision, count_collision/float(np.sum(count_list))
#space = space/float(np.sum(count))
#space = space/float(t)
ave = np.mean(count_list)
std = np.std(count_list)
print "ave =", ave, ", std =", std

exit()
print ""
for i in range(size):
	for j in range(size): print "%4.2f" %(space[i][j]*100),
	print ""

plt.plot([i-size/2 for i in range(size)], space[:, size/2], 'b-')
plt.plot([i-size/2 for i in range(size)], space[size/2, :], 'g-')
plt.ylabel('Prob')
plt.xlabel('Position. Size = ' + str(size))
#plt.hist(count_list, facecolor = 'r', alpha = 0.75)
plt.show()



	
