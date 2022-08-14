import matplotlib.pyplot as plt
import sys

def F(w, points):
	sum_F = 0
	for p_index, tuple  in enumerate(points):#(1, 1, 1), (1, 2, 2.5), (1, 3, 2), (1, 4, 4), (1, 5, 4.5), (1, 6, 6.3)
		sum_w_values = 0
		#(1,1,1)
		for w_index in range(len(w)):#[w_0, w_1]
			sum_w_values = sum_w_values + w[w_index] * tuple[w_index] #w_1*x_1 + w_0*x_0 -> x_0 = 1
		sum_w_values = (sum_w_values - tuple[len(w)])**2 #((w_1*x_1 + w_0*x_0) - y) ^2
		sum_F = sum_F + sum_w_values
	
	return (sum_F/len(points))

def dF(w, points, index):
	sum_dF = 0
	for p_index, tuple  in enumerate(points):
		sum_w_values = 0
		for w_index in range(len(w)):
			sum_w_values = sum_w_values + w[w_index] * tuple[w_index]
		sum_dF = sum_dF + 2*(sum_w_values - tuple[len(w)]) * tuple[index] 
		
		
	return (sum_dF/(len(points)))


def calculate_weights (w, points, eta):
	adjusted_weights = []
	for index in range(len(w)):
		w[index] = w[index] - eta  * dF(w, points, index)
	return (w)

def print_line(points, w, iteration, line_color = None, line_style = 'dotted'):
	list_x = []
	list_y = []

	for index, tuple in enumerate(points):
		sum_y_value = 0
		for w_index in range(len(w)):
			sum_y_value = sum_y_value + tuple[w_index] * w[w_index]
		y = sum_y_value
		list_x.append(tuple[1])
		list_y.append(y)
	plt.text(points[index][len(w)-1],y, iteration, horizontalalignment='right')
	plt.plot(list_x, list_y, color = line_color, linestyle= line_style)
	
if __name__=='__main__':
	points = [(65, 2, 5, 2000000), (150, 3, 10, 3500000), (120, 3, 20, 1800000), (250, 4, 15, 4200000), (70, 2, 4, 3000000), (180, 4, 25, 3700000), (130, 3, 10, 2800000), (200, 4, 23, 5000000), (150, 4, 30, 4500000)]
	points2 = [(65, 2000000), (150, 3500000), (120, 1800000), (250, 1800000), (70, 3000000), (180, 3700000), (130, 2800000), (200, 5000000), (150, 4500000)]
	iterations = 250
	# int(sys.argv[1])
	plt.scatter(*zip(*points2))
	
	w= [0,0,0]
	eta = 0.0000001
	for t in range(iterations):
		value = F(w, points)
		print ('value: ', value)
		w = calculate_weights(w, points, eta)
		print ('iteration {}: w = {}, wF(w) = {}'.format(t, w, value))
		print_line(points, w, t)

	print_line(points, w, t, 'red', 'solid')
	plt.show()