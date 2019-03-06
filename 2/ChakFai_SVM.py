import sys
import numpy as np
import time
from svmutil import *
import matplotlib.pyplot as plt
import pickle

def normalize_data(x, scale):
	return x / scale

def compare_test(predicted, actual):
    results = list(zip(predicted,actual))
    correct = 0
    total = 0
    for (pred,act) in results:
        if act == pred:
            correct += 1
        total += 1
    print('Results accuracy = ' + str(correct) + '/' + str(total))
    print('Percentage = ' + str(correct/total))
    print()


train_x = []
train_y = []
test_x = []
test_y = []
norm_train_x = []
norm_test_x = []
prob = None

def read_train_data(limit):
	global train_x, train_y

	with open('train.csv', 'r') as train_data:
	    for line in train_data:
	        bits = list(map(int, line.rstrip().split(',')))
	        train_x.append(bits[:-1])
	        train_y.append(bits[-1])
	        limit -= 1
	        if limit == 0:
	        	break

	train_x = np.array(train_x)
	train_y = np.array(train_y)

def read_test_data(limit):
	global test_x, test_y

	with open('test.csv', 'r') as test_data:
	    for line in test_data:
	        bits = list(map(int, line.rstrip().split(',')))
	        test_x.append(bits[:-1])
	        test_y.append(bits[-1])
	        limit -= 1
	        if limit == 0:
	        	break
	
	test_x = np.array(test_x)
	test_y = np.array(test_y)


# Q2(a) Pegasos implementation
def pegasos_train_pair(x, y, maxT, batch_size, penalty_const):	
	w = np.zeros(x.shape[1])
	b = 0

	m = len(x)
	for itr in range(maxT):
		k = np.random.choice(m, batch_size, replace=False)
		eta = 1/(itr+1)

		old_w = w
		old_b = b

		indicator = y[k]*(x[k].dot(old_w) + old_b)
		w = (1-eta)*old_w + eta*penalty_const*y[k[indicator < 1]].dot(x[k[indicator < 1]])

		b = old_b + eta*penalty_const*np.sum(y[k[indicator < 1]])

	return (w, b)

# Q2(b) Pegasos testing 
def pegasos_test(w, b, x):
	match_result = []

	for di in range(10):
		for dj in range(di+1, 10):
			hx = x.dot(w[(di,dj)]) + b[(di,dj)]
			match_result.append(np.maximum((hx < 0)*di, (hx >= 0)*dj))

	match_result = np.array(match_result)

	test_result_y = list(map(lambda x:np.where(x == x.max())[0][-1], map(np.bincount, match_result.T)))

	return test_result_y

def pegasos(train_x, train_y, test_x, test_y):
	print('Start training')
	start = time.time()

	w = {}
	b = {}

	for di in range(10):
		for dj in range(di+1, 10):
			xi = train_x[train_y == di]
			xj = train_x[train_y == dj]

			x = np.append(xi, xj, axis=0)
			y = np.append(len(xi)*[-1],len(xj)*[1])

			w[(di,dj)],b[(di,dj)] = pegasos_train_pair(x, y, 500, 100, 1.0)

	pickle.dump( (w,b), open('pegasos.p','wb') )

	print('End training (Time: ' + str(time.time() - start) + ')')

	print('Training result using Pegasos')
	train_result_y = pegasos_test(w, b, train_x)
	compare_test(train_result_y, train_y)

	print('Prediction result using Pegasos')
	test_result_y = pegasos_test(w, b, test_x)
	compare_test(test_result_y, test_y)


# Q2(c) Libsvm training and testing
def libsvm_testing(x, y, model):
	start = time.time()
	p_label, p_acc, p_val = svm_predict(y.tolist(), x.tolist(), model)
	print('(Time: ' + str(time.time() - start) + ')')
	print()

	return list(map(int,p_label)), p_acc[0]

def libsvm_linear():
	global prob, norm_test_x, norm_train_x, train_y, test_y

	# Linear Kernel, C = 1.0
	linear_param = svm_parameter('-t 0 -c 1')
	print(prob)
	linear_m = svm_train(prob, linear_param)
	svm_save_model('libsvm_linear.model', linear_m)

	print('Training result using libsvm linear kernel')
	libsvm_testing(norm_train_x, train_y, linear_m)

	print('Prediction result using libsvm linear kernel')
	libsvm_testing(norm_test_x, test_y, linear_m)

def libsvm_gaussian():
	global prob, norm_test_x, norm_train_x, train_y, test_y

	# RBF (Gaussian) Kernel, C = 1.0, gamma = 0.05
	rbf_param = svm_parameter('-t 2 -c 1 -g 0.05')
	rbf_m = svm_train(prob, rbf_param)
	svm_save_model('libsvm_gaussian.model', rbf_m)

	print('Training result using libsvm RBF kernel')
	libsvm_testing(norm_train_x, train_y, rbf_m)

	print('Prediction result using libsvm RBF kernel')
	libsvm_testing(norm_test_x, test_y, rbf_m)

# Q2(d) K-fold cross validation
def libsvm_k_fold(c):
	global prob, norm_test_x, test_y

	start = time.time()
	cross_param = svm_parameter('-t 2 -c '+str(c)+' -g 0.05 -v 10')
	accuracy = svm_train(prob, cross_param)

	rbf_param = svm_parameter('-t 2 -c '+str(c)+' -g 0.05')
	rbf_m = svm_train(prob, rbf_param)
	svm_save_model('libsvm_gaussian_C_'+str(c)+'.model', rbf_m)

	print('Prediction result using libsvm RBF kernel with C = '+str(c))
	[_, pred_acc] = libsvm_testing(norm_test_x, test_y, rbf_m)

	return accuracy, pred_acc

def cross_validation(Cs):
	cross_accuracy = []
	test_accuracy = []
	for c in Cs:
		cross_acc, test_acc = libsvm_k_fold(c)
		cross_accuracy.append(cross_acc)
		test_accuracy.append(test_acc)

	plt.plot(Cs,cross_accuracy,label='Cross validation accuracy')
	plt.plot(Cs,test_accuracy,label='Prediction accuracy')
	plt.xscale('log')

	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.title('Cross validation result for different C')
	plt.show()

	return cross_accuracy

def find_best_c(Cs):
	cross_accuracy = cross_validation(Cs)
	best_c = Cs[np.argmax(cross_accuracy)]
	return best_c

def confusion_matrix(num_class, predict, actual):
	# Q2(e) Confusion matrix of best C
    confusion = np.zeros((num_class,num_class), dtype=int)
    results = list(zip(predict,actual))
    for (pred,act) in results:
        confusion[pred][act] += 1

    print('Confusion matrix')
    print(confusion)
    print()

def load_libsvm(modelname):
	global norm_test_x, test_y

	print('Loading file: '+ modelname)
	best_rbf_m = svm_load_model( modelname )
	[test_result_y, _] = libsvm_testing(norm_test_x, test_y, best_rbf_m)
	return test_result_y

def visualize(id):
	global test_x

	im = test_x[id].reshape(28,28).tolist()
	plt.imshow(im, interpolation='nearest', cmap='Greys')
	plt.show()

def visualize_two_class(test_result_y, test_y, pre_num, act_num):
    test_result_y = np.array(test_result_y)
    predict_res = np.argwhere(test_result_y == pre_num)
    actual_res = np.argwhere(test_y == act_num)
    misclass_ids = np.intersect1d(predict_res, actual_res)

    cnt = 0
    for mid in misclass_ids:
            visualize(mid)
            cnt += 1
            if cnt == 3:
                    break


def home_use():
	global prob, test_x, test_y, train_x, train_y, norm_train_x, norm_test_x

	print('Reading text files')
	start = time.time()

	read_train_data(20000)
	read_test_data(10000)

	print('Done reading text files (Time: ' + str(time.time() - start) + ')')

	pegasos(train_x, train_y, test_x, test_y)

	scale = np.max(train_x)
	norm_train_x = normalize_data(train_x, scale)
	norm_test_x = normalize_data(test_x, scale)

	prob = svm_problem(train_y.tolist(), norm_train_x.tolist())

	libsvm_linear()
	libsvm_gaussian()

	Cs = [1e-5,1e-3,1,5,10]
	best_c = find_best_c(Cs)
	best_c = 10

	test_result_y = load_libsvm('libsvm_gaussian_C_'+str(best_c)+'.model')
	confusion_matrix(10, test_result_y, test_y)

	# visualize_two_class(test_result_y, test_y, 2, 7)

def demo(model, input_file, output_file):
	global test_x, norm_test_x, test_y

	with open(input_file, 'r') as test_data:
	    for line in test_data:
	        bits = list(map(int, line.rstrip().split(',')))
	        test_x.append(bits)
	
	test_x = np.array(test_x)
	test_y = np.array([0]*len(test_x))

	norm_test_x = normalize_data(test_x, 255)

	if model == '1':
		w,b = pickle.load( open('pegasos.p', 'rb') )
		predicted_label = pegasos_test(w,b,test_x)
	elif model == '2':
		predicted_label = load_libsvm('libsvm_linear.model')
	elif model == '3':
		predicted_label = load_libsvm('libsvm_gaussian_C_10.model')

	with open(output_file, 'w') as out:
	    for lbl in predicted_label:
	    	out.write(str(lbl) + '\n')


if len(sys.argv) < 4:
	home_use()
else:
	demo(sys.argv[1], sys.argv[2], sys.argv[3])