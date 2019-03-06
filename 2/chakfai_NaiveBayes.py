import sys
import string
import numpy as np
import time
import pickle
"""herer we are mapping 1,2,3,to 0,1,2 and dict is kind of hashing"""

label_map = dict(zip([1,2,3,4,7,8,9,10], [0,1,2,3,4,5,6,7]))
label_goodbad_map = dict(zip([1,2,3,4,7,8,9,10], [0,0,0,0,1,1,1,1]))

def get_dict_word(index):
	""" list is the ordereed values """
	return (list(word_dict.keys())[list(word_dict.values()).index(index)])

def normalize_line(line):
	# Remove EOL
	"""The rstrip() method returns a copy of the string with trailing characters removed"""
	data = line.rstrip()
	# Remove punctuations
	data = data.replace('<br />',' ')
	"""String of ASCII characters which are considered punctuation characters in the C locale"""
	for c in string.punctuation:
		data = data.replace(c,' ')
	# nomalize to lower case
	data = data.lower()
	# Split into word lists
	""" Remove empty strings from a list of strings"""
	words = list(filter(None, data.split(' ')))
	"""print(words)"""
	return words

def read_train_input(trainfile):

	"""  # empty dictionary
my_dict = {}

# dictionary with integer keys
my_dict = {1: 'apple', 2: 'ball'}  """
	word_dict = {}
	train_x = []

	with open(trainfile, 'r', encoding='utf-8') as train_in:
		for line in train_in:
			words = normalize_line(line)
			# Index word
			wid = []
			for word in words:
				if not word in word_dict:
					word_dict[word] = len(word_dict)
				wid.append(word_dict[word])
				"""the append() method only modifies the original list. It doesn't return any value"""
			train_x.append(wid)

	return np.array(train_x), word_dict

def read_test_input(testfile, word_dict):
	test_x = []

	with open(testfile, 'r', encoding='utf-8') as test_in:
		for line in test_in:
			words = normalize_line(line)
			""" >>> filter(lambda k: 'ab' in k, lst)
					['ab', 'abc']"""
			old_words = list(filter((lambda x: x in word_dict), words))
			"""print(old_words)"""
			wids = list(map((lambda x: word_dict[x]), old_words))
			"""print(wids) check the index of the old words"""
			test_x.append(wids)

	return np.array(test_x)

def read_label_input(labelfile, mapping):
	train_y = []

	with open(labelfile, 'r') as label_in:
		for line in label_in:
			lbl = line.rstrip()
			train_y.append(mapping[int(lbl)])
	"""print(np.array(train_y))"""
	return np.array(train_y)

def naive_bayes(num_class, train_x, train_y, word_dict, laplace_c = 1):
	print('Start training')
	start = time.time()
	"""numclass = 8
	print(num_class)"""
	label_freq = np.zeros(num_class)
	label_wordcnt = np.zeros(num_class)
	for i in np.arange(num_class):
		sum_freq = np.sum(train_y == i)
		"""print(sum_freq)"""
		if sum_freq > 0:
			label_freq[i] = sum_freq
			label_wordcnt[i] = np.sum(list(map(len, train_x[train_y == i])))


	""" we find parameter phi y where we find the no of times there is the rating "0,1,2,3,4,5,6,7,8" in the training set"""
	phi_j = np.zeros(num_class)
	for k in np.arange(num_class):
		phi_j[k] = label_freq[k] / np.sum(label_freq)

	 
	# Build word matrix
	word_matrix = np.zeros((num_class,len(word_dict)))
	zxy = list(zip(train_x,train_y))
	"""print(train_x)"""
	for (xs,ys) in zxy:
		""" >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
>>> u, indices = np.unique(a, return_index=True)
>>> u
array(['a', 'b', 'c'],
	   dtype='|S1')
>>> indices
array([0, 1, 3])
>>> a[indices]   """
		k,v = np.unique(xs, return_counts=True)
		"""print("both k and v")
		print(k)
		print(v)"""
		for (x,cnt) in zip(k,v):
			word_matrix[ys][x] += cnt

	phi_j_given_k = np.zeros((num_class,len(word_dict)))
	for k in np.arange(num_class):
		phi_j_given_k[k] = (word_matrix[k]+laplace_c) / (label_wordcnt[k]+len(word_dict)*laplace_c)

	"""print(phi_j_given_k)"""

	print('End training (Time: ' + str(time.time() - start) + ')')

	return phi_j, phi_j_given_k

def naive_bayes_test(phi_j, phi_j_given_k, test_x):
	test_result = []
	for wid in test_x:
		probs = np.zeros(len(phi_j))
		# wid = list(set(wid))
		for k in np.arange(len(phi_j)):
			prob = np.log(phi_j[k])
			# Note that we don't care about the word not in training data because it adds equal probably to each class
			prob += np.sum(np.log(phi_j_given_k[k][wid]))
			probs[k] = prob
		test_result.append(np.argmax(probs))

	return test_result


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


def baseline(phi_j, test_y):
	## Random Prediction
	random_result = np.random.choice(np.arange(len(phi_j)),len(test_y))

	print('Random Prediction')
	compare_test(random_result, test_y)

	# Majority Prediction
	majority_result = [np.argmax(phi_j)]*len(test_y)
	print('Majority Prediction')
	compare_test(majority_result, test_y)

def confusion_matrix(predict, actual):
	global label_map

	num_class = len(label_map)
	confusion = np.zeros((num_class,num_class), dtype=int)
	results = list(zip(predict,actual))
	for (pred,act) in results:
		confusion[pred][act] += 1

	print('Confusion matrix')
	print(confusion)
	print()





# Q1(e) Feature engineering
def read_train_input_bigram(trainfile):
	word_dict = {}
	train_x = []

	with open(trainfile, 'r', encoding='utf-8') as train_in:
		for line in train_in:
			words = normalize_line(line)
			# Index word
			wid = []
			last_word = ''
			for i, word in enumerate(words):

				# if not word in word_dict:
				#   word_dict[word] = len(word_dict)
				# wid.append(word_dict[word])

				if len(last_word) > 0:
					bigram = last_word + ' ' + word
					if not bigram in word_dict:
						word_dict[bigram] = len(word_dict)
					wid.append(word_dict[bigram])

				last_word = word

			train_x.append(wid)

	return np.array(train_x), word_dict

def read_test_input_bigram(testfile, word_dict):
	test_x = []

	with open(testfile, 'r', encoding='utf-8') as test_in:
		for line in test_in:
			words = normalize_line(line)
			wids = []

			last_word = ''
			for i, word in enumerate(words):

				# if word in word_dict:
				#   wids.append(word_dict[word])

				if len(last_word) > 0:
					bigram = last_word + ' ' + word
					if bigram in word_dict:
						#print(bigram)
						wids.append(word_dict[bigram])

				last_word = word

			test_x.append(wids)

	return np.array(test_x)

def feature_bigram():
	train_x, word_dict = read_train_input_bigram('imdb_train_text_nostem.txt')
	train_y = read_label_input('imdb_train_labels.txt', label_map)

	phi_j, phi_j_given_k = naive_bayes(len(label_map), train_x, train_y, word_dict)

	pickle.dump( (phi_j,phi_j_given_k,word_dict), open('naive_bayes_bigram.p','wb') )

	train_result_y = naive_bayes_test(phi_j, phi_j_given_k, train_x)
	print('Naive Bayes - Training')
	compare_test(train_result_y, train_y)

	test_x = read_test_input_bigram('imdb_test_text_nostem.txt', word_dict)
	test_y = read_label_input('imdb_test_labels.txt', label_map)

	test_result_y = naive_bayes_test(phi_j, phi_j_given_k, test_x)
	print('Naive Bayes - Prediction')
	compare_test(test_result_y, test_y)


def feature_goodbad():
	train_x, word_dict = read_train_input_bigram('imdb_train_text_nostem.txt')
	train_y_gb = read_label_input('imdb_train_labels.txt', label_goodbad_map)
	train_y = read_label_input('imdb_train_labels.txt', label_map)

	phi_j_gb, phi_j_given_k_gb = naive_bayes(2, train_x, train_y_gb, word_dict)

	train_x_good = train_x[train_y_gb == 1]
	train_y_good = train_y[train_y_gb == 1] - 4
	train_x_bad = train_x[train_y_gb == 0]
	train_y_bad = train_y[train_y_gb == 0]

	phi_j_good, phi_j_given_k_good = naive_bayes(len(np.unique(train_y_good)), train_x_good, train_y_good, word_dict)
	phi_j_bad, phi_j_given_k_bad = naive_bayes(len(np.unique(train_y_bad)), train_x_bad, train_y_bad, word_dict)

	pickle.dump( (phi_j_gb,phi_j_given_k_gb,word_dict), open('naive_bayes_goodbad.p','wb') )
	pickle.dump( (phi_j_good,phi_j_given_k_good), open('naive_bayes_goodbad_phi_good.p','wb') )
	pickle.dump( (phi_j_bad,phi_j_given_k_bad), open('naive_bayes_goodbad_phi_bad.p','wb') )


	train_result_gb_y = naive_bayes_test(phi_j_gb, phi_j_given_k_gb, train_x)
	train_result_gb_y = np.array(train_result_gb_y)
	good_ids = np.where(train_result_gb_y == 1)
	bad_ids = np.where(train_result_gb_y == 0)

	train_result_good_y = naive_bayes_test(phi_j_good, phi_j_given_k_good, train_x[good_ids])
	train_result_good_y = np.array(train_result_good_y) + 4
	train_result_bad_y = naive_bayes_test(phi_j_bad, phi_j_given_k_bad, train_x[bad_ids])
	train_result_bad_y = np.array(train_result_bad_y)

	train_result_y = np.zeros(len(train_x), dtype=int)

	cid = 0
	for ids in np.nditer(good_ids):
		train_result_y[ids] = train_result_good_y[cid]
		cid += 1
	cid = 0
	for ids in np.nditer(bad_ids):
		train_result_y[ids] = train_result_bad_y[cid]
		cid += 1

	print('Naive Bayes - Training')
	compare_test(train_result_y, train_y)


	test_x = read_test_input_bigram('imdb_test_text_nostem.txt', word_dict)
	test_y_gb = read_label_input('imdb_test_labels.txt', label_goodbad_map)
	test_y = read_label_input('imdb_test_labels.txt', label_map)

	test_result_gb_y = naive_bayes_test(phi_j_gb, phi_j_given_k_gb, test_x)
	test_result_gb_y = np.array(test_result_gb_y)
	good_ids = np.where(test_result_gb_y == 1)
	bad_ids = np.where(test_result_gb_y == 0)

	test_result_good_y = naive_bayes_test(phi_j_good, phi_j_given_k_good, test_x[good_ids])
	test_result_good_y = np.array(test_result_good_y) + 4
	test_result_bad_y = naive_bayes_test(phi_j_bad, phi_j_given_k_bad, test_x[bad_ids])
	test_result_bad_y = np.array(test_result_bad_y)

	test_result_y = np.zeros(len(test_x), dtype=int)

	cid = 0
	for ids in np.nditer(good_ids):
		test_result_y[ids] = test_result_good_y[cid]
		cid += 1
	cid = 0
	for ids in np.nditer(bad_ids):
		test_result_y[ids] = test_result_bad_y[cid]
		cid += 1


	print('Naive Bayes - Prediction')
	compare_test(test_result_y, test_y)


def home_use():
	global label_map

	print('Starting naive bayes')

	train_x, word_dict = read_train_input('imdb_train_text.txt')
	train_y = read_label_input('imdb_train_labels.txt', label_map)

	# Q1(a) Naive Bayes
	phi_j, phi_j_given_k = naive_bayes(len(label_map), train_x, train_y, word_dict)
	pickle.dump( (phi_j,phi_j_given_k,word_dict), open('naive_bayes.p','wb') )

	train_result_y = naive_bayes_test(phi_j, phi_j_given_k, train_x)

	print('Naive Bayes - Training')
	compare_test(train_result_y, train_y)

	# ----- Testing -----
	test_x = read_test_input('imdb_test_text.txt', word_dict)
	test_y = read_label_input('imdb_test_labels.txt', label_map)

	test_result_y = naive_bayes_test(phi_j, phi_j_given_k, test_x)

	print('Naive Bayes - Prediction')
	compare_test(test_result_y, test_y)

	# Q1(b) Baseline
	baseline(phi_j, test_y)

	# Q1(c) Confusion matrix
	confusion_matrix(test_result_y, test_y)

	# Q1(d) Stem word and Stop word preprocessing
	train_stem_x, word_dict_stem = read_train_input('imdb_train_text_nostem.txt')
	phi_j, phi_j_given_k_stem = naive_bayes(len(label_map), train_stem_x, train_y, word_dict_stem)
	pickle.dump( (phi_j,phi_j_given_k_stem,word_dict_stem), open('naive_bayes_stem.p','wb') )

	test_stem_x = read_test_input('imdb_test_text_nostem.txt', word_dict_stem)

	test_stem_result_y = naive_bayes_test(phi_j, phi_j_given_k_stem, test_stem_x)

	print('Naive Bayes - Stem words / Stop words removal')
	compare_test(test_stem_result_y, test_y)

	# Q1(e) Feature Engineering
	print('Feature engineering - bi-gram')
	feature_bigram()
	print('Feature engineering - positive / negative sentiment analysis')
	feature_goodbad()


def demo(model, input_file, output_file):

	if model == '1':
		phi_j,phi_j_given_k,word_dict = pickle.load( open('naive_bayes.p', 'rb') )
		test_x = read_test_input(input_file, word_dict)
		predicted_label = naive_bayes_test(phi_j, phi_j_given_k, test_x)

	elif model == '2':
		phi_j,phi_j_given_k,word_dict = pickle.load( open('naive_bayes_stem.p', 'rb') )
		test_x = read_test_input(input_file, word_dict)
		predicted_label = naive_bayes_test(phi_j, phi_j_given_k, test_x)

	elif model == '3':
		phi_j_gb,phi_j_given_k_gb,word_dict = pickle.load( open('naive_bayes_goodbad.p','rb') )
		phi_j_good,phi_j_given_k_good = pickle.load( open('naive_bayes_goodbad_phi_good.p','rb') )
		phi_j_bad,phi_j_given_k_bad = pickle.load( open('naive_bayes_goodbad_phi_bad.p','rb') )
		test_x = read_test_input_bigram(input_file, word_dict)
		test_result_gb_y = naive_bayes_test(phi_j_gb, phi_j_given_k_gb, test_x)
		test_result_gb_y = np.array(test_result_gb_y)
		good_ids = np.where(test_result_gb_y == 1)
		bad_ids = np.where(test_result_gb_y == 0)

		test_result_good_y = naive_bayes_test(phi_j_good, phi_j_given_k_good, test_x[good_ids])
		test_result_good_y = np.array(test_result_good_y) + 4
		test_result_bad_y = naive_bayes_test(phi_j_bad, phi_j_given_k_bad, test_x[bad_ids])
		test_result_bad_y = np.array(test_result_bad_y)

		predicted_label = np.zeros(len(test_x), dtype=int)

		cid = 0
		if len(good_ids[0]) > 0:
			for ids in np.nditer(good_ids):
				predicted_label[ids] = test_result_good_y[cid]
				cid += 1
		cid = 0
		if len(bad_ids[0]) > 0:
			for ids in np.nditer(bad_ids):
				predicted_label[ids] = test_result_bad_y[cid]
				cid += 1


	with open(output_file, 'w') as out:
		for lbl in predicted_label:
			out.write(str(lbl) + '\n')


if len(sys.argv) < 4:
	home_use()
else:
	demo(sys.argv[1], sys.argv[2], sys.argv[3])