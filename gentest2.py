from urllib import request
import random
import datetime
import pandas as pd
import itertools
from multiprocessing.dummy import Pool as ThreadPool

starttime = datetime.datetime.now()

base_url = "http://splitit.cs.loyola.edu/cgi/splitit.cgi"
max_int = 9999
num_of_splitting = 1
verbose = True

df = pd.read_csv("tmp/hardsplit_binkley_oracle_samples.csv", header=None, keep_default_na=False)
identifiers = list(itertools.chain.from_iterable(df.values[:1000, 0:1]))


splitted_identifiers = list(itertools.chain.from_iterable(df.values[:1000, 1:2]))
print(','.join(splitted_identifiers))
lendata = len(identifiers)


pool = ThreadPool(10)
def split(identifier):
	rand = random.randint(0, max_int)
	# handle exception of url请求
	identifier = identifier.replace('.', '_')
	url = base_url + "?&id=" + identifier + "&lang=java&n=" + str(num_of_splitting) + "&rand=" + str(rand)
	# print("proceesing ", identifier)
	body = request.urlopen(url).read()
	# print("done with", identifier)
	print(identifier, body)
	# return body.decode("utf-8")  
	return identifier, body.decode("utf-8") 



identifiers, bodies =  zip(*pool.map(split, identifiers))

pool.close()
pool.join()
count = 0
precision = 0
recall = 0
index = 0 
for identifier, body in zip(*(identifiers, bodies)):
	# print("got body from", identifier)
	# print(body)
	wrong_split = True
	softwords = body.split('\n')
	gentest_split_result = []
	for i in range(len(softwords) - 1):
		softword = softwords[i].strip('\t1234567890')
		gentest_split_result = gentest_split_result + softword.split('_')

	splitted_identifier = splitted_identifiers[index]
	parts = splitted_identifier.split('-')
	condition = lambda part : part not in ['.', ':', '_', '~']
	parts = [x for x in filter(condition, parts)]
	# calculate precision, recall, fmeasure
	correct_splits = set([i for i in range(len(splitted_identifier)) if splitted_identifier[i:].startswith('-')])
	predict_splits = set()
	prev_pos = 0
	for i in range(len(gentest_split_result) - 1) :
		prev_pos = identifier.find(gentest_split_result[i], prev_pos) + len(gentest_split_result[i])
		predict_splits.add(prev_pos)
	precise_splits = correct_splits & predict_splits
	precision = precision + ((1 + len(precise_splits)) / (1 + len(predict_splits)))
	recall = recall + (1 + len(precise_splits))/ (1+len(correct_splits))
	# calculate accuracy 
	if len(parts) == len(gentest_split_result):
		difference = list(set(parts).difference(set(gentest_split_result)))
		if len(difference) == 0:
			count = count + 1
			wrong_split = False
	if verbose and wrong_split:
		print(parts)
		print(gentest_split_result)

	index = index+1

precision = round((precision/lendata),3)
recall = round((recall/lendata),3)
fmesure = round( 2 * precision * recall / (precision + recall), 3)
print("accuracy of gentest: %d" % (count/lendata))
print("precision of gentest: %d" % precision)
print("recall of gentest: %d" % recall)
print("fmeasure of gentest: %d" % fmeasure )
# print(identifiers)
# print("=========")
# print(bodies)	
endtime = datetime.datetime.now()

print((endtime - starttime).total_seconds())