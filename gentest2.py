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

df = pd.read_csv("tmp/cheat_splitting_file.csv", header=None)
identifiers = list(itertools.chain.from_iterable(df.values[34595:, 0:1]))

splitted_identifiers = list(itertools.chain.from_iterable(df.values[34595:, 1:2]))
lendata = len(identifiers)


pool = ThreadPool(10)
def split(identifier):
	rand = random.randint(0, max_int)
	# handle exception of url请求
	identifier = identifier.replace('.', '_')
	url = base_url + "?&id=" + identifier + "&lang=all&n=" + str(num_of_splitting) + "&rand=" + str(rand)
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
	if len(parts) == len(gentest_split_result):
		difference = list(set(parts).difference(set(gentest_split_result)))
		if len(difference) == 0:
			count = count + 1
			wrong_split = False
	if verbose and wrong_split:
		print(parts)
		print(gentest_split_result)

	index = index+1
print(count/lendata)

# print(identifiers)
# print("=========")
# print(bodies)	
endtime = datetime.datetime.now()

print((endtime - starttime).total_seconds())