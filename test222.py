import pandas as pd

df = pd.read_csv('tmp/final_training_set.csv',header=None)

datas = df.values[:, 2:32]

chars_set = set()

for data in datas:
	for char in data:
		chars_set.add(char)

chars_list = list(chars_set)
chars_list.sort()

print(chars_list)




