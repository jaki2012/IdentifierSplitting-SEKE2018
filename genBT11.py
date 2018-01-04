import os
import csv
import pandas as pd

BT11_TESTS_SET_PATH = "/Users/lijiechu/Documents/huawei2/tests"
DATACSV = "tmp/bt11_data.csv"
REPEAT_DATACSV = "tmp/repeat_bt11_data.csv"
bt11_datas = []
i = 0
for path, subpaths, files in os.walk(BT11_TESTS_SET_PATH):
	for file in files:
		if(-1 != file.find("plain")):
			continue
		i = i +1
		print("processing bt11 file #%d" % (i))
		all_the_text = open(os.path.join(path,file)).readlines()
		for line in all_the_text:
			bt11_data = line.strip().replace("___","_").replace("__","_")
			if len(bt11_data) <= 3:
				continue
			bt11_datas.append(bt11_data)
with open(DATACSV, "w") as datacsv:
	df = pd.read_csv(REPEAT_DATACSV, header=None)
	vl = df.values
	c = []
	d = []
	e = []
	for i in range(len(vl)):
		c.append(vl[i][0])
		d.append(vl[i][1])
		e.append(vl[i][2])
	print(len(bt11_datas))
	print(len(set(bt11_datas)))
	a = []
	b = []
	repeat_count = 0
	final_datas = [x.split('\t') for x in set(bt11_datas)]
	final_final_datas = []
	for i in range(len(final_datas)):
		if final_datas[i][0] not in a:
			a.append(final_datas[i][0])
			b.append(final_datas[i][1])
			final_final_datas.append([final_datas[i][0],final_datas[i][1]])
		else:
			index = a.index(final_datas[i][0])
			# 重复了三次日
			if(final_datas[i][0]=="FNAME_TOKEN"):
				final_final_datas[index] = [final_datas[i][0],"f-name-token"]
				continue
			which = 0
			if b[index] in c:
				index_c = c.index(b[index])
				which = index_c
			elif b[index] in d:
				index_d = d.index(b[index])
				which = index_d
			else:
				print(b[index])
				print("haha")
			if(e[which] == 1):
				final_final_datas[index] = [final_datas[i][0],c[which]]
			elif(e[which] == 2):
				final_final_datas[index] = [final_datas[i][0],d[which]]
			else:
				print("??")
			repeat_count = repeat_count + 1
			# repeat_datas.append([b[index],final_datas[i][1]])
	print(len(a))
	print(repeat_count)
	csvwriter = csv.writer(datacsv)
	csvwriter.writerows(final_final_datas)
	# with open(REPEAT_DATACSV,"w") as repeatdatacsv:
	# 	csvwriter = csv.writer(repeatdatacsv)
	# 	csvwriter.writerows(repeat_datas)