import os

for i in range(10):
	os.system("python hardsplit.py")
	os.system("python lstm_preprocess.py")
	os.system("mv tmp/hardsplit_bt11_oracle_samples.csv tmp/hs_random_oracles/bt11/"+str(i+1)+"_hardsplit_bt11_oracle_samples.csv")
	os.system("mv tmp/hardsplit_bt11_coded_files.csv tmp/hs_random_oracles/bt11/"+str(i+1)+"_hardsplit_bt11_coded_files.csv")