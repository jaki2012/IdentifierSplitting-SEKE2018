import pandas as pd

df = pd.read_csv("tmp/non_hardsplit_jhotdraw_oracle_samples.csv")

datas = df.values[:, :]

suck1 = 0
suck2 = 0
suck3 = 0
suck4 = 0
for data in datas:
	seqs = data[32:62]
	lens = len(seqs)
	for i in range(lens-1):
		if seqs[i] == 'E' and seqs[i+1] =='E':
			suck1 = suck1 + 1
			print(data)
		if seqs[i] == 'S' and seqs[i+1] == 'E':
			suck2 = suck2 + 1
			print(data)
		if seqs[i] == 'B' and seqs[i+1] == 'S':
			suck3 = suck3 + 1
			print(data)
		if seqs[i] == 'E' and seqs[i+1] == 'M':
			suck4 = suck4 + 1
			print(data)


print(suck1, suck2, suck3, suck4)