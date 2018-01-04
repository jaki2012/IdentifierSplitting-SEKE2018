import os

file = open("tmp/coded_file.csv")
c = []
for line in file.readlines():
	a = line.strip().split(',')
	b = a[:30]
	c.append("+".join(b))
print(len(set(c)))