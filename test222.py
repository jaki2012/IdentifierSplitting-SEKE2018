txt= "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"

lines = open(txt).readlines()

count = 0
c10 = 0
c20 = 0
c30 = 0
c40 = 0


for line in lines:
	count = count +1
	if (count == 1) :
		continue
	if (len(line.split(' ')[1]) <= 10):
		c10 = c10 + 1
	elif (len(line.split(' ')[1]) <= 20):
		c20 = c20 + 1
	elif (len(line.split(' ')[1]) <= 30):
		c30 = c30 + 1
	else:
		c40 = c40 + 1

# count = count - 1
print(count)
print(c10/count,c20/count,c30/count,c40/count)

