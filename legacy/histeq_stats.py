
BT11_File = "tmp/bt11_data.csv"
Binkley_File = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"

a1_5 = 0
a6_10 = 0
a11_15 = 0
a16_20 = 0
a21_25 = 0
a26_30 = 0
a31_ = 0

file = open(Binkley_File)
count = 0
for line in file.readlines():
	count = count + 1
	identifier = line.split(' ')[1]
	print(identifier)
	lenorigin = len(identifier)

	if lenorigin <= 5:
		a1_5 = a1_5 + 1
	elif lenorigin <= 10:
		a6_10  = a6_10 + 1
	elif lenorigin <= 15:
		a11_15 = a11_15 +1
	elif lenorigin <=20:
		a16_20 = a16_20 +1
	elif lenorigin <= 25:
		a21_25 = a21_25 +1
	elif lenorigin <= 30:
		a26_30 = a26_30 + 1
	else:
		a31_ = a31_ + 1
print(a1_5, a6_10, a11_15, a16_20, a21_25, a26_30, a31_)
print(a1_5/count, a6_10/count, a11_15/count, a16_20/count, a21_25/count, a26_30/count, a31_/count)
