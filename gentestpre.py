a = ["good", "book"]
b = "GOODBOOK"
c = "GOOD-BOOK"

# binkley是java 切需要转大小写
# 而BT11则不需要

correct_index = set([i for i in range(len(c)) if c[i:].startswith('-')])
predict_index = set()
prev_pos = 0
for i in range(len(a) - 1) :
	prev_pos = b.lower().find(a[i], prev_pos) + len(a[i])
	predict_index.add(prev_pos)

precise_index = correct_index & predict_index
# pyython 2 不能这样写
# precison = 1/2
precison = (1 + len(precise_index)) / (1 + len(predict_index))
print(precison)