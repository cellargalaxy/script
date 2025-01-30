#       0       1    2      3     4
data = ['a1', 'b2', 'c3', 'd4', 'e5']
indexs = []
for i in range(0, len(data), 2):
    indexs.append(i)
indexs.append(len(data) - 1)
indexs = list(set(indexs))
print(indexs)


