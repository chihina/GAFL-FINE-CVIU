import itertools

data_size = range(1)
data_patterns = itertools.permutations(data_size, 2)
for data_pattern in data_patterns:
    print(data_pattern)