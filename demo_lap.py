import lap
import numpy as np

inp_arr = np.array([[0, 0, 3], [0, 2, 0], [1, 0, 0]])
inp_arr *= -1
cost, x, y = lap.lapjv(inp_arr, extend_cost=True)
print('cost:', cost)

B = 4
T = 1
N = 12


query_arr = np.random.rand(N, 2)
print(query_arr.shape)

ret_arr = np.random.rand(N, 2)
print(ret_arr.shape)

# compute the cost matrix
cost_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cost_matrix[i, j] = np.linalg.norm(query_arr[i] - ret_arr[j])
print(cost_matrix.shape)
print(cost_matrix)

cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
print('cost:', cost)