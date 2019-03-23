import numpy as np
from scipy import sparse #scipy.sparse 稀疏矩阵
x = np.eye(4)
x = sparse.csr_matrix(x)
print(x)