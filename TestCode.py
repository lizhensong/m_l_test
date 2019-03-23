import numpy
X=[[1,2],[4,5],[7,8]]
X=numpy.array(X)
print(X)
y=numpy.array([1,0,1,0,0,1])
z=y==0
print(z)
print(X[~0,0])
print(numpy.hstack([X,X[:, 1:]]))