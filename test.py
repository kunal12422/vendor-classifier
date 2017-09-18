import numpy as np
class A:
	def __init__(self, a,b):
		self._a = a
		self._b = b
	def hello(self):
		print(self._a)	

a = A(1,2)
# a.hello()	
mat = np.array([1,2,3])	
print(mat+1)