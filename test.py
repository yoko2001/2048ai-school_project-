from env import chessboard
import tensorflow as tf 
import numpy as np
import random 

#map =np.random.permutation(4)
#print(map)
counter1 = 0
counter2 = 0
for i in range(1000):
    a =  1 if random.uniform(0, 10) > 1 else 2
    if (a == 1):
        counter1 += 1
    if (a == 2):
        counter2 += 1
print(counter1, ' ', counter2)