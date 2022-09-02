from fileinput import filename
import numpy as np

total = 0

for i in range(50):
    index = i+1
    filename = "trainer_test_"+str(index)+".npy"
    r = np.load(filename)
    total = total+r
    print(r)

# print(total/50)