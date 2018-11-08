import random
import numpy as np


x = 0
y = 0

def move(x,y):
    flag = 0
    while flag == 0:
        c = random.randint(1,8)
        if c == 1:
            if x - 2 >= 0 and y - 1 >= 0:
                x = x - 2
                y = y - 1
                flag = 1
        elif c == 2:
            if x - 1 >= 0 and y - 2 >= 0:
                x = x - 1
                y = y - 2
                flag = 1
        elif c == 3:
            if x - 2 >= 0 and y + 1 <= 7:
                x = x - 2
                y = y + 1
                flag = 1
        elif c == 4:
            if x - 1 >= 0 and y + 2 <= 7:
                x = x - 1
                y = y + 2
                flag = 1
        elif c == 5:
            if x + 2 <= 7 and y + 1 <= 7:
                x = x + 2
                y = y + 1
                flag = 1
        elif c == 6:
            if x + 1 <= 7 and y + 2 <= 7:
                x = x + 1
                y = y + 2
                flag = 1
        elif c == 7:
            if x + 2 <= 7 and y - 1 >= 0:
                x = x + 2
                y = y - 1
                flag = 1
        elif c == 8:
            if x + 1 <= 7 and y - 2 >= 0:
                x = x + 1
                y = y - 2
                flag = 1
    return x, y

a = np.zeros((8,8))
n=200000
for i in range(0,n):
    x,y = move(x,y)
    a[x][y] += 1
for i in range(0,8):
    for j in range(0,8):
        a[i][j] = a[i][j]/n
print(a)

