#env
import numpy as np 
import random
import tensorflow as tf

class chessboard:
    hash_table = [1, 16, 256, 4096, 
                  65536, 1048576, 16777216, 268435456,
                  4294967296, 68719476736, 1099511627776,  17592186044416,
                  281474976710656, 4503599627370496, 72057594037927936, 1152921504606846976]
    real_value = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 
                  256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0]
    my_reward = [0, 0, 0, 1, 1, 4, 4, 16, 32,  64, 256, 1024, 4096, 4096, 4096]
    boardhash = 0.0
    score = 0.0
    n_features = 16
    n_actions = 4
    maxgrid = 0.0
    alive = True
    actions = ["a", "w", "d", "s"]
    map = np.zeros((4,4), dtype = float)
    orimap = map.copy()

    def __init__(self, name):
        self.name = name
        
    
    def maintain(self):
        self.alive = self.falive()
        self.maxgrid = np.max(self.map)
        self.score = np.sum(self.map)
        self.boardhash = self.getboardhash()
        self.orimap = self.map.copy()
        return self.map.flatten()

    def showmap(self):
        print(self.map, '\n')
    
    def put_rand(self):
        temp = [[],[]]
        for i in range(0, 4):
            for j in range(0, 4):
                if self.map[i][j] == 0:
                    temp[0].append(i)
                    temp[1].append(j)
        n = len(temp[1])
        tar = np.random.uniform(0, n)
        for i in range(0, n):
            if tar >= i and tar < i+1 :
                self.map[temp[0][i]][temp[1][i]] = 1 if random.uniform(0, 10) > 1 else 2
        return self.map[temp[0][i]][temp[1][i]]

    def get_map(self):
        return self.map.copy()

    def retreat(self):
        self.map = self.orimap.copy()

    def getboardhash(self):
        temp = 0.0
        a = 16
        [row, col] = self.map.shape
        for i in range(row):
            for j in range(col):
                temp+=self.map[i][j] * self.hash_table[i * 4 + j]
        return temp

    def render(self):
        self.maintain()
        return map.copy().flatten()

    def step(self, actions):
        if(self.move(self.actions[int(actions[0])]) == True):
            x = self.put_rand()
            arr = self.map.copy()
            return(arr.flatten(), self.getreward(x), 0, True) 
        else:
            if(self.move(self.actions[int(actions[1])]) == True):
                x = self.put_rand()
                arr = self.map.copy()
                return(arr.flatten(), self.getreward(x),1,  True)   
            else:
                if(self.move(self.actions[int(actions[2])]) == True):
                    x = self.put_rand()
                    arr = self.map.copy()                   
                    return(arr.flatten(), self.getreward(x),2,  True)  
                else:
                    if(self.move(self.actions[int(actions[3])]) == True):
                        x = self.put_rand()
                        arr = self.map.copy()
                        return(arr.flatten(), self.getreward(x),3,  True) 
                    else:
                        arr = self.map.copy()
                        return(arr.flatten(), 0, 0, False)

    def getreward(self, x):
        [rows, cols] = self.map.shape
        reward = 0
        reward += self.real_value[int(self.map.max())]
        for i in range(rows):
            for j in range(cols):
                reward += self.real_value[int(self.map[i][j])]
        return (reward/2048.0)
        #return(self.my_reward[int(self.map.max())] - self.my_reward[int(self.orimap.max())])

    def init(self):
        alive = True
        tempx = random.randint(0, 3)
        tempy = random.randint(0, 3)
        self.map[tempx][tempy] = 1 if random.randrange(0, 10) > 1 else 2
        self.maxgrid = np.max(self.map)
        self.score = np.sum(self.map)
        self.boardhash = self.getboardhash()
        #print(self.map, ' \n', self.maxgrid, ' ', self.score, ' ',self.boardhash)
        self.orimap = self.map.copy()
        arr = self.map.copy()
        return arr.flatten()
        
    def falive(self):
        flag = False
        temp = -1
        for i in range(0, 4):
            for j in range(0, 4):
                if(self.map[i][j] == 0):
                    flag = True
                    break
                if j<3:
                    if (self.map[i][j] == self.map[i][j+1]):
                        flag = True
                        break
                    if (self.map[j][i] == self.map[j+1][i]):
                        flag = True 
                        break
            if (flag == True ):
                break
        return flag
    
    def move(self, action):
        if action == 'a':
            buffer = []
            for i in range(0, 4):
                jj = -1
                first = 0
                while (self.map[i][first] == 0):
                    first += 1
                    if (first >= 3):
                        break
                buffer.append( self.map[i][first])
                jj = self.map[i][first]
                first += 1
                for j in range(first, 4):
                    if (self.map[i][j] == jj):
                        a = buffer.pop() + 1
                        buffer.append(a)
                        jj = -1
                    else: 
                        if (self.map[i][j] == 0):
                            pass
                        else:
                            jj = self.map[i][j]
                            buffer.append(jj)         
                a = buffer.pop()
                if (a != jj and jj != -1):
                    buffer.append(a)
                    buffer.append(jj)
                else:
                    buffer.append(a)
                while (len(buffer) <= 3):
                    buffer.append(0)
                for j in range(0, 4):
                    self.map[i][j] = buffer[j]
                buffer.clear()
        if action == 'w':
            buffer = []
            for i in range(0, 4):
                jj = -1
                first = 0
                while (self.map[first][i] == 0):
                    first += 1
                    if (first >= 3):
                        break
                buffer.append( self.map[first][i])
                jj =  self.map[first][i]
                first += 1
                for j in range(first, 4):
                    if (self.map[j][i] == jj):
                        a = buffer.pop() + 1
                        buffer.append(a)
                        jj = -1
                    else: 
                        if (self.map[j][i] == 0):
                            pass
                        else:
                            jj = self.map[j][i] 
                            buffer.append(jj)   
                a = buffer.pop()
                if (a != jj and jj != -1):
                    buffer.append(a)
                    buffer.append(jj)
                else:
                    buffer.append(a)
                while (len(buffer) <= 3):
                    buffer.append(0)
                for j in range(0, 4):
                    self.map[j][i] = buffer[j]
                buffer.clear()   
        if action == 'd':
            buffer = []
            for i in range(0, 4):
                jj = -1
                first = 3
                while (self.map[i][first] == 0):
                    first -= 1
                    if (first <= 0):
                        break
                buffer.append( self.map[i][first])
                jj = self.map[i][first]
                for j in range(0, first):
                    if (self.map[i][first - j - 1] == jj):
                        a = buffer.pop() + 1
                        buffer.append(a)
                        jj = -1
                    else: 
                        if (self.map[i][first - j - 1] == 0):
                            pass
                        else:
                            buffer.append(self.map[i][first - j - 1])
                            jj = self.map[i][first - j - 1]
                a = buffer.pop()
                if (a != jj and jj != -1):
                    buffer.append(a)
                    buffer.append(jj)
                else:
                    buffer.append(a)
                while (len(buffer) <= 3):
                    buffer.append(0)
                for j in range(0, 4):
                    self.map[i][3 - j] = buffer[j]
                buffer.clear()
        if action == 's':
            buffer = []
            for i in range(0, 4):
                jj = -1
                first = 3
                while (self.map[first][i] == 0):
                    first -= 1
                    if (first <= 0):
                        break
                buffer.append(self.map[first][i])
                jj = self.map[first][i]
                for j in range(0, first):
                    if (self.map[first - j - 1][i] == jj):
                        a = buffer.pop() + 1
                        buffer.append(a)
                        jj = -1
                    else: 
                        if (self.map[first - j - 1][i] == 0):
                            pass
                        else:
                            jj = self.map[first - j - 1][i]
                            buffer.append(jj)
                a = buffer.pop()
                if (a != jj and jj!= -1):
                    buffer.append(a)
                    buffer.append(jj)
                else:
                    buffer.append(a)
                while (len(buffer) <= 3):
                    buffer.append(0)
                for j in range(0, 4):
                    self.map[3 - j][i] = buffer[j]
                buffer.clear()   
        if(self.boardhash == self.getboardhash()):
            self.map = self.orimap.copy()
            return False
        else:            
            return True