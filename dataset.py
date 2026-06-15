#we are experimenting with a sample of numbers.
# we will  take numbers 1 to 113 and do modular addition of two numbers
import numpy as np
p=113

def Dataset(p):
    
    data=[(x,y) for x in range(p) for y in range(p)] #this will be our x

    modular=[(x+y)%p for x in range(p) for y in range(p)]  #this is our y

    data=np.array(data)
    modular=np.array(modular)

    #lets split the data

    perms= np.random.permutation(len(data))
    train_split= int(0.3*len(data))

    train_indices=perms[:train_split]
    test_indices=perms[train_split:]

    train_x=data[train_indices]
    test_x=data[test_indices]

    train_y=modular[train_indices]
    test_y=modular[test_indices]

    return train_x,train_y,test_x,test_y