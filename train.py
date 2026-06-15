from dataset import Dataset

import numpy as np

p=113
h_dimension=128

teain_x,train_y,test_x,test_y=Dataset(p)
lr=0.03

#define weights
w1=np.random.randn(p,h_dimension)/np.sqrt(2/h_dimension)
b1=np.zeros(h_dimension)
w2=np.random.randn(h_dimension,p)/np.sqrt(2/h_dimension)
b2=np.zeros(h_dimension)


def relu(x):
    return np.maximum(0.0,x)

def forward(x):
    z1=x @ w1
    Activation=relu(z1)
    logits=Activation@w2
    return z1,Activation,logits

def backward(x,y,z1,Activation,logits):
    """
    forward:
          X->z1->Activation->logits->probs->loss
    backward:
          dloss/dlogits=probs-y
          dloss/dw2=Activation.T @ dloss/dlogits

          dloss/db2=np.sum(dloss/dlogits,axis=0)

          dloss/dActivation=dloss/dlogits * w2.T
          dloss/dz1=dloss/dActivation * relu'(z1)

          dloss/dw1=x.T @ dloss/dz1
          dloss/db1=np.sum(dloss/dz1,axis=0)

    """
    probs=logits-y

    dw2=Activation.T@probs

    db2=np.sum(probs,axis=0)

    dhidden=probs@w2.T

    dz1=dhidden*(z1>0)
    dw1=x.T@dz1
    db1=np.sum(dz1,axis=0)

    w1=lr*dw1
    b1=lr*db1
    w2=lr*dw2
    b2=lr*db2

    return np.mean(np.square(probs))

def train(train_x,train_y,epochs=1000,learning_rate=1e-3):
    for i in range(epochs):
        z1,Activation,logits=forward(train_x)

        loss=backward(train_x,train_y,z1,Activation,logits)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

if __name__ == "__main__":
    train(teain_x,train_y)