from dataset import Dataset
from Embeddings import Embeddings

import numpy as np

p = 113
embed_dim = 64
h_dimension = 128
output_dim = p

# embeddings matrix (p x embed_dim)
embeddings = Embeddings(p, embed_dim)

train_x, train_y, test_x, test_y = Dataset(p)
lr = 0.1

# define weights: input will be concatenation of two embeddings
w1 = np.random.randn(2 * embed_dim, h_dimension) / np.sqrt(2 * embed_dim)
b1 = np.zeros(h_dimension)
w2 = np.random.randn(h_dimension, output_dim) / np.sqrt(h_dimension)
b2 = np.zeros(output_dim)

# momentum buffers
vw1 = np.zeros_like(w1)
vb1 = np.zeros_like(b1)
vw2 = np.zeros_like(w2)
vb2 = np.zeros_like(b2)
vE = np.zeros_like(embeddings)
beta = 0.9

weight_decay = 1e-3  # L2 regularization


def relu(x):
    return np.maximum(0.0, x)


def forward(x):
    # x: (batch, 2) indices
    e1 = embeddings[x[:, 0].astype(int)]
    e2 = embeddings[x[:, 1].astype(int)]
    inp = np.concatenate([e1, e2], axis=1)  # (batch, 2*embed_dim)

    z1 = inp @ w1 + b1
    Activation = relu(z1)
    logits = Activation @ w2 + b2
    return inp, z1, Activation, logits


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    n = y.shape[0]
    eps = 1e-12
    correct_logprobs = -np.log(np.clip(probs[np.arange(n), y], eps, 1.0))
    return np.mean(correct_logprobs)


def backward(x, y, inp, z1, Activation, logits, learning_rate=lr):
    """
    Backprop including embeddings.
    """
    global w1, b1, w2, b2, embeddings, vw1, vb1, vw2, vb2, vE

    n_samples = x.shape[0]

    # softmax
    probs = softmax(logits)

    # loss gradient wrt logits
    dlogits = probs.copy()
    dlogits[np.arange(n_samples), y] -= 1
    dlogits /= n_samples

    # grads for w2/b2
    dw2 = Activation.T @ dlogits
    db2 = np.sum(dlogits, axis=0)

    # backprop into hidden
    dhidden = dlogits @ w2.T
    dz1 = dhidden * (z1 > 0)

    # grads for w1/b1
    dw1 = inp.T @ dz1
    db1 = np.sum(dz1, axis=0)

    # gradient w.r.t input embeddings before updating w1
    dinput = dz1 @ w1.T  # (batch, 2*embed_dim)
    de1 = dinput[:, :embed_dim]
    de2 = dinput[:, embed_dim:]

    grad_E = np.zeros_like(embeddings)
    np.add.at(grad_E, x[:, 0].astype(int), de1)
    np.add.at(grad_E, x[:, 1].astype(int), de2)

    # momentum-style update without extra scaling by (1-beta)
    vw1 = beta * vw1 + (dw1+weight_decay * w1)
    vb1 = beta * vb1 + db1
    vw2 = beta * vw2 + (dw2+weight_decay * w2)
    vb2 = beta * vb2 + db2
    vE = beta * vE + (grad_E+weight_decay * embeddings)

    w1 -= learning_rate * vw1
    b1 -= learning_rate * vb1
    w2 -= learning_rate * vw2
    b2 -= learning_rate * vb2
    embeddings -= learning_rate * vE

    # compute loss for reporting
    probs_for_loss = softmax(logits)
    loss = cross_entropy_loss(probs_for_loss, y)
    return loss

def test_accuracy(test_x, test_y):
    _, _, _, logits = forward(test_x)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == test_y)
    return accuracy * 100

def train_accuracy(train_x, train_y):
    _, _, _, logits = forward(train_x)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == train_y)
    return accuracy * 100


def train(train_x, train_y, epochs=70000, learning_rate=0.02):
    for i in range(epochs):
        

        inp, z1, Activation, logits = forward(train_x)

        probs = softmax(logits)
        loss = cross_entropy_loss(probs, train_y)

        _ = backward(train_x, train_y, inp, z1, Activation, logits, learning_rate)

        
        if i % 50 == 0:
            test_acc = test_accuracy(test_x, test_y)
            train_acc = train_accuracy(train_x, train_y)
            print(f"Epoch {i}, Loss: {loss}, Test Accuracy: {test_acc}%, Train Accuracy: {train_acc}%")


if __name__ == "__main__":
    train(train_x, train_y)