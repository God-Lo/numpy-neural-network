import numpy as np 
import matplotlib.pyplot as plt

with np.load("mnist.npz") as f:
    images, labels = f["x_train"], f["y_train"]
images = images.astype("float32") / 255
images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
labels = np.eye(10)[labels]

learning_rate = 0.5
layer = images.shape[1], 20, 15, 10
w0 = np.random.uniform(-0.5,0.5,(layer[1], layer[0]))
w1 = np.random.uniform(-0.5,0.5,(layer[2], layer[1]))
w2 = np.random.uniform(-0.5,0.5,(layer[3], layer[2]))
b0 = np.zeros(layer[1])
b1 = np.zeros(layer[2])
b2 = np.zeros(layer[3])

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
def train(image, label):
    global w0, w1, w2, b0, b1, b2
    y = label
    a0 = image
    
    # forwardprop
    z1 = np.sum(a0*w0, axis=1) + b0
    a1 = sigmoid(z1)
    z2 = np.sum(a1*w1, axis=1) + b1
    a2 = sigmoid(z2)
    z3 = np.sum(a2*w2, axis=1) + b2
    a3 = sigmoid(z3)
    
    # backprop
    y_d_b2 = (a3-y)*sigmoid_derivative(z3)
    y_d_w2 = y_d_b2.reshape(-1,1)*a2
    y_d_b1 = np.average(y_d_b2.reshape(-1,1)*w2, axis=0)*sigmoid_derivative(z2)
    y_d_w1 = y_d_b1.reshape(-1,1)*a1
    y_d_b0 = np.average(y_d_b1.reshape(-1,1)*w1, axis=0)*sigmoid_derivative(z1)
    y_d_w0 = y_d_b0.reshape(-1,1)*a0
    
    w2 -= learning_rate*y_d_w2
    b2 -= learning_rate*y_d_b2
    w1 -= learning_rate*y_d_w1
    b1 -= learning_rate*y_d_b1
    w0 -= learning_rate*y_d_w0
    b0 -= learning_rate*y_d_b0

    return np.square(a3-label).sum()

epochs = int(input("Epoch: "))
for epoch in range(epochs):
    loss = 0
    for iamge, label in zip(images, labels):
        loss += train(iamge,label)/len(images)
    print(f"Epoch: {epoch}; Loss: {loss}")
print("Finished traing...")

with np.load("C:\\Users\\ethan\\Desktop\\Python\\Machine Learning\\numpy\\mnist.npz") as f:
    images, labels = f["x_test"], f["y_test"]
images = images.astype("float32") / 255
images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
labels = np.eye(10)[labels]

while True:
    x = int(input("Test: "))
    try:
        a0 = images[x]
        y = labels[x]
        plt.imshow(a0.reshape(28, 28), cmap="Greys")
        plt.show()
        z1 = np.sum(a0*w0, axis=1) + b0
        a1 = sigmoid(z1)
        z2 = np.sum(a1*w1, axis=1) + b1
        a2 = sigmoid(z2)
        z3 = np.sum(a2*w2, axis=1) + b2
        a3 = sigmoid(z3)
        print(f"Predict: {a3.argmax()}; Actual: {y.argmax()}; Loss: {np.square(a3-y).sum()}")
    except:
        pass