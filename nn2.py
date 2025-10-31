import numpy as np 
import matplotlib.pyplot as plt

with np.load("mnist.npz") as f:
    images, labels = f["x_train"], f["y_train"]
images = images.astype("float32") / 255
images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
labels = np.eye(10)[labels]

learning_rate = 0.05
layers = images.shape[1], 40, 10
layers_count = len(layers)
w = []
b = []
for i in range(1,layers_count):
    w.append(np.random.uniform(-0.5,0.5,(layers[i], layers[i-1])))
    b.append(np.zeros(layers[i]))

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
def logit(x):
    return np.log(x / (1 - x))
def train(image, label):
    global w, b, learning_rate, layers_count
    y = label
    z = [0]
    a = [image]
    bd = [0]*(layers_count-1)
    wd = [0]*(layers_count-1)
    
    # forwardprop
    for i in range(layers_count-1):
        z.append(np.sum(a[i]*w[i], axis=1) + b[i])
        a.append(sigmoid(z[i+1]))
    
    # backprop
    bd[-1] = (a[-1]-y)*sigmoid_derivative(z[-1])
    wd[-1] = bd[-1].reshape(-1,1)*a[-2]
    for i in range(-2,-layers_count,-1):
        bd[i] = np.average(bd[i+1].reshape(-1,1)*w[i+1], axis=0)*sigmoid_derivative(z[i])
        wd[i] = bd[i].reshape(-1,1)*a[i-1]
    for i in range(layers_count-1):
        w[i] -= wd[i]
        b[i] -= bd[i]

    return np.square(a[-1]-label).sum()

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
        z = [0]
        a = [images[x]]
        y = labels[x]
        plt.imshow(a[0].reshape(28, 28), cmap="Greys")
        plt.show()
        for i in range(layers_count-1):
            z.append(np.sum(a[i]*w[i], axis=1) + b[i])
            a.append(sigmoid(z[i+1]))
        print(f"Predict: {a[-1].argmax()}; Actual: {y.argmax()}; Loss: {np.square(a[-1]-y).sum()}")
    except:
        pass