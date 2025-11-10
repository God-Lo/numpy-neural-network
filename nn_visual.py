import numpy as np 
import matplotlib.pyplot as plt
import vpython as vp

with np.load("mnist.npz") as f:
    images, labels = f["x_train"], f["y_train"]
images = images.astype("float32") / 255
images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
labels = np.eye(10)[labels]

learning_rate = 0.05
layers = images.shape[1], 20, 20, 20, 10
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

def show_nn(image, label):
    global w, b, learning_rate, layers_count
    
    y = label
    z = [0]
    a = [image]
    bd = [0]*(layers_count-1)
    wd = [0]*(layers_count-1)
    
    for i in range(layers_count-1):
        z.append(np.sum(a[i]*w[i], axis=1) + b[i])
        a.append(sigmoid(z[i+1]))

    distant = 20
    scene = vp.canvas(width=1200, height=800, center=vp.vector(0, 0, 0), background=vp.vec(0.6,0.8,0.8))
    layers_balls = [[] for i in range(layers_count)]
    lines = [[] for i in range(layers_count)]
    layers_balls[0] = [vp.sphere(pos=vp.vec((i%28)-13.5,int(i/28)*-1+13.5,0), radius=0.5, color=vp.color.hsv_to_rgb(vp.vec(0,0,image[i]))) for i in range(784)]
    
    input()
    
    for i in range(28):
        vp.rate(20)
        for j in range(784):
            layers_balls[0][j].pos.x += int(j/28)-13.5  
    for i in range(27):
        vp.rate(20)
        for j in range(784):
            if layers_balls[0][j].pos.y < 0:
                layers_balls[0][j].pos.y += 0.5
            elif layers_balls[0][j].pos.y > 0:
                layers_balls[0][j].pos.y -= 0.5
    
    for layer in range(1,layers_count):
        
        input()
        
        layers_balls[layer] = [vp.sphere(pos=vp.vec(i*2-layers[layer]+1,0,layers_balls[layer-1][0].pos.z+(distant*2 if layer==1 else distant)), radius=0.5, color=vp.color.black) for i in range(layers[layer])]
        lines[layer] = [[] for i in range(layers[layer])]
        for current_layer in range(layers[layer]):
            for previous_layer in range(layers[layer-1]):
                vp.rate(layers[layer-1]*5)
                c = w[layer-1][current_layer][previous_layer]
                c = vp.vec(0,0,sigmoid(c)) if c > 0 else vp.vec(sigmoid(abs(c)),0,0)
                lines[layer][current_layer].append(vp.cylinder(color=c, radius=0.1, pos=layers_balls[layer-1][previous_layer].pos, axis=layers_balls[layer][current_layer].pos-layers_balls[layer-1][previous_layer].pos))
            layers_balls[layer][current_layer].color = vp.color.hsv_to_rgb(vp.vec(0,0,a[layer][current_layer]))
            for line in lines[layer][current_layer]:
                line.color += vp.vec(0.5,0.5,0.5)
                line.radius = 0.05
        scene.center = vp.vec(0,0,layers_balls[layer][0].pos.z)
        
        if layer == 1:
            input()
            for i in range(27):
                for j in range(784):
                    if layers_balls[0][j].pos.y < int(j/28)*-1+13.5:
                        layers_balls[0][j].pos.y += 0.5
                    elif layers_balls[0][j].pos.y > int(j/28)*-1+13.5:
                        layers_balls[0][j].pos.y -= 0.5
                    for k in range(layers[1]):
                        lines[layer][k][j].pos = layers_balls[0][j].pos
                        lines[layer][k][j].axis = layers_balls[1][k].pos - layers_balls[0][j].pos
            for i in range(28):
                for j in range(784):
                    layers_balls[0][j].pos.x -= int(j/28)-13.5
                    for k in range(layers[1]):
                        lines[layer][k][j].pos = layers_balls[0][j].pos
                        lines[layer][k][j].axis = layers_balls[1][k].pos - layers_balls[0][j].pos
    
    y_balls = [vp.sphere(pos=vp.vec(i*2-9,0,layers_balls[-1][0].pos.z+distant/2), radius=0.5, color=vp.color.hsv_to_rgb(vp.vec(0,0,y[i]))) for i in range(10)]
    input()
    scene.delete()

epochs = int(input("Epoch: "))
for epoch in range(epochs):
    loss = 0
    for iamge, label in zip(images, labels):
        loss += train(iamge,label)/len(images)
    print(f"Epoch: {epoch}; Loss: {loss}")
print("Finished training...")

with np.load("C:\\Users\\ethan\\Desktop\\Python\\Machine Learning\\numpy\\mnist.npz") as f:
    images, labels = f["x_test"], f["y_test"]
images = images.astype("float32") / 255
images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
labels = np.eye(10)[labels]

while True:
    x = int(input("Test: "))

    show_nn(images[x],labels[x])
