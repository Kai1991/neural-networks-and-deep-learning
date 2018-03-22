import numpy as np

import random

class network(object):

    ###初始化权重和偏置
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for zip(self.sizes[1:], self.sizes[:-1])]

    ###训练神经网络入口
    ###training_data：训练数据；epochs：训练频次；mini_batch_size：最小批次数量；etd:学习速率；test_data：测试数据
    def SGD(self,training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data) ###打乱数据
            mini_batchs = [training_data[k : k + mini_batch_size] for k in xrange(0, n ,mini_batch_size)] ###将训练数据分解成小批次数据

            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch)


    def update_mini_batch(self,mini_batch):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nable_b , delta_nale_w = self.backprop(x,y)####计算梯度
            nable_b = [ nb + dnb for nb , dnb in zip(nable_b,delta_nable_b)]
            nable_w = [ nw + dnw for nw , dnw in zip(nable_w,delta_nale_w)]

            self.weights = [w - (eta/len(mini_batch))* nw for w , nw in zip(self.weights,nable_w)]
            self.biases = [b - (eta/len(mini_batch))* nb for b , nb in zip(self.biases,nable_b)]


    def backprop(self, x, y):
        nable_b = [np.zeros(b) for b in self.biases]
        nable_w = [np.zeros(w) for w in self.weights]

        ###计算每一层的激活值，和带权和
        activation = x
        activations = [x]

        zs= [];
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        ###计算每一层的的梯度值
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta,activations[-2].transpose())

        for l in xrange(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l - 1].transpose(),delta) * sp
            nable_b[-l] = delta;
            nable_w[-l] = np.dot(delta,activations[-l - 1].transpose())

        return (nable_b,nable_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def evaluate(self , test_data):



def sigmoid(z):
    return  1.0/(1.0 + np.exp(-z)) ###激活函数

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))



