import math

class Neuron:
    weights = [1, -2]

    def activationFunction(self, pa):
        # sigmoid function
        #return 1 / (1 + math.exp(-pa))
        #threshold
        if pa > 0.0:
            return 1.0
        else:
            return 0.0
 
    def output(self, inputs):
        pa = 0
        for i in range(len(inputs)):
            pa += inputs[i] * self.weights[i]

        return self.activationFunction(pa)
    
    def learn(self, x, target):
        out = self.output(x)
        error = target - out

        print('input', x, 'target', target, 'error', error)

        learning_rate = 0.01

        for i, w in enumerate(self.weights):
            self.weights[i] += learning_rate * error * x[i]  * self.activationFunction(x[i])
    
neuron = Neuron()

X =  [
    [1,1],
    [1,0],
    [0,1],
    [0,0]
]

Y = [0, 1, 0, 0]

for epoch in range(100):
    print('epoch', epoch)
    for i, x in enumerate(X):
        target = Y[i]
        neuron.learn(x, target)
