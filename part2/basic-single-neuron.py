import math

class Neuron:
    weights = [1, 0.0001, -2]

    def activationFunction(self, ap):
        # sigmoid function
        return 1 / (1 + math.exp(-ap))

    def output(self, inputs):
        ap = 0
        for i in range(len(inputs)):
            ap += inputs[i] * self.weights[i]

        return self.activationFunction(ap)
    
neuron = Neuron()

print( neuron.output([1,1,1]) )
