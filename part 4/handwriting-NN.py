import matplotlib.pyplot as plt  #visualization of digits
import tensorflow as tf #machine larning

mnist = tf.keras.datasets.mnist #dataset
#training data = data to train model
#testing data = data to test accuracy of model on new data
#usually 80-20

(x_train, y_train), (x_test, y_test) = mnist.load_data()    #returns 2 tuples with training and testing data

#normalization of data (everything gets scaled between 0 and 1)
#only pixels get normalized
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#standard sequential model
model = tf.keras.models.Sequential()

#adding layer to the model
#since the images are 28x28 pixels, the input size will be (28, 28)
#Flatten means that it is not a 28x28 grid, but rather a line of 28x28 neurons (784 neurons)
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
#Dense layer means that every neuron is connected to every previous neuron
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
#this will be our output layer, which has 1 neuron for each digit (0-9)
#softmax = sum of all neurons will be 1, neuron with biggest number means biggest probability of it being the right one
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#compiling the model, the metrics we are interested in are the accuracy
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#training the model with training data
model.fit(x_train, y_train, epochs=3)

model.save("handwritten.model")

#since we have saved the trained model, we can now just load it like so
model = tf.keras.models.load_model("handwritten.model")

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)



