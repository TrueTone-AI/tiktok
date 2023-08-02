import os
import cv2  #computer vision
import numpy as np  #arrays
import matplotlib.pyplot as plt  #visualization of digits
import tensorflow as tf #machine larning

model = tf.keras.models.load_model("handwritten.model")

image_number = 0
#while there is still a file in the directory
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:    #try to read the image, [:,:,0] = we don't care about colors, but just about shapes
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        #invert since now it is white on black, not black on white
        img = np.invert(np.array([img]))
        #the model makes a prediction
        prediction = model.predict(img)
        #it prints just the index of the prediction with the highest probability
        print(f"the number is {np.argmax(prediction)}")
        #showing the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    
    except:
        print("error")
    finally:
        image_number += 1