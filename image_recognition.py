import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

model = resnet50.ResNet50()
# Image file => resizing it to 224x224 pixels (required by this model)
img = image.load_img("tesla.png", target_size=(224, 224))

x = image.img_to_array(img)
# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)
# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)
predictions = model.predict(x)
# Top 5 predictions of ResNet
predicted_classes = resnet50.decode_predictions(predictions, top=5)

print("This is an image of:")

for image_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))
