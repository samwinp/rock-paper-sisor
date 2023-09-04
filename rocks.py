from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

model = load_model("./keras_model.h5", compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


class_names = open('./labels.txt' , 'r').readlines() 
print(class_names[0])
print(model)

image = Image.open('./sisor.jpeg')
size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_re = image.resize(size)
image_np = np.array(image_re)

normalized_image_array = (image_np.astype(np.float32) / 127.5) - 1

data[0] = normalized_image_array

prediction = model.predict(data)
index = np.argmax(prediction)
class_names = class_names[index]
confidence = prediction[0][index]
print(f'{class_names} and {confidence}')