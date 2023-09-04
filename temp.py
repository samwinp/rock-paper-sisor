from fastapi import FastAPI, UploadFile,File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel


app = FastAPI()

class ImageType(BaseModel):
    image: UploadFile = File(...)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

model = tf.keras.models.load_model('./rock')

class_name = ['paper', 'rock', 'sissor']


# def read_file_as_image(data) -> np.ndarray:
#     return(np.array(Image.open(BytesIO(data))))

    
    

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/sayhi")
def sayhi():
    print('Called say hi')
    return {"message": "say hi"} 


@app.get('/hey')
def hey():
    return {"message": "hey"}


@app.post('/predict')
async def predict(file: ImageType):   
    contents =  await file.image.read()   
    image = Image.open(BytesIO(contents))
    # resized_image = image.resize((256, 256))
    # image = np.array(resized_image)
    # image_batch = np.expand_dims(image, 0)
    # prediction = model.predict(image_batch)
    # prediction = class_name[np.argmax(prediction[0])]
    # confidence = np.max(prediction)
    # print(image.shape)
    
    return {
        'prediction' : 1,
        'confidence' : 2
    }

    # image_batch = np.expand_dims(image, 0)
    # print(image_batch)
    # prediction = model.predict(image_batch)        
    # print(prediction)
    # predicted_class =  class_name[np.argmax(prediction[0])]
    # confidence = np.max(prediction[0])
    # return {
    #     'class': predicted_class,
    #     'confidence': confidence
    # }




### some random work that is did to check how to open an image and get pred from the model
# import matplotlib.pyplot as plt
# from PIL import Image
# from io import BytesIO
# import tensorflow as tf
# import numpy as np
# import cv2

# class_names = ['paper', 'rock', 'sisor']

# img = Image.open('./sam.jpeg')
# data = np.array(img)

# print(data)
# print(data.shape)
# plt.imshow(img)
# data = cv2.resize(data, (256, 256))

# plt.colorbar()
# plt.show()

# model = tf.keras.models.load_model('./rock')

# image_batch = np.expand_dims(data, 0)
# print(data.shape)

# print(image_batch.shape)

# prediction = model.predict(image_batch)
# print(prediction[0])
# print(class_names[np.argmax(prediction[0])])

