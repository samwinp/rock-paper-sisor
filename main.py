from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import io
from io import BytesIO
from PIL import Image, ImageOps
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)


model = tf.keras.models.load_model('./rock')
class_name = ['paper', 'rock', 'sisor']

rpc_model = tf.keras.models.load_model('./keras_model.h5', compile=False)
rpc_names = ['rock', 'paper', 'sisor']

# criminal_model = tf.keras.models.load_model('./face_model/face_model.h5', compile=False)
# criminal_name = ['samwin', 'person2']



@app.get("/")
async def root():
    return {"message": "Hello World"}


# @app.post('/predict')
# async def predict(file: UploadFile = File(...)):
#     contents =  await file.read()
#     image = Image.open(BytesIO(contents)) 
#     image = np.array(image)
  
    
#     return {
#         "prediction" : 1,
#         "image" : image        
#     }


class User(BaseModel):
    name: str
    email: str

class Vai(BaseModel):
    data : str

@app.post('/user')
async def user(user : User):
    print(user)
    return user
    
@app.get("/test")
def test():
    return {"message": "test"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    print('hi')   


    return {'did' : file.filename }



@app.post('/image')
async def get_image(file : UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image_re = image.resize((256, 256))
    image_np = np.array(image_re)
    print(image_np.shape)    
    image_batch = np.expand_dims(image_np, 0)
    prediction = model.predict(image_batch)    
    answer = class_name[np.argmax(prediction[0])]

    # print(image_data)
    
    return {'file' : answer }

@app.post('/pinto')
async def use_this(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image))
    print(image)
    size = (224, 224)
    image_resize = image.resize(size)
    image_np = np.array(image_resize)
    # print(image_np.shape)
    normalized_image = (image_np.astype(np.float32) / 127.5) - 1
    # print(normalized_image)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)    
    data[0] = normalized_image
    print(data.shape)
    prediction = rpc_model.predict(data)     
    index = np.argmax(prediction)
    ans_names = rpc_names[index]
    print(ans_names)
    confidence = prediction[0][index]
    confidence = f"{confidence}"

    return {"hand" : ans_names , "confidence" : confidence}
    return {'pls' : "work"}
   


# @app.post('/charis')
# async def dbms(file: UploadFile = File(...)):
#     image = await file.read()
#     image = Image.open(BytesIO(image))
#     print(image)
#     size = (224, 224)
#     image_resize = image.resize(size)
#     image_np = np.array(image_resize)
#     # print(image_np.shape)
#     normalized_image = (image_np.astype(np.float32) / 127.5) - 1
#     # print(normalized_image)
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)    
#     data[0] = normalized_image
#     print(data.shape)
#     prediction = criminal_model.predict(data)     
#     index = np.argmax(prediction)
#     ans_names = criminal_name[index]
#     print(ans_names)
#     confidence = prediction[0][index]
#     confidence = f"{confidence}"

#     return {"hand" : ans_names , "confidence" : confidence}
#     return {'pls' : "work"}





@app.post('/vaibav')
def dbms_vai(name : Vai):
    print(name)
    return {"name" : name.data}