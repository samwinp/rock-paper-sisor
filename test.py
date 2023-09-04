import tensorflow as tf

model = tf.keras.models.load_model('./face_model/face_model.h5', compile=False)
print(model)