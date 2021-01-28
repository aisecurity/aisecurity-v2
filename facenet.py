from keras.models import load_model
import numpy as np
import cv2

model = load_model("facenet_keras.h5")
im = cv2.imread("zayn.jpg")


def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = np.expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

embeddings = np.array(get_embedding(model, im))
print(embeddings)