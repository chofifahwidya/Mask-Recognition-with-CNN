import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
model = load_model('model-007.model')

"""model_name = 'masker training'"""
"""tf.keras.models.save_model(model_name,model_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model_name)
tflite_model = converter.convert()
with open("tf_model.tflite", "wb") as tf:
  tf.write(tflite_model)
interpreter = tf.lite.Interpreter(model_path='tf_model.tflite')
interpreter.get_tensor_details()"""
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
"""load gambar dari webcam"""
""" ret, img =source.read()"""
source=cv2.VideoCapture(0)
"""Load gambar dari Library"""

labels_dict={0:'NO MASK',1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}
while (True):

    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
        print(result)
        label = np.argmax(result, axis=1)[0]

        fix_label = f'{labels_dict[label]} {result[0][label]}'

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, fix_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
source.release()