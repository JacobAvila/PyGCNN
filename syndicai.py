import numpy as np
import json

from keras.preprocessing import image
from keras import models


from PIL import Image



args = {
    "image": "sample/cat.10.jpg",
    "perrosygatos": "clasificador",
    "model": "RedCNN_PerrosyGatos.h5",
}


class PythonPredictor:

    def __init__(self, config):
        # cargamos el clasificador de imagenes desde el disco
        print("[INFO] cargando el modelo entrenado...")
        self.model = models.load_model(args["model"])

    def predict(self, payload):

        # Obtenemos la imagen del post
        img = Image.open(payload["image"].file)
        print("[SIZE]", img.size)
        img = img.resize((64,64))
        print("[RESIZE]", img.size)
        img_tensor = np.array(img)
        #img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        print("[SHAPE]", img_tensor.shape)
        img_tensor = img_tensor/255.
  
        resultado = np.round(self.model.predict(img_tensor)[0][0])
        valor = "Perro"
        if resultado == 0:
            valor = "Gato"
        
        res = {"resultado": valor}

        return json.dumps(res)