import numpy as np

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
        try: 
            img = Image.open(payload["image"].file)
        except:
            img = Image.open(payload["image"].file)

        orig = img.copy()

        img_tensor = image.img_to_array(img)
        img_tensor = img_tensor.resize((64,64))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        print("[SHAPE]", img_tensor.shape)
        #img_tensor = img_tensor/255.
  

        return 0 #self.model.predict(img_tensor)[0][0]