
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import keras.backend as K
from datetime import datetime as dt
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
import uuid
from PIL import Image
import os
import tempfile
from keras.models import load_model
import imageio
from keras.preprocessing import image



def resize_image_pnm(image):
    resized_image = cv2.resize(image, (128,128)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image



"""Instantiating the flask object"""
app = Flask(__name__)
CORS(app)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/checkup')
def checkup():
    return render_template('checkup.html')



@app.route('/malaria.html')
def malaria():
    return render_template('malaria.html')



@app.route('/pnm.html')
def pnm():
    return render_template('pnm.html')

@app.route('/retino.html')
def retino():
    return render_template('retino.html')

@app.route('/index.html')
def index_from_checkup():
    return render_template('index.html')

@app.route('/checkup.html')
def checkup_from_any():
    return render_template('checkup.html')

@app.route('/blog.html')
def blog():
    return render_template('blog.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route("/", methods = ["POST", "GET"])
def index():
  if request.method == "POST":
    type_ = request.form.get("type", None)
    data = None
    final_json = []
    print(type_)
    if 'img' in request.files:
      file_ = request.files['img']
      name = os.path.join(tempfile.gettempdir(), str(uuid.uuid4().hex[:10]))
      file_.save(name)
      print("[DEBUG: %s]"%dt.now(),name)

      if(type_=="mal"):
        test_image = image.load_img(name, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data=test_image



      elif(type_=='pnm' or type_=='dia_ret'):
        test_image = Image.open(name)                                  #Read image using the PIL library
        test_image = test_image.resize((128,128), Image.ANTIALIAS)     #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                              #Convert the image to numpy array
        test_image = test_image/255                                    #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)                #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image


      



      model=get_model(type_)[0]

      if(type_=='mal'):
         preds, pred_val = translate_malaria(model["model"].predict(data))
         final_json.append({"empty": False, "type":model["type"], 
                            "para":preds[0], 
                            "unin":preds[1],
                            "pred_val": pred_val})
      
      elif(type_=='pnm'):
         preds, pred_val = translate_pnm(model["model"].predict(data))
         print(preds[0])
         final_json.append({"empty": False, "type":model["type"], 
                            "viral":preds[0], 
                            "normal":preds[1],
                            "pred_val": pred_val})

      
      elif(type_=='dia_ret'):
         preds, pred_val = translate_retinopathy(model["model"].predict(data))
         final_json.append({"empty": False, "type":model["type"], 
                            "mild":preds[0], 
                            "norm":preds[1],
                            "pred_val": pred_val})
      
    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      pred_val =" "
      final_json.append({"pred_val": warn,"para": " ","unin": " ","viral": " ","normal": " ",
                         "mild": " ","norm": " "}) 

    K.clear_session()
    return jsonify(final_json)
  return jsonify({"empty":True})

"""This function is used to load the model from disk."""
def load_model_(model_name):
  model_name = os.path.join("static/weights",model_name)
  model = load_model(model_name)
  return model

"""This function is used to load the specific model for specific request calls. This
function will return a list of dictionary items, where the key will contain the loaded
models and the value will contain the request type."""
def get_model(name = None):
  model_name = []
  if(name=='mal'):
    model_name.append({"model": load_model_("m_v3.h5"), "type": name})
  elif(name=='pnm'):
    model_name.append({"model": load_model_("p_v3.h5"), "type": name})
  elif(name=='dia_ret'):
    model_name.append({"model": load_model_("dr_v3.h5"), "type": name})

  return model_name

"""preds will contain the predictions made by the model. We will take the class probabalities and 
store them in individual variables. We will return the class probabilities and the final predictions
made by the model to the frontend. The value contained in variables total and prediction will be
displayed in the frontend HTML layout."""
def translate_malaria(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  para_prob="Probability of the cell image to be Parasitized: {:.2f}%".format(y_proba_Class1)
  unifected_prob="Probability of the cell image to be Uninfected: {:.2f}%".format(y_proba_Class0)

  total = para_prob + " " + unifected_prob
  total = [para_prob,unifected_prob]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The cell image shows strong evidence of Malaria."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence of Malaria."
      return total,prediction


def translate_pnm(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  viral="Probability of the Chest X-ray to be infected: {:.2f}%".format(y_proba_Class1)
  normal="Probability of the Chest X-ray to be Uninfected: {:.2f}%".format(y_proba_Class0)
  

  total = viral + " " + normal
  total = [viral,normal]

  if (y_proba_Class1 >   y_proba_Class0):
      
      prediction="Inference: The Chest X-ray shows strong evidence of pneumonia."
      return total,prediction
  else:
      prediction="Inference: The Chest X-ray shows no evidence of pneumonia."
      return total,prediction


def translate_retinopathy(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  mild="Probability of the fundus image to be Parasitized: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the fundus image to be Uninfected: {:.2f}%".format(y_proba_Class0)

  total = mild + " " + norm

  total = [mild,norm]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The fundus image shows strong evidence of DR."
      return total,prediction
  else:
      prediction="Inference: The fundus image shows no evidence of DR."
      return total,prediction



#main file to run

if __name__=="__main__":
  app.run()