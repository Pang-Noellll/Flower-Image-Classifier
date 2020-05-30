
import json
import tensorflow as tf
import argparse
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224,224))
  image /= 255
  #image = np.expand_dims(image,axis =0)
  #image = np.asarray(image)
  return image

def predict(image_path,model,top_k):
  im = Image.open(image_path)
  test_image = np.asarray(im)
  processed_test_image = process_image(test_image)
  processed_test_image = np.expand_dims(processed_test_image,axis =0)

  pred_prop = model.predict(processed_test_image)
  pred_prop = np.squeeze(pred_prop)
  top_idx = np.argsort(pred_prop)[-top_k:]
  top_values = [pred_prop[i] for i in top_idx]

  return top_values, top_idx


parser = argparse.ArgumentParser(description = 'use this program to classify flowers XD.')
parser.add_argument('-c','--model_dir', action='store',type=str, help='model to be loaded and create classification')
parser.add_argument('-i','--image_path',action='store',type=str, help='file path of image e.g. flowers/test/class/image')
parser.add_argument('-k', '--topk', action='store',type=int, help='number of top ranked classification by probability in descending order.')
parser.add_argument('-j', '--json', action='store',type=str, help='class name assignment json file.')
parser.add_argument('-g','--gpu', action='store_true', help='turn on GPU mode if available')

results = parser.parse_args()


model_dir = results.save_directory
image_path = results.image_path
top_k = results.topk
gpu_mode = results.gpu
class_names = results.json

with open(class_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
model = tf.keras.models.load_model(model_dir)
device = ("cuda" if results.gpu else "cpu")
model.to(device)
processed_image = process_image(image)

probs, classes = predict(processed_image, loaded_model, top_k, gpu_mode)


if results.class_names != None:  
    top_class_names = []
    for idx in classes:
      idx_str = str(idx)
      class_name = class_names[idx_str]
      top_class_names += class_name
    print('Top {} probabilities are (from low to high): {}'.format(top_k,probs))
    print('And they have according class: {}'.format(top_class_names))
    print("This flower is most likely to be a:{} with a probability of {}".format(top_class_names[-1],round(probs[-1]*100,4))
        
else:         
    print('Top {} probabilities are (from low to high): {}'.format(top_k,probs))
    print('And they have according class index: {}'.format(classes))
