import os
import sys
import json
import numpy as np
import pandas as pd
import time
import skimage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
print("gpu available: ", len(tf.config.experimental.list_physical_devices('GPU')))

rdir = './Semantic-Segmentation'


sys.path.append(rdir) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib


mdir = os.path.join(rdir, "logs")


modelpath = os.path.join(rdir, " ")


if not os.path.exists(modelpath):
    utils.download_trained_weights(modelpath)

class conf(Config):

    gpu_count = 1
    per_gpu = 1
    classes = 1 + 1  
    min_dim = 512
    max_min = 512
    steps_epoch = 500
    val_steps = 5
    backbone = 'resnet101'
    val = (8, 16, 32, 64, 128)
    roi = 32
    instances = 50 
    post_inference = 500 
    post_trainning = 1000 
    
config = conf()
config.display()

data = []
for file in sorted(os.listdir(r'path_to_directory\train_data')):
    data.append(file)
df_train_data = pd.DataFrame(data, columns=['imageid'])
df_train_data['imageid'] = df_train_data['imageid'].str.replace('.jpg','')
image_id = df_train_data['imageid'].tolist()
for file in sorted(os.listdir(r'path_to_directory\validation_mask')):
    data.append(file)
df_validation_mask = pd.DataFrame(data, columns=['imageid'])
df_validation_mask['imageid'] = df_validation_mask['imageid'].str.replace('.png','')
val_mask = df_validation_mask['imageid'].tolist()
for file in sorted(os.listdir(r'path_to_directory\mask')):
    data.append(file)

def resize(path, image_list, form):
    for image in image_list:
        img = Image.open(''.format(path, image, form))
        img = img.resize((512,512), Image.ANTIALIAS)
        img.save(''.format(path, image, form))
        
        
resize('train_data', image_id, 'jpg')
resize('validation_data', val_img, 'jpg')
resize('train_mask', train_img, 'png')
resize('validation_mask', val_mask, 'png')

class dataset(utils.Dataset):

    def load_data(self, image_ids, form, image_group):
        self.add_class('', 1, '')
        
        for image in image_ids:
            self.add_image('', image_id=image, path=(''.format(image_group, image, form)), labels=1, height=512, width=512)
            
            
    def load_mask(self, image_id):

        info = self.image_info[image_id]
        image_id = info['id']
        data = []
        for file in sorted(os.listdir(r'path_to_directory\mask')):
            data.append(file)
            
        matching = [s for s in data if image_id in s]
        
        result = [] 
        
        for match in matching:
            im = Image.open(''.format(match))
            im = np.asarray(im)
            im = im.reshape((im.shape[0], im.shape[1], 1))
            result.append(im)
            
        
        im = np.dstack(result)
        
        
        class_ids = np.array([1 for _ in range(im.shape[-1])])
        return im, class_ids

dataset_train = dataset()
dataset_train.load_data(image_id, 'jpg', 'train')
dataset_train.prepare()

dataset_val = dataset()
dataset_val.load_data(val_img, 'jpg', 'validation')
dataset_val.prepare()

dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


model = modellib.MaskRCNN(mode="training", config=config, model_dir=mdir)

init_with = "coco"

model.load_weights(modelpath, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

start_train = time.time()
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE, epochs=4, layers='heads')

end_train = time.time()

minutes = round((end_train - start_train) / 60, 2)

history = model.keras_model.history.history

start_train = time.time()
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=8, layers="all")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)


history_fine_tune = model.keras_model.history.history

class infer_con(conf):
    gpu_count = 1
    per_gpu = 1
    min_dim = 512
    max_min = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config = infer_con()

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=mdir)

model_path = model.find_last()
model.load_weights(model_path, by_name=True)

real_test_dir = ''
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))
for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]


