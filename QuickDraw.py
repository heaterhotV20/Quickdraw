%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from glob import glob
import ntpath

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import tensorflow as tf

import numpy as np
import pickle

!wget -qq https://www.dropbox.com/s/gdlb8dnjzcly51o/quickdraw.zip
  
!unzip -qq quickdraw.zip

!rm -r __MACOSX
!rm quickdraw.zip


arr = np.load('./quickdraw/bee.npy')
arr.shape

file_names = glob('./quickdraw/*.npy')

# make some class names
class_names = []

for file in file_names:
  name = ntpath.basename(file)
  class_names.append(name[:-4])
  
  
print(class_names)

x = []
x_load = []
y = []
y_load = []

def load_data():
    count = 0
    for file in file_names:
        x = np.load(file)
        x = x.astype('float32') / 255.
        x = x[0:10000, :]
        x_load.append(x)
        y = [count for _ in range(10000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load
  
features, labels = load_data()
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')



features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])


with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)
	
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense,Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

def keras_model(image_x, image_y):
    num_of_classes = 20
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(Conv2D(64, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))#0.6
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "QuickDraw.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels

features, labels = loadFromPickle()
#features, labels = augmentData(features, labels)

features, labels = shuffle(features, labels)
labels=prepress_labels(labels)
train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
model, callbacks_list = keras_model(28,28)
print_summary(model)

model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=30, batch_size=64, callbacks=[TensorBoard(log_dir="QuickDraw")])

model.save('QuickDraw.h5')

from google.colab import auth
auth.authenticate_user()

MODEL_OUTPUT = 'gs://test-cats-dogs/QuickDraw.h5'  

import os
from tensorflow.python.lib.io import file_io

def copy_file_to_gcs(job_dir, file_path):
  with file_io.FileIO(file_path, mode='rb') as input_f:
    with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
      output_f.write(input_f.read())
      
copy_file_to_gcs(MODEL_OUTPUT, 'QuickDraw.h5')

# Predicting portion
import cv2
import matplotlib.pyplot as plt

def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class
  
def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

from keras.models import load_model
model1 = load_model('gs://test-cats-dogs/QuickDraw.h5/QuickDraw.h5')

bucket_path = "gs://test-cats-dogs"
bucket_object = bucket_path + "/angel5.bmp"

img = file_io.FileIO(bucket_object, mode='rb')

import matplotlib.pyplot as plt
from matplotlib.image import imread

im = imread(img)
#im = cv2.bitwise_not(im)

imgplot = plt.imshow(im)

plt.show()

pred_probab, pred_class = keras_predict(model1, im)
print(pred_class, pred_probab) 

class_names[pred_class]

	