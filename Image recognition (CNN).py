# Libraries used
import tensorflow as tf
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import skimage

# Preparing the image data set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'C:/Users/bhati/anaconda3/_Programs/CNN - image recognition (dogs & cats)/data/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(
    'C:/Users/bhati/anaconda3/_Programs/CNN - image recognition (dogs & cats)/data/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Building the CNN layers
cnn = tf.keras.models.Sequential()

# shape 64x64 and 3=color (1=b&w)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))

# Pool method Max, stride = 2, size 2x2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the loss function
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training CNN on the training set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Optimized post training: Highest accuracy at epoch = 25, training acc=0.9668 & test acc=0.8415

link = "https://i.pinimg.com/originals/df/85/81/df8581994ad2a1f50c7cf193144cea42.png"
image_is_really = "cat"  # updated after loading from link
load_img = skimage.io.imread(link)

# Commented code block to load a downloaded image
# test_image = image.load_img('path.jpg', target_size=(64, 64))
# convert image to array
# test_img = image.img_to_array(test_img)
# test_img = np.expand(test_img, axis = 0)

img_resize = skimage.transform.resize(load_img, (64, 64), anti_aliasing=True)
test_img = tf.expand_dims(img_resize, 0)

# Prediction
result = cnn.predict(test_img)

if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

if prediction == image_is_really:
    print(f"Correctly identified as {prediction}.")
else:
    print(f"Incorrectly identified as {prediction}, it is a {image_is_really}.")
