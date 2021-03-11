import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def showfig():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

print(train_images.shape)
print(test_images.shape)

#model
model = models.Sequential()
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
model.add(Conv2D(64,kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax',kernel_initializer='he_normal'))


model.summary()
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)



