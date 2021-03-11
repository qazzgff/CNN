from tensorflow.keras import datasets
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import numpy as np
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

img = train_images[0]
PEAK = 1
noise = np.random.poisson(img)
img_noisy = img + noise
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img_noisy)

from sklearn.preprocessing import RobustScaler

noisy = img_noisy.T
print(noisy.shape)
after = np.zeros(shape=(3, 32, 32))
for i in range(0, 3):
    temp = noisy[i]
    robust_scaler = RobustScaler()
    robust = robust_scaler.fit_transform(temp)
    after[i] = temp

after = after.T
plt.subplot(1,3,3)
plt.imshow(after)
plt.show()

