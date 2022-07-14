import data
import matplotlib.pyplot as plt
num_of_images = 60

for imgs, targets in data.data_train_loader:
    break

for index in range(num_of_images):
    plt.subplot(6, 10, index+1)
    plt.axis('off')
    img = imgs[index, ...]
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
plt.show()
