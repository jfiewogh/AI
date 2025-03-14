import gzip
import numpy
from matplotlib import pyplot
import random
import time

image_size = 28
image_count = 3

random.seed(time.time())

with gzip.open("samples/train-images-idx3-ubyte.gz", "r") as f:
    f.read(16)
    images_data = f.read(image_size * image_size * image_count)
    images = numpy.frombuffer(images_data, dtype=numpy.uint8).reshape(image_count, image_size, image_size)

with gzip.open("samples/train-labels-idx1-ubyte.gz", "r") as f:
    f.read(8)
    labels_data = f.read(image_size * image_size * image_count)
    labels = numpy.frombuffer(labels_data, dtype=numpy.uint8)

print(labels)

def show_image(index):
    image = numpy.asarray(images[index]).squeeze() # 2D array of size 28
    # range from 0 (empty) to 255 (filled)

    print(labels[index])

    pyplot.imshow(image)
    pyplot.show()

a = random.randint(0, image_count - 1)
print(a)
show_image(a)