import gzip
import numpy
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

a = 10
b = 2

matrix = [[0] * 28 * 28] +  [[0] * a for _ in range(b)][0] + [[0] * 10]

print(len(matrix))

def get_image_and_label(index):
    image = numpy.asarray(images[index]).squeeze() # 2D array of size 28 # 0 - 255
    label = labels[index]
    return image, label

image, label = get_image_and_label(0)

def add_image_to_matrix(image):
    for x in range(image_size):
        for y in range(image_size):
            matrix[x * image_size + y] = float(image[x][y]) / 255

add_image_to_matrix(image)

print(matrix[0])