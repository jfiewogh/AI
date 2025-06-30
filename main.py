import gzip
import numpy
import random
import time
import numpy as np
import math


image_size = 28
image_count = 10000


neurons_per_layer = 20
inner_layers_count = 8


LEARNING_RATE = 0.01




random.seed(time.time())

with gzip.open("samples/train-images-idx3-ubyte.gz", "r") as f:
    f.read(16)
    images_data = f.read(image_size * image_size * image_count)
    images = numpy.frombuffer(images_data, dtype=numpy.uint8).reshape(image_count, image_size, image_size)

with gzip.open("samples/train-labels-idx1-ubyte.gz", "r") as f:
    f.read(8)
    labels_data = f.read(image_size * image_size * image_count)
    labels = numpy.frombuffer(labels_data, dtype=numpy.uint8)


neurons = []

class Neuron:
    def __init__(self):
        self.id = len(neurons)
        self.value = 0
        self.desired_value = 0
        self.bias = 0
        neurons.append(self)



weights = {}

def get_weight_key(first_neuron, second_neuron):
    return f"{first_neuron.id}-{second_neuron.id}"



def get_initial_weight():
    n_inputs = len(inputs)
    return np.random.normal(0, np.sqrt(2 / n_inputs))

def get_weight(first_neuron, second_neuron):
    weight_key = get_weight_key(first_neuron, second_neuron)
    if weight_key not in weights:
        weights[weight_key] = get_initial_weight()
    return weights[weight_key]


def change_weight(weight_key, weight_change):
    weights[weight_key] -= LEARNING_RATE * weight_change

def change_bias(neuron, bias_change):
    neuron.bias -= LEARNING_RATE * bias_change

def change_input_neuron_desired_value(neuron, value_change):
    neuron.desired_value = neuron.value - LEARNING_RATE * value_change


# Neurons
inputs = [Neuron() for _ in range(image_size * image_size)]
outputs = [Neuron() for _ in range(10)]
inner_layers = [[Neuron() for _ in range(neurons_per_layer)] for _ in range(inner_layers_count)]

layers = [inputs] + inner_layers + [outputs]



# get image

def get_image_and_label(index):
    image = numpy.asarray(images[index]).squeeze() # 2D array of size 28 # 0 - 255
    label = labels[index]
    return image, label

def add_image_to_inputs(image):
    for x in range(image_size):
        for y in range(image_size):
            index = x * image_size + y 
            inputs[index].value = float(image[x][y]) / 255

## matrix

def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return max(0, x)

def ReLU_derivative(x):
    return 1 if x >= 0 else 0



def get_result(neuron, input_neurons):
    sum = neuron.bias
    for input_neuron in input_neurons:
        weight = get_weight(input_neuron, neuron)
        sum += input_neuron.value * weight
    return sigmoid(sum)

def get_output():
    for i in range(1, len(layers)):
        input_layer = layers[i - 1]
        for neuron in layers[i]:
            neuron.value = get_result(neuron, input_layer)


num_correct = 0
total = 0

while True:
    image, label = get_image_and_label(random.randint(0, image_count - 1))
    add_image_to_inputs(image)

    get_output()

    # print(weights)



    # if total == 1:
    #     break


    output_number = outputs.index(max(outputs, key=lambda x: x.value))

    if output_number == label:
        num_correct += 1
    total += 1



    print(f"Guess: {output_number} ({max(outputs, key=lambda x: x.value).value}), Actual: {label}, {num_correct}/{total} ({round(num_correct / total * 100, 2)}%)")




    # backpropagation


    # y is either 0 (incorrect) or 1 (correct)

    for i, output_neuron in enumerate(outputs):
        output_neuron.desired_value = 1 if i == label else 0

    # cost = 0
    # for i, output_neuron in enumerate(output_layer):
    #     desired_output = 1 if i == label else 0
    #     cost += (output_neuron.value - desired_output) ** 2
    # print(cost)

    weight_changes = {}
    bias_changes = {}

    # first layer


    for i in range(len(layers) - 1, 1, -1):
        output_layer = layers[i]
        input_layer = layers[i - 1]

        input_neuron_value_changes = {}

        for output_neuron in output_layer:
            desired_output_value = output_neuron.desired_value
            
            z = output_neuron.bias
            for input_neuron in input_layer:
                z += get_weight(input_neuron, output_neuron) * input_neuron.value  
            m = ReLU_derivative(z)

            for input_neuron in input_layer:
                h = (m) * (2 * (output_neuron.value - desired_output_value))

                # Weight change
                weight_sensitivity = (input_neuron.value) * h
                weight_key = get_weight_key(input_neuron, output_neuron)
                if weight_key not in weight_changes:
                    weight_changes[weight_key] = []
                weight_changes[weight_key].append(weight_sensitivity)

                # Bias change
                bias_sensitivity = h
                if input_neuron not in bias_changes:
                    bias_changes[input_neuron] = []
                bias_changes[input_neuron].append(bias_sensitivity)

                # Input neuron value change
                input_neuron_value_change = get_weight(input_neuron, output_neuron) * h
                if input_neuron not in input_neuron_value_changes:
                    input_neuron_value_changes[input_neuron] = []
                input_neuron_value_changes[input_neuron].append(input_neuron_value_change)
 
        # Update desired neuron values
        for neuron, changes in input_neuron_value_changes.items():
            average_value_change = sum(changes)
            change_input_neuron_desired_value(neuron, average_value_change)
            neuron.value = neuron.desired_value
    
    ### Update
    # weights
    for weight_key, changes in weight_changes.items():
        average_weight_change = sum(changes) / len(changes)
        change_weight(weight_key, average_weight_change)
    # biases
    for neuron, changes in bias_changes.items():
        average_bias_change = sum(changes) / len(changes)
        change_bias(neuron, average_bias_change)


    # print layers
    for i, layer in enumerate(layers[1:]):
        print(i + 1, [round(x.value, 3) for x in layer])
    


    # average for all outputs

    time.sleep(3)