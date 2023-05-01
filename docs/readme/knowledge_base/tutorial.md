# Tutorial: Verifiable NN inference for MNIST

In this tutorial we will build a Neural Network using the ONNX Cairo 1.0 Runtime for the MNIST dataset. We will follow these steps:

1. Build the Neural Network and train it using your prefered framework. In this case we will use Tensorflow.
2. Save the weights to import them in our Cairo 1.0 Neural Network.
3. Build the NN in Cairo 1.0.
4. Load the weights in our Cairo 1.0 NN.
5. Load a sample image to perform inference.
6. Perform verfiable inference with our NN in Cairo 1.0.

## Setup

Follow the steps defined in the section [Getting Started](../onnx\_cairo\_runtime/getting\_started.md) for Cairo 1.0 setup instructions.

## Building our Neural Network in Tensorflow

The NN has a simple two-layer architecture:

* Input layer 𝑎\[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image.
* A hidden layer 𝑎\[1] will have 10 units with ReLU activation.
* Output layer 𝑎\[2] will have 10 units corresponding to the ten digit classes with softmax activation.

1. Clone the repository [Neural Network Cairo](https://github.com/franalgaba/neural-network-cairo)
