# Documentation for CNN project on embedded system

##CNN :

A Convolutional Neural Network is a derived form of Artificial Neural Networks
that is particularly fitted for patterns recognition. Such neural networks are very
intensive in ressource use, which makes it difficult for them to be used on small, ressource
limited embedded systems.
The aim of the project is to implement a lightweight CNN using C++, optimizing the heavy
computations and use a light model suchas SqueezeNet in order to proceed to face reognition on
embedded systems.
It is to be noted that the training phase is to be performed on GPU powered laptop as it would take
hours or days to train it on a limited CPU, or even on the most powerful CPUs actually developed.

What is the theory behind the CNN ?
-------------------------------------------------------------------------
The CNN architecture is essentially composed of convolution layers followed by
pooling layer. Further description of those terms are to be discussed later on.
The CNN takes as an input chunks of pixels and output a probability number comprised
between 0 and 1 for every registered labels.
The convolution plus pooling layers are usually followed by fully connected (FC)
layers.
If a classification is necessary, as in our project, where we need to classify either faces
or objects, a final softmax layer is necessary.

Activation function
----------------------

An activation function allows non linearity in a ANN. It´s features are composed of :
* non linearity
* differentiable everywhere
* Extended
* Monotone
* Identity in x=0

Commonly used activation functions in ANNs are :
* Identify functions
* Heaviside
* Sigmoid

Convolution
----------------------
Convolution is a mathematical operation to merge two sets of informations. It is very used in
the field of electrical engineering, where the convolution can output similarities between two signals.
In the CNN, a convolution is applied on ourinput data, using a convolution filter, in order to produce a feature map. The filter is equally named kernel. It is usually a matrix or a tensor of smaller size than the original input.
If the kernel is of size 3x3, the convolution is called 3x3 convolution.
The convolution is obtained by sliding the filter over the input. At every location, we do element wise
matrix multiplication and sum the result, which goes on the feature map.
The area that is feature by the kernel is called *receptive field*.
The kernel is then slided to the right by a number of rows named the *stride*.

Convolutions are most commonly performed in 3D on colored images rather than 2D (luminance images). The kernel should accordingly be of 3x3x3 size in order to perform a 3X3X3 convolution.

The convolutions layer are often of different convolution size and are then stacked to form a full feature map.

Ex: Having a 32x32x3 and then using a filter of size 5x5x3, the difference with the previous description of a 2D conv is that we will sum the result of the 3 depth-wise convolutions, resulting in a feature map of 32x32x1.
If 5 conv layers would have been used, regardless of the kernel dimension, we would end up with a 32x32x5 feature map.


Non-linearity layers
--------------------------

A neural network, composed of multiple layers with many neurons, can be represented as a unique neuron
if we had a linearity at each layer. In order to avoid this, which would make our network useless at performing decisionss, it is necessary to add non linearity. As soon as the convolutional layers are done, we need to pass the result of the convolution operation through relu activation function.

Stride and padding
--------------------------

The stride, as mentionned earlier, specify by how much we move the filter at each step. It defines
the size of the resulting feature map.
The padding, as opposed to the stride, allows us to keep the input size, by adding paddings to surround the input with zeros.
Padding are always used in CNNs.

Pooling layers
----------------------------

Pooling allows to reduce the size of feature maps, which enable us to reduce the number of parameters,
thus shortening training time and response output.
The pooling more commonly used is the max pooling which just takes the maximum value in the pooling window which is of usually of small dimension, with a defined stride.
Pooling keep the depth of the feature map intact.
If we have a feature map of dimension 32x32x10 and a pooling with a 2x2 windows using a stride of 2, we will end up with a 16x16x10 feature map.

In most CNN architectures, pooling is typically performed with a 2x2 window and a stride of 2 while the convolution is performed using a 3x3 window, stride of 1 and using padding.

Hyper-parameters of the CNNs
----------------------------------

Four important parameters are decided on CNNs :
* Filter size, as most of the time, 3x3 kernels are used but some other dimensions can also be used depending on the application.
* Filter count, which is a power of two comprised between 32 and 1024. More filters output a more accurate result, but risk overfitting and obviously use more computation ressources.
A small number of filter are usually used in the initial layers and increased as we go deeper in the network.
* The stride is usually kept at a value of 1 but in order to optimie our network, this parameter is to be discussed.
* Padding: the parameters is also to be discussed as it is undoubtedly being used.

Fully connected layers
-------------------------

In CNNs, fully connected layers are always used after the convolutions layers to wrap up the CNN archtecture.
FC layers expect a 1D vectore of numbers, so we have to flatten the output of the ultimate pooling layer which is basically just a rearrangement of 3D vectors in a 1D.

Training
-------------------------

Training is realized using backpropagation, which is to be discussed as we dig further in the functionning of the ANN and it´s perspective of evolutions.
Training in CNNs are more computational heavy as the convolution is pretty ressource eating.


Global vision of the CNNs
------------------------------------------------------------------------------------

The CNN can be seen as a suite of a feature extraction followed by a classification part. Convolution and pooling perform feature extraction.
The classification is performed by the FC layers.

How to implement a CNN ?
-------------------------------------------------------------------------------------

Dropout
-----------------

Dropout allows a subsequent gain of up to 2% in precision. It is used to limit overfitting. The idea behind is that at run time, at each iteration, a neuron is dropped with a probability p, which çeans that all inputs and outputs of this neuron are turned off for this iteration.
At each iteration, the dropped neuron can be reactivated.
p is called the dropout rate and is usually around 0.5, which means that any neuron has 50% of chance to being dropped out.
The dropout is working because it is preventing the network to be dependent of a number of neurons and force independance in each neuron.

Dropout is only applied during training time.

An alternative to dropout is batch normalization which is to be analyzed.

Overfitting
----------------------------

I previously mentionned overfitting, but without going deeper into explanations, which will be rectified in the following :

"With four parameters I can fit an elephant, and with five I can make him wiggle his trunk." John von Neumann

This quote is pretty descriptive of overfitting.
It basically define a model that define the training data way too well and output perfect result for the training data. It is fitted for it and the parameters are modeled after it. However, for a set of new real world data, the accuracy of the model drop out subsequently.
Overfitting happens when the accuracy over the training batch start to get stuck.

Avoiding overfitting
---------------------------------

Data augmentation
---------------------------------

Data augmentation enriches the training data by generating new examples via random transformation of existing ones.
It is done dynamically at training time.
Common transformations are rotation, shifting, resizing, adjusting exposure, etc...

Stacking convolutions
----------------------------------

In state-of-art to this day, stacking convolution layer is very used in order to limit computational use of ressources, as two 3x3 convolutions would produce the same output as a 5x5 convolution.
Models such as VGG16 or AlexNet take advantage of stacking convolutions layers, which come along with the advantage of using two ReLu layers resulting in an increased non linearity, giving more power to the model.
