1. The Perceptron Model
The perceptron is a linear classifier inspired by biological neurons that computes a weighted sum of inputs and outputs a binary value based on a threshold
. It can model simple Boolean gates like AND and OR but cannot solve non-linearly separable problems like XOR
.
Formula: y=1 if ∑w 
i
​
 x 
i
​
 ≥T, else 0
.
Example: A perceptron recognizing a printed letter "A" via a 20×20 grid of photocells
.
2. Activation Functions
Activation functions introduce non-linearity, enabling networks to model complex data; without them, a multi-layer network behaves like a simple linear model
. Modern choices like ReLU prevent gradients from "vanishing" during training compared to traditional Sigmoid or Tanh functions
.
ReLU Formula: f(x)=max(0,x)
.
Example: If a neuron receives a value of -5, ReLU outputs 0; if it receives +3, it outputs 3
.
3. Multi-Layer Perceptron (MLP)
An MLP is a feedforward network consisting of an input layer, one or more "fully connected" hidden layers, and an output layer
. Hidden layers allow the network to learn a hierarchical decomposition of features to solve complex, non-linear problems
.
Formula (Layer Output): a 
1
​
 =σ(W 
1
​
 x+b 
1
​
 )
.
Example: Predicting the category (e.g., "T-shirt," "Trouser") of an image from the Fashion-MNIST dataset
.
4. Universal Approximation Theorem
This theorem states that a feedforward network with at least one hidden layer and a finite number of neurons can approximate any continuous function to arbitrary accuracy
. It guarantees that neurons act as flexible building blocks to mimic any complex curve
.
Formula: g(x)=∑ 
j=1
m
​
 a 
j
​
 σ(w 
j
T
​
 x+b 
j
​
 )
.
Example: Replicating a highly jagged mathematical function by increasing the number of hidden neurons m
.
5. Deep vs. Wide Networks
Wide networks rely on many neurons in few layers and tend to memorize training examples, while deep networks learn hierarchical features (edges → parts → objects)
. Deep networks are more parameter-efficient, representing complex functions with fewer weights than very wide shallow ones
.
Complexity Formula: Number of "pieces" modeled ≈m 
L
  (where L is depth and m is width)
.
Example: Lower layers in a deep net learn color blobs, while higher layers learn "semantic concepts" like a dog's ear
.
6. Linear Softmax Layer
Softmax is used in the final layer of a classification network to squash raw output scores into a probability distribution that sums to 1.0
. It allows the model to express confidence levels for each possible class
.
Formula: f 
c
​
 = 
∑e 
g 
i
​
 
 
e 
g 
c
​
 
 
​
 
.
Example: A 3-class model outputting [0.8,0.1,0.1] indicating an 80% probability for the first class
.
7. Loss Functions (MSE and Cross-Entropy)
Loss functions measure the error between a model's prediction and the true target
. Mean Squared Error (MSE) is typically used for regression, while Cross-Entropy is standard for classification as it penalizes wrong confident predictions
.
MSE Formula: J= 
N
1
​
 ∑(y 
i
​
 −y 
i
∗
​
 ) 
2
 
.
Cross-Entropy Formula: L=−∑y 
i
​
 logf 
i
​
 
.
Example: A classification model has a high Cross-Entropy loss if it predicts "Cat" with 10% confidence when the actual label is "Cat"
.
8. Stochastic Gradient Descent (SGD)
SGD is an optimization algorithm that trains the network by taking small steps in the direction opposite to the gradient of the loss function
. This iterative process adjusts weights to minimize total error over the training data
.
Formula: θ 
(t+1)
 =θ 
(t)
 −η∇ℓ(f 
θ
​
 (x 
i
​
 ),y 
i
​
 )
.
Example: Adjusting a weight from 0.5 to 0.48 after an error to "slide down" the slope of the loss curve
.
9. Computation Graphs (Static vs. Dynamic)
A computation graph is an unrolled representation of a network's operations as a sequence of nodes (tensors and operations)
. Static graphs are built once before execution (strong optimization), while dynamic graphs are built during execution (easier debugging)
.
Nodes Example: Value nodes (x,W,b) and operation nodes (matmul, add, ReLU)
.
Directed Edges: Show dependencies, such as which values flow into a "softmax" operation
.
10. Automatic Differentiation (Reverse Mode)
Reverse-mode AD (Backpropagation) is the standard for neural networks because it computes gradients for millions of parameters in a single backward pass
. It is significantly more efficient than forward-mode AD for functions with a single scalar output (like Loss)
.
Logic: It asks, "How does the loss depend on each intermediate variable?" starting from the end of the graph
.
Example: Calculating how a weight in the first layer affects the final error using a single pass from the output back to the input
.
11. Backpropagation and the Chain Rule
Backpropagation uses the calculus Chain Rule to propagate the error signal backward and determine how much each weight contributed to the total loss
. Connection weights are adjusted proportionally to this contribution
.
Chain Rule Formula:  
dx 
k
​
 
dy 
i
​
 
​
 =∑ 
j=1
J
​
  
du 
j
​
 
dy 
i
​
 
​
 ⋅ 
dx 
k
​
 
du 
j
​
 
​
 
.
Example: If L=v 
2
  and v=u+c, then  
∂u
∂L
​
 = 
∂v
∂L
​
 ⋅ 
∂u
∂v
​
 =2v⋅1
.
12. Vanishing and Exploding Gradients
Vanishing gradients occur when small derivatives (like Sigmoid's max of 0.25) are multiplied repeatedly, making gradients in early layers near-zero
. Exploding gradients happen when large weights/derivatives cause gradients to grow exponentially, leading to unstable training
.
Vanishing Example: 0.1×0.1×0.1=0.001 after only three layers
.
Exploding Example: 3.0×3.0×3.0=27.0 after three layers
.
13. Invariance and Equivariance
Invariance means the output remains the same even if the input is transformed (e.g., a classifier saying "Cat" regardless of position)
. Equivariance means the output transforms in the same way as the input (e.g., a segmentation mask moving with the object)
.
Invariance Formula: f[x]=f[t(x)]
.
Equivariance Formula: f[t(x)]=t(f[x])
.
14. Convolution Operation
The core of CNNs involves sliding a learnable kernel (filter) across an image to perform element-wise multiplication and summation
. This process builds "feature maps" that highlight local patterns like edges and textures
.
Formula: (I∗K)(i,j)=∑ 
m
​
 ∑ 
n
​
 I(i+m,j+n)⋅K(m,n)
.
Example: A 3x3 kernel moving across a 10x10 image to detect vertical lines
.
15. Convolution Parameters (Stride, Padding, Dilation)
Padding adds zeros to the image border to maintain spatial size
. Stride determines the kernel's shift size (e.g., stride 2 downsamples by half), and Dilation spreads the kernel weights to capture a larger area without adding parameters
.
Output Size Formula: O= 
S
I−K+2P
​
 +1
.
Example: A 227x227 input with an 11x11 kernel, stride 4, and no padding results in a 55x55 output
.
16. Receptive Fields
The receptive field is the specific region of the input image that a particular neuron "sees"
. As you go deeper into a CNN, the receptive field size increases, allowing neurons to detect more complex, global patterns
.
Logic: Multiple stacked small kernels (e.g., two 3x3) emulate the receptive field of one large kernel (e.g., 5x5) more efficiently
.
17. Separable Convolutions (Spatial and Depthwise)
Spatially separable convolutions split a 2D kernel into two 1D kernels (e.g., 3x3 into 3x1 and 1x3) to reduce multiplications
. Depthwise separable convolutions split the operation into a depthwise step (per channel) and a pointwise step (1x1 conv) for extreme efficiency
.
Efficiency Example: A standard 3x3 convolution on a 10x10 image takes 576 multiplications, while the spatially separable version takes only 432
.
18. Pooling (Downsampling) Layers
Pooling layers reduce the resolution of feature maps to decrease computation and memory while providing local invariance to small shifts
. Max pooling is the most common, taking the highest value in a window, while Mean pooling takes the average
.
Max Pooling Formula: P 
out
​
 =max 
m,n∈W
​
 F 
in
​
 (m,n)
.
Example: In a 2x2 window of [3,2;0,7], Max pooling outputs 7
.
19. Transposed Convolutions and Upsampling
Transposed convolution (sometimes called "deconvolution") maps a single pixel to a larger area (e.g., 1x1 to 3x3) to increase image resolution
. Other upsampling methods include simple duplication or bilinear interpolation
.
Output Height Formula: H 
out
​
 =(H 
in
​
 −1)×s−2p+k
.
Example: Mapping a 1x1 input to a 3x3 area using a stride of 3 and kernel of 3
.
20. Batch Normalization (BN)
BN standardizes layer activations across a mini-batch to have zero mean and unit variance, which stabilizes and accelerates training
. It introduces learnable scale (γ) and shift (β) parameters to allow the network to undo the normalization if necessary
.
Normalization Formula:  
x
^
 = 
σ 
2
 +ϵ

​
 
x−μ
​
 
.
Regularization Effect: Calculation on random mini-batches adds noise, which helps prevent overfitting
.
21. Layer Normalization
Layer Normalization standardizes activations across the feature dimension for a single sample rather than across a batch
. This makes it independent of batch size and ideal for sequence data like RNNs or Transformers where batch statistics are inconsistent
.
Logic: It calculates mean and variance from all activations within a single layer for one data point
.
Example: Normalizing a flattened feature vector 
 results in approximately [−1.34,−0.45,0.45,1.34]
.
22. Regularization (Dropout and L2)
Regularization techniques prevent the model from memorizing training data (overfitting)
. Dropout randomly "turns off" neurons during training with probability p, while L2 regularization (weight decay) penalizes large weight values in the loss function
.
Dropout Formula: h 
′
 =h⊙m (where m is a random binary mask)
.
L2 Formula: L 
total
​
 =L 
error
​
 + 
2
λ
​
 W 
2
 
.
23. Classic CNN Architectures
AlexNet (2012): Used 5 conv layers, ReLU, Dropout, and GPUs to prove the power of deep CNNs
.
VGG-Net: Emphasized depth (16-19 layers) using uniform 3x3 kernels to increase non-linearity
.
Inception (GoogLeNet): Introduced parallel branches and 1x1 "bottleneck" convolutions for parameter efficiency
.
ResNet: Introduced "skip connections" to learn residual mappings (H(x)=F(x)+x), enabling training of over 150 layers without performance degradation
.
DenseNet: Utilizes dense blocks where each layer receives feature maps from all preceding layers
.