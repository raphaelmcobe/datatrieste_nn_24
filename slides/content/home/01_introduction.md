
## Neural Networks

* Neurons as structural constituents of the brain <a href="http://hobertlab.org/wp-content/uploads/2014/10/Andres-Barquin_Cajal_2001.pdf" target="_blank">[Ramón y Cajál, 1911]</a>; 
* Five to six orders of magnitude <em>slower than silicon logic gates</em>; 
* In a silicon chip happen in the <em>nanosecond (on chip)</em> vs <em>millisecond range (neural events)</em>; 
* A truly staggering number of neurons (nerve cells) with <em>massive interconnections between them</em>;

---
## Neural Networks 

* Receive input from other units and decides whether or not to fire;
* Approximately <em>10 billion neurons</em> in the human cortex, and <em>60 trillion synapses</em> or connections <a href="https://www.researchgate.net/publication/37597256_Biophysics_of_Computation_Neurons_Synapses_and_Membranes" target="_blank">[Shepherd and Koch, 1990]</a>;
* Energy efficiency of the brain is approximately $10^{−16}$ joules per operation per second against ~ $10^{−8}$ in a computer;

---
{{<slide background-image="neuron2.png">}}
## Neurons

---
## Neurons

* input signals from its <em>dendrites</em>;
* output signals along its (single) <em>axon</em>;

<img src="neuron1.png"/>

{{% note %}}
* three major types of neurons: <em>sensory neurons</em>, <em>motor neurons</em>, and <em>interneurons</em>
{{% /note %}}

---
## Neurons
### How do they work?

<ul>
<div align="left">
{{% fragment %}} <li>Control the influence from one neuron on another:</li> {{% /fragment %}}
</div>

<ul>
<div align="left">
{{% fragment %}} <li><em>Excitatory</em> when weight is positive; or</li> {{% /fragment %}}
</div>

<div align="left">
{{% fragment %}} <li><em>Inhibitory</em> when weight is negative;</li> {{% /fragment %}}
</div>
</ul>

<div align="left">
{{% fragment %}} <li>Nucleus is responsible for summing the incoming signals;</li> {{% /fragment %}}
</div>

<div align="left">
{{% fragment %}} <li><strong>If the sum is above some threshold, then <em>fire!</em></strong></li> {{% /fragment %}}
</div>
</ul>

---
## Neurons
### Artificial Neuron

<center><img src="artificial_neuron.jpeg" width="800px"/></center>

---
{{<slide background-image="neurons.png">}}
## Neural Networks

---
## Neural Networks
* It appears that one reason why the human brain is <em>so powerful</em> is the
sheer complexity of connections between neurons;
* The brain exhibits <em>huge degree of parallelism</em>;

---
## Artificial Neural Networks
* Model each part of the neuron and interactions;
* <em>Interact multiplicatively</em> (e.g. $w_0x_0$) with the dendrites of the other neuron based on the synaptic strength at that synapse (e.g. $w_0$ ); 
* Learn <em>synapses strengths</em>;

---
## Artificial Neural Networks
### Function Approximation Machines
* Datasets as composite functions: $y=f^{*}(x)$
  * Maps $x$ input to a category (or a value) $y$;
* Learn synapses weights and aproximate $y$ with $\hat{y}$:
  * $\hat{y} = f(x;w)$
  * Learn the $w$ parameters; 

---
## Artificial Neural Networks
* Can be seen as a directed graph with units (or neurons) situated at the vertices;
  * Some are <em>input units</em>
* Receive signal from the outside world;
* The remaining are named <em>computation units</em>;
* Each unit <em>produces an output</em>
  * Transmitted to other units along the arcs of the directed graph;

---
## Artificial Neural Networks
* <em>Input</em>, <em>Output</em>, and <em>Hidden</em> layers;
* Hidden as in "not defined by the output";
<center><img src="nn1.png" height="200px" style="margin-top:50px;"/></center>

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* Imagine that you want to forecast the price of houses at your neighborhood;
  * After some research you found that 3 people sold houses for the following values:

<br />

Area (sq ft) (x)|	Price (y) in USD
----------------|----------
2,104	          |  $399,900$
1,600	          |  $329,900$
2,400	          |  $369,000$

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 

{{% fragment %}} If you want to sell a 2K sq ft house, how much should ask for it? {{% /fragment %}}
<br /><br />
{{% fragment %}} How about finding the <em>average price per square feet</em>?{{% /fragment %}}
<br /><br />
{{% fragment %}} <em>$180 per sq ft.</em> {{% /fragment %}}


---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* Our very first neural network looks like this:
{{% fragment %}}<center><img src="nn2.png" width="600px"/></center> {{% /fragment %}}

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* Multiplying $2,000$ sq ft by $180$ gives us  $360,000$. 
* Calculating the prediction is simple multiplication. 
* <strong><em>We needed to think about the weight we’ll be multiplying by.</em></strong>
* That is what training means!

<br />

Area (sq ft) (x)|	Price (y)  | Estimated Price($\hat{y}$)
----------------|------------|---------------------------
2,104	          |  $399,900$ |          $378,720$
1,600	          |  $329,900$ |          $288,000$
2,400	          |  $369,000$ |          $432,000$

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* How bad is our model?
	* Calculate the <em>Error</em>;
	* A better model is one that has less error; 

{{% fragment %}} <em>Mean Square Error</em>{{% /fragment %}}{{% fragment %}}: $2,058$ {{% /fragment %}}

<br />

Area (sq ft) (x)|	Price (y)  | Estimated Price($\hat{y}$) | $y-\hat{y}$ | $(y-\hat{y})^2$
----------------|------------|----------------------------|-------------|---------------
2,104	          |  $399,900$ |          $378,720$         | $21$        |  $449$
1,600	          |  $329,900$ |          $288,000$         | $42$        |  $1756$
2,400	          |  $369,000$ |          $432,000$         | $-63$       |  $3969$

---
## Artificial Neural Networks
* Fitting the line to our data:

<center><img src="manual_training1.gif" width="450px"/></center>

Follows the equation: $\hat{y} = W * x$

---
## Artificial Neural Networks

How about addind the <em>Intercept</em>?

{{% fragment %}} $\hat{y}=Wx + b$ {{% /fragment %}}

---
## Artificial Neural Networks
### The Bias

<center><img src="nn3.png" width="500px"/></center>

---
## Artificial Neural Networks
### Try to train it manually:

<iframe src="manual_NN1.html" height="500px" width="800px">
</iframe>

---
## Artificial Neural Networks
### How to discover the correct weights?
* Gradient Descent:
  * Finding the <em>minimum of a function</em>;
    * Look for the best weights values, <em>minimizing the error</em>;
  * Takes steps <em>proportional to the negative of the gradient</em> of the function at the current point.
  * Gradient is a vector that is <em>tangent of a function</em> and points in the direction of greatest increase of this function. 

---
## Artificial Neural Networks
### Gradient Descent
* In mathematics, gradient is defined as <em>partial derivative for every input variable</em> of function;
* <em>Negative gradient</em> is a vector pointing at the <em>greatest decrease</em> of a function;
* <em>Minimize a function</em> by iteratively moving a little bit in the direction of negative gradient;

---
## Artificial Neural Networks
### Gradient Descent
* With a single weight: 

<center><img src="gd1.jpeg" width="500px"/></center>


---
## Artificial Neural Networks
### Gradient Descent

<iframe src="manual_NN2.html" height="500px" width="800px">
</iframe>

---
## Artificial Neural Networks
### Perceptron
* In 1958, Frank Rosenblatt proposed an algorithm for training the perceptron.
* Simplest form of Neural Network;
* One unique neuron;
* Adjustable Synaptic weights

---
## Artificial Neural Networks
### Perceptron
* Classification of observations into two classes:
<center><img src="perceptron1.png" height="350px"/></center>

###### Images Taken from <a href="https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975" target="_blank">Towards Data Science</a> 

---
## Artificial Neural Networks
### Perceptron
* Classification of observations into two classes:
<center><img src="perceptron2.png" height="350px"/></center>

###### Images Taken from <a href="https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975" target="_blank">Towards Data Science</a> 

---
## Artificial Neural Networks
### Perceptron
* E.g, the OR function:

<center><img src="or1.png" width="550px"/></center>

#### Find the $w_i$ values that could solve the or problem. 

---
## Artificial Neural Networks
### Perceptron
* E.g, the OR function:

<br />
<center><img src="or2.png" width="550px"/></center>

---
## Artificial Neural Networks
### Perceptron
* One possible solution $w_0=-1$, $w_1=1.1$, $w_2=1.1$:

<center><img src="or4.png" width="450px"/></center>

---
## Artificial Neural Networks
### The <a href="https://keras.io" target="_blank">Keras framework</a>

* <em>High-level</em> neural networks API;
* Capable of running on top of <em>TensorFlow</em>, <em>CNTK</em>, or <em>Theano</em>;
* Focus on enabling <em>fast experimentation</em>;
  * Go from idea to result with the <em>least possible delay</em>;
* Runs seamlessly on <em>CPU</em> and <em>GPU</em>;
* Compatible with: <em>Python 2.7-3.12</em> and <em>R</em>;

---
## Artificial Neural Networks
### The <a href="https://keras.io" target="_blank">Keras framework</a>
* Use the implementation of the tensorflow:
  * Create a sequential model (perceptron)
```python
# Import the Sequential model
from tensorflow.keras.models import Sequential

# Instantiate the model
model = Sequential()
```

---
## Artificial Neural Networks
### The <a href="https://keras.io" target="_blank">Keras framework</a>
* Create a single layer with a single neuron:
  * `units` represent the number of neurons;
```python
# Import the Dense layer
from tensorflow.keras.layers import Dense

# Add a forward layer to the model
model.add(Dense(units=1, input_dim=2))
```
{{% note %}}
* Dense means a fully connected layer. 
{{% /note %}}

---
## Artificial Neural Networks
### The <a href="https://keras.io" target="_blank">Keras framework</a>
* Compile and train the model
  * The compilation creates a <a
    href="https://medium.com/tebs-lab/deep-neural-networks-as-computational-graphs-867fcaa56c9" target="_blank">computational graph</a> of the training;
```python
# Specify the loss function (error) and the optimizer
#   (a variation of the gradient descent method)
model.compile(loss="mean_squared_error", optimizer="sgd")

# Fit the model using the train data and also
#   provide the expected result
model.fit(x=train_data_X, y=train_data_Y)
```
{{% note %}}
* Computational Graphs:
  * Nodes represent both inputs and operations;
  * Even relatively “simple” deep neural networks have hundreds of thousands of nodes and edges;
  * Lots of operations can run in parallel;
    * Example: $(x*y)+(w*z)$
  * Makes it easier to create an auto diferentiation strategy;
  * We can user `verbose=1` to increase the output;
{{% /note %}}

---
## Artificial Neural Networks
### The <a href="https://keras.io" target="_blank">Keras framework</a>
* Evaluate the quality of the model:
```python
# Use evaluate function to get the loss and other metrics that the framework
#  makes available
loss_and_metrics = model.evaluate(train_data_X, train_data_Y)
print(loss_and_metrics)
#0.4043288230895996

# Do a prediction using the trained model
prediction = model.predict(train_data_X)
print(prediction)
# [[-0.25007164]
#  [ 0.24998784]
#  [ 0.24999022]
#  [ 0.7500497 ]]
```
{{% note %}}
We can use verbose during the evaluate
{{% /note %}}

---
## Artificial Neural Networks
### Activation Functions
* Describes <em>whether or not the neuron fires</em>, i.e., if it forwards its value for the next neuron layer;
* Historically they translated the output of the neuron into either 1 (On/active) or 0 (Off) - Step Function:
```r
if(prediction[i]>0.5){
  return 1
}
return 0
```

---
## Artificial Neural Networks
### The <a href="https://keras.io" target="_blank">Keras framework</a>
#### Exercise:
Run the example of the Jupyter notebook:
<br />
<a href="https://colab.research.google.com/drive/1hNOR60jfru-b0Vb-ec-Y_yF9pyuy8Wtj?usp=sharing" target="_blank">Perceptron - OR</a>

---
## Artificial Neural Networks
### Perceptron
#### Exercise:
* What about the <em>AND</em> function?

$x_1$|$x_2$|$y$
-----|-----|----
0    |0    |0
0    |1    |0
1    |0    |0
1    |1    |1

{{% fragment %}} <a href="https://colab.research.google.com/drive/10O_OwdFJj9OCNVJJSp8n3_vHhHWrmrbp?usp=sharing" target="_blank">My solution</a>. {{% /fragment %}}

---
## Artificial Neural Networks
### Perceptron - What it <em>can't do</em>!

* The <em>XOR</em> function:

<center><img src="xor1.png" width="500px"/></center>

Check-out what happens when we try to use the same architecture for solving the
XOR function <a
href="https://colab.research.google.com/drive/1NKIpV-SZ38SU6szy_e2hZNBG7MC2_d9G?usp=sharing"
target="_blank">here</a>.

---
## Artificial Neural Networks
### Understanding the training
* Plotting the training progress of the XOR ANN:
```python
history = model.fit(x=X_data, y=Y_data, epochs=2500, verbose=0)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model Training Progression')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')
plt.show()
```
<center><a href="loss_trainning2.png" target="_blank"><img src="loss_trainning2.png" width="250px" /></a></center>

{{% note %}}
* This is called the <em>learning curve</em>;
* In the case of the XOR. <em>What is wrong with that?</em>
{{% /note %}}

---
## Artificial Neural Networks
### Activation Functions
* <em>Multiply the input</em> by its <em>weights</em>, <em>add the bias</em> and <em>applies activation</em>;
* Sigmoid, Hyperbolic Tangent, Rectified Linear Unit;
* <em>Differentiable function</em> instead of the step function;
<center> <img src="activation_functions.png" width="500px"/></center>

{{% note %}}
* With this modification, a multi-layered network of perceptrons would become
differentiable. Hence gradient descent could be applied to minimize the
network’s error and the chain rule could “back-propagate” proper error
derivatives to update the weights from every layer of the network.

* At the moment, one of the most efficient ways to train a multi-layer neural
network is by using gradient descent with backpropagation. A requirement for
backpropagation algorithm is a differentiable activation function. However, the
Heaviside step function is non-differentiable at x = 0 and it has 0 derivative
elsewhere. This means that gradient descent won’t be able to make a progress in
updating the weights.  

* The main objective of the neural network is to learn the values of the weights
and biases so that the model could produce a prediction as close as possible to
the real value. In order to do this, as in many optimisation problems, we’d
like a small change in the weight or bias to cause only a small corresponding
change in the output from the network. By doing this, we can continuously
tweaked the values of weights and bias towards resulting the best
approximation. Having a function that can only generate either 0 or 1 (or yes
and no), won't help us to achieve this objective.
{{% /note %}}

---
## Artificial Neural Networks
### The Bias
<center><img src="bias1.png" width="600px"/></center>

---
## Artificial Neural Networks
### The Bias
Give even more power to our model
<center><img src="bias2.png" width="450px"/></center>

---
## Artificial Neural Networks
### Perceptron - Solving the XOR problem
* 3D example of the solution of learning the OR function:
  * Using <em>Sigmoid</em> function;
<center> <img src="or5.png" width="600px"/></center>

{{% note %}}
That creates a <strong>hyperplane</strong> that separates the classes;
{{% /note %}}

---
## Artificial Neural Networks
### Perceptron - Solving the XOR problem
* Maybe there is a combination of functions that could create hyperplanes that separate the <em>XOR</em> classes:
  * By increasing the number of layers we increase the complexity of the function represented by the ANN:
<center><a href="xor2.png" target="_blank"><img src="xor2.png" width="580px"/></a></center>

{{% note %}}
Now, there are 2 hyperplanes, that when put together, can perfectly separate the classes;
{{% /note %}}

---
## Artificial Neural Networks
### Perceptron - Solving the XOR problem
* The combination of the layers:
<center><a href="xor3.png" target="_blank"><img src="xor3.png" width="300px"/></a></center>

{{% note %}}
* That is what people mean when they say we don't know how deep neural networks
  work. We know that it is a composition of functions, but the shape of that
  remains a little bit hard to define;
* Yesterday we saw polynomial transformation of features - in that we saw that
  we changed the shape of the regression line being built;

{{% /note %}}


---
## Artificial Neural Networks
#### <em>Multilayer Perceptrons</em> - Increasing the model power
* Typically represented by composing many different
functions:
$$y = f^{(3)}(f^{(2)}(f^{(1)}(x)))$$

* The <em>depth</em> of the network - the <em>deep</em> in deep learning! (-;

---
## Artificial Neural Networks
#### <em>Multilayer Perceptrons</em> - Increasing the model power
* Information flows from $x$ , through computations and finally to $y$.
* No feedback!

---
## Artificial Neural Networks
### Perceptron - Solving the XOR problem
* Implementing an ANN that can solve the XOR problem:
  * Add a new layer with a larger number of neurons:

```python
...
#Create a layer with 4 neurons as output
model.add(Dense(units=4), activation="sigmoid", input_dim=2)

# Connect to the first layer that we defined
model.add(Dense(units=1, activation="sigmoid")
```

Let's check if that solves our XOR problem <a
href="https://colab.research.google.com/drive/1hpRRtJuC78uPXJE68oOjRaM03LVV_rgo?usp=sharing"
target="_blank">here</a>.

{{% note %}}
Train for little steps and then increase the number of epochs
{{% /note %}}

---
## Artificial Neural Networks
### Understanding the training  
* Plot the architecture of the network:
```python
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)
```
<center><img src="nn_architecture.png" width="300px" /></center>


{{% note %}}
The ? means that they take as much examples as possible;
{{% /note %}}


---
## Artificial Neural Networks
### Problems with the training procedure:
* Saddle points:
	* No matter how long you train your model for, <em> the error remains (almost) constant!</em> 
<center><a href="saddle.png" target="_blank"><img src="saddle.png" width="300px" /></a></center>

{{% note %}}
* That eventually happens because of a bad optimization function;
* Imagine that you could add momentum to the gradient descent - probably it
  could continue updating;
* In the XOR case, there are 16 local minimums that have the highest conversion
  if the weights are initialized between 0.5 and 1. 
{{% /note %}}

---
## Artificial Neural Networks
### Optimization alternatives
* The Gradient Descent is <em>not always the best option</em> to go with:
  * Only does the update after <em>calculating the derivative for the whole
    dataset</em>;
  * Can take a <em>long time to find the minimum</em> point;

---
## Artificial Neural Networks
### Optimization alternatives
* The <a href="gd.gif" target="_blank">Gradient Descent</a> is <em>not always the best option</em> to go with:
  * For non-convex surfaces, it may only find the local minimums - <a href="gd2.gif" target="_bank">the saddle situation</a>;
  * <strong><em>Vectorization</em></strong>

<a href="vectorization.jpeg" target="_blank"><center><img src="vectorization.jpeg" width="450px" /></center></a>

{{% note %}}
* For large datasets, the vectorization of data doesn’t fit into memory.
{{% /note %}}

---
## Artificial Neural Networks
### Optimization alternatives
* Gradient Descent alternatives:
  * <a href="sgd.gif" target="_blank">Stochastic Gradient Descent</a>: updates at each input;
  * <a href="minibatch.gif" target="_blank">Minibatch Gradient Descent</a>: updates after reading a batch of examples;
<br /><br />

<center>

###### Animations taken from Vikashraj Luhaniwal <a href="https://towardsdatascience.com/why-gradient-descent-isnt-enough-a-comprehensive-introduction-to-optimization-algorithms-in-59670fd5c096" target = "_blank">post</a>.
</center>

{{% note %}}
Minibatch:
* Updates are less noisy compared to SGD which leads to better convergence.
* A high number of updates in a single epoch compared to GD so less number of epochs are required for large datasets.
* Fits very well to the processor memory which makes computing faster.
{{% /note %}}

---
## Artificial Neural Networks
### Optimization alternatives
###### Adaptative Learning Rates:
* <a href="adagrad.gif" target="_blank">Adagrad</a>, <a href="rmsprop.gif" target="_blank">RMSProp</a>, <a href="adam.gif" target="_blank">Adam</a>;
###### 
<br /><br />

<center>

###### Animations taken from Vikashraj Luhaniwal <a href="https://towardsdatascience.com/why-gradient-descent-isnt-enough-a-comprehensive-introduction-to-optimization-algorithms-in-59670fd5c096" target = "_blank">post</a>.
</center>

{{% note %}}
* For Adagrad: 
  * Parameters with small updates(sparse features) have high learning rate whereas the parameters with large updates(dense features) have low learning rateupdates at each input;
  * The learning rate decays very aggressively
* RMSProp: A large number of oscillations with high learning rate or large gradient
{{% /note %}}

---
## Artificial Neural Networks
### Multilayer Perceptron - XOR
* Try another optimizer:
```python
model.compile(loss="mean_squared_error", optimizer="adam")
```
My <a href="https://colab.research.google.com/drive/1tsK3aESxc0xGTVLhry-vwMpK2j8mRl-o?usp=sharing" target="_blank">solution</a>

---
## Artificial Neural Networks
### Predicting probabilities
* Imagine that we have <em>more than 2 classes</em> to output;
* One of the <em>most popular usages</em> for ANN;
<center><a href="classification_example.jpeg" target="_blank"><img src="classification_example.jpeg" width="300px"/></a></center>

---
## Artificial Neural Networks
### Predicting probabilities
* The <a href="https://en.wikipedia.org/wiki/Softmax_function" target="_blank">Softmax</a> function;
* Takes an array and outputs a probability distribution, i.e., <em>the probability
  of the input example belonging to each of the classes</em> in my problem;
* One of the activation functions available at `Keras`:
```r
layer_dense(units = 2, activation = 'softmax')
```


{{% note %}}
* Softmax - function that takes as input a vector of K real numbers, and normalizes it into a probability distribution
{{% /note %}}

---
## Artificial Neural Networks
### Loss functions
* For regression problems
  * Mean squared error is <em>not always the best one to go</em>;
    * What if we have a three classes problem?
  * Alternatives: `mean_absolute_error`, `mean_squared_logarithmic_error`

{{% note %}}
* logarithm means changing scale as the error can grow really fast;
{{% /note %}}

---
## Artificial Neural Networks
### Loss functions
* <a href="https://en.wikipedia.org/wiki/Cross_entropy" target="_blank">Cross Entropy</a> loss:
  * Default loss function to use for binary classification problems.
  * Measures the <em>performance of a model</em> whose output is a probability value between 0 and 1;
  * <em>Loss increases</em> as the <em>predicted probability diverges</em> from the actual label;
  * A <em>perfect model</em> would have a log loss of 0;

{{% note %}}
*  As the correct predicted probability decreases, however, the log loss increases rapidly:
  * In case the model has to answer 1, but it does with a very low probability;

* If you have events and probabilities, how likely is it that the events happen based on the probabilities? 
  * If it is very likely, we have a small cross-entropy and if it is not likely we have a high cross-entropy. 

{{% /note %}}

---
## Artificial Neural Networks
### What about the overfitting?
<img src="overfitting.png" width="300px"/>

---


## Interpretation of the Test Set

### Does Perfect Metric on the Test Set Mean the Model is Perfect?

**Not necessarily:** In no non-trivial problem will you have access to a completely representative database of the problem.
Evaluating with a test set *alleviates* but doesn't solve the problem.
There will never be enough examples to perfectly model the phenomenon.

---


## Analysis of Model Error

### Types of Errors

* Prediction error can be divided into three parts:
  - **Irreducible Error:** cannot be eliminated regardless of the algorithm used.
    - **Introduced from the chosen problem framework**.
    - **Caused by unknown factors**.
  - **Bias Error:** assumptions made by a model to make the target function easier to learn.
  - **Variance Error:** the amount the target function estimate will change if different training data are used.

---


## Analysis of Model Error

### Bias Error

* Difference between the expected (or average) prediction of our model and the correct value we are trying to predict.
* Imagine repeating the entire model-building process more than once:
  - **Each time you gather new data and perform a new analysis, you create a new model.**
  - **Due to randomness in the underlying data sets, resulting models will have a variety of predictions.**
  - **Measures how far, on average, predictions of these models are from the correct value.**
* Our model has bias if it systematically predicts below or above the target variable.

---


## Analysis of Model Error

### Variance Error

* In a sense, captures the **model's generalization capability**.
* How much our prediction would change if we trained it with different data.
* Ideally, it shouldn't change much from one training data set to the next.
* Algorithms with high variance are **strongly influenced by the specifics of training data.**
* Generally, nonlinear machine learning algorithms that are very flexible have **high variance.**
  - **For example, Polynomial Regression with high-degree polynomials!**

---

## The Bias-Variance Tradeoff

* **Bias** error arises due to incorrect assumptions made by the learning algorithm. Excessive bias can lead the algorithm to overlook important connections between features and target outcomes, resulting in underfitting.

* **Variance** represents the error stemming from the algorithm's susceptibility to minor variations in the training dataset. Elevated variance might occur when the algorithm models the random noise present in the training data, causing overfitting.

---


## Dilemma: Variance vs Bias

* Low bias: suggests fewer assumptions about the shape of the target function.
  - **Regression Trees, KNN Regression**
* High bias: suggests more assumptions about the shape of the target function.
  - **Linear Regression, Logistic Regression**
* Low variance: suggests small changes in the estimate of the target function with changes in the training data set.
  - **Linear Regression, Logistic Regression**
* High variance: suggests large changes in the estimate of the target function with changes in the training data set.
  - **Regression Trees, KNN Regression**

---


## Dilemma: Variance vs Bias

<img src="variance_x_bias.png" width="400px"/>
<br />

* Increasing bias will decrease variance.
* Increasing variance will decrease bias.

---


## Dilemma: Variance vs Bias
## Tradeoff

<img src="bias_variance1.png" width="450px"/>

---


## Dilemma: Variance vs Bias

* A very simple model with few parameters has high Bias and low Variance.
* A complex model with a large number of parameters will have high Variance and low Bias.
* One should aim for balance, avoiding overfitting while not underfitting the data.

---

## Dilemma: Variance vs Bias

* Models should try to **generalize** beyond what is observed in the training set.
* **Regularization** plays a role in controlling the overfitting of classifiers.

---


## Visualizing the Overfitting
### How Decision Trees change after removing a few examples

<img src="fit_tree.png" width="300px"/>
<img src="fit_tree_mod.png" width="300px"/>

---

## Regularization

* Decreases variance by reducing learning effectiveness.
* Penalizes model complexity.
* Nearly all learning algorithms have some form of regularization mechanism.

---


## Artificial Neural Networks
### Dealing with overfitting
* <em>Dropout</em> layers:
  * Randomly *disable* some of the neurons during the training passes;

<center><a href="dropout.gif" target="_blank"><img src="dropout.gif" width="500px"/></a></center>

---
## Artificial Neural Networks
### Dealing with overfitting
* <em>Dropout</em> layers:
```python
# Drop half of the neurons outputs from the previous layer
model.add(Dropout(0.5))
```

{{% note %}}
* “drops out” a random set of activations in that layer by setting them to zero;
* forces the network to be redundant;
* the net should be able to provide the right classification for a specific example even if some of the activations are dropped out;
{{% /note %}}

---
## How does it realy do it?

<video width="620" height="440" controls>
  <source src="ann.mp4" type="video/mp4">
</video>

---
## Artificial Neural Networks
### Larger Example
* The <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> dataset: database of handwritten digits;
* Dataset included in Keras;
<center><a href="mnist.png" target="_blank"><img src="mnist.png" width="500px"/></a></center>

---
## Artificial Neural Networks
### The MNIST MLP
* Try to improve the classification results using <a href="https://colab.research.google.com/drive/1AnGJz_R0PJF0d83ye_3y7NPGuX5YipBi?usp=sharing" target="_blank">this notebook</a>:
* Things to try:
  * Increase the number of neurons at the first layer;
  * Change the optimizer and the loss function;
  * Try `categorical_crossentropy` and `rmsprop` optimizer;
  * Try adding some extra layers;

---
## Artificial Neural Networks
### The MNIST MLP
* Try to improve the classification results using <a href="https://colab.research.google.com/drive/1AnGJz_R0PJF0d83ye_3y7NPGuX5YipBi?usp=sharing" target="_blank">this notebook</a>:
* Things to try:
  * Try addind `Dropout` layers;
  * Increase the number of `epochs`;
  * Try to <em>normalize the data</em>!

* What is the best accuracy?
* <a href="https://colab.research.google.com/drive/1LnkhSA7XbEWMNdaebOXxsOENr6m-0vpZ?usp=sharing" target="_blank">My solution</a>.
---
# Convolutional Neural Networks

---
### Introduction

* **Deep learning** techniques belong to a branch of machine learning that uses neural networks with multiple layers. The basic idea is to employ layers to progressively learn different levels of features from input data. The levels of abstraction increase as more layers are used for feature extraction and learning.

* Among deep learning techniques, special attention has been given to Convolutional Neural Networks (CNNs). These models have a high capacity for data representation, yielding promising results in numerous fields of knowledge.

---
* The basic idea is to use "raw data" as input and allow the network to learn the most important features for the problem at hand. This eliminates the need to manually extract features (handcrafted features).

<img width="450" src="CNN_Fig1.png" />

---
* Generally, CNNs are composed of two main modules: (i) feature learning and (ii) classification. Feature learning is performed through convolution, pooling, and activation operations. The classification step typically consists of fully connected layers and a softmax output layer.

<img width="700" src="CNN_Fig2.png" />

---
* But what makes CNNs so interesting for various pattern classification tasks? The secret lies in the **feature learning** step, where important information (like texture) is learned at different levels. The interesting part is that these networks are less susceptible to rotation, translation, and scale issues.

* To understand how they work, we will first study the feature learning layers and then move on to the classification layers. As mentioned, each layer of the learning step consists mainly of (i) convolution, (ii) pooling, and (iii) activation operations.

---
### Feature Learning

#### Convolution

* The first operation we will see is **convolution**, which is widely used in image processing and computer vision tasks, such as image filtering (blurring and noise) and edge detection, for example. Let's see an example.

<img width="450" src="CNN_Fig3.png" />

---
<img width="450" src="CNN_Fig4.png" />

* The central position of the 9x9 input is replaced by its convolution with the 5x5 mask
* The value is stored in the 5x5 output matrix (feature map).
* The procedure is repeated until the entire input matrix has been evaluated. 

---
### What are the hyperparameters involved in a convolution operation?

- **Parameters vs. hyperparameters**.
  - What type of mask (kernel) will we use?
  - What is the best dimension?
  - How many filters will be employed?
  - What is the value of the stride?


---
* What we have, in practice, are values in the masks that can be interpreted as **weights** that will be learned by the CNN during its training process. 
* How do we calculate the number of these parameters?

<img width="450" src="CNN_Fig7.png" />

---

| Layer            | Type            | Input   | Output                               |
|------------------|-----------------|---------|--------------------------------------|
| Input            | Data            | 0       | 3                                    |
| Intermediate 1   | Convolutional   | 3       | (27+27)+2 biases = 56                |
| Intermediate 2   | Convolutional   | 2       | 54+3 biases = 57                     |
| Intermediate 3   | Flattened       | -       | 20x20x3 = 1,200                      |
| Output           | Dense           | 1,200   | 1,200x2+2 biases = 2402              |

**Number of parameters to be learned: 2,515.**

---
### What is the role of the hyperparameter values in a CNN?

- **Kernel size:** plays a very important role. **Small kernels** extract more local information (local features) as size reductions between layers are smaller, allowing for deeper architectures. On the other hand, **larger kernels** result in faster size reductions of feature maps and extract more global information.
- Stride value: has a similar impact to kernel size, with larger values resulting in faster reductions of feature maps. Smaller stride values result in more features being learned.
- Since smaller stride values and kernel sizes enable more features to be learned, why not always adopt them? **This requires larger datasets.**

---
#### In summary, we have to make decisions about the following items regarding a convolution layer:

1. Type of padding.
2. Kernel size.
3. Stride value.
4. Number of filters.

---
#### Activation

* CNNs are usually composed of numerous layers, making it undesirable to use sigmoid or hyperbolic tangent activation functions (saturation problems). Another interesting feature of activation functions is their **non-linearity**, allowing the network to learn non-linear decision functions.

* One of the most used activation functions is ReLU (Rectified Linear Unit) due to its simplicity and high degree of non-linearity. Its formulation is given as follows:

$ \text{ReLU(x)} = \max\{0,x\}. $

---
* Below is the graph of the ReLU(x) function. Note that the function only returns a value when its input is greater than 0, aiding in the training time.
* The derivative of $ReLU(x)=0$, case $x\leq0$, and $1$ otherwise

<img width="350" src="CNN_Fig10.png" />

---
#### Pooling

* There are different types of pooling operations, whose main goal is to **decrease the resolution** (downsampling) of the feature maps and add **invariance properties** to the network. The size reduction of feature maps leads to a decrease in the number of parameters to be learned by the network, allowing for more efficient training.

* Among the main types of pooling, we can mention:
  - Max-Pooling
  - Average Pooling
  - Global Pooling

---
#### Some illustrations to exemplify the functioning of the mentioned types of pooling.
* Max-Pooling (stride = 2)
<p><img width="450" src="CNN_Fig8.png" /></p>

---
#### Some illustrations to exemplify the functioning of the mentioned types of pooling.
* Average Pooling (stride = 2)
<p><img width="450" src="CNN_Fig9.png" /></p>

---
* The global pooling technique is more radical in the context of downsampling, as it reduces the entire feature map to a single value. In this case, we can use either max-pooling or average pooling.

* Generally, max-pooling layers tend to provide better results, as it is more informative to use the highest value within a window than to "mask" them with their average value.

---
#### Flattening

* Before sending our data through convolution, pooling, and activation layers to the fully connected layers, we need to "flatten" the **tensor** (data). The operation in this layer is quite simple, as we receive input with multiple dimensions (feature maps) and the output is a one-dimensional vector, as illustrated below.

<img style="margin-right:100px;" width="250" src="CNN_Fig11.png" /> 
<img width="280" src="CNN_Fig12.png" />

---
#### Fully Connected + Dropout

* The final part consists of adopting fully connected layers, similar to an MLP Neural Network, with a softmax output at the end. It is also common to adopt a regularization technique known as **Dropout**, which "removes" neurons randomly to speed up the training process and prevent overfitting.

<img width="500" src="CNN_Fig13.png" />

---
* The softmax function $\sigma:\mathbb{R}^K\rightarrow [0,1]^K$ is a generalization of the logistic function, where \(K\) corresponds to the number of classes.

<img width="250" src="CNN_Fig14.png" />

* **Why softmax and not the logistic function?** Usually, the logistic function is applied to each output neuron without considering all the others. In this case, softmax results in a probability of the neuron of each class responding to an input stimulus.

---
# Group Projects

---
{{<slide background-image="cms.png">}}
# <span style="color:#fff;"> Particle Physics</span>

---
## Artificial Neural Networks
### The Particle Physics Project

<center><a href="atlas_particle_shower.jpg" target="_blank"><img src="atlas_particle_shower.jpg" width="500px"/></a></center>

---
## Artificial Neural Networks
### The Particle Physics Project

<center><a href="jet-images.png" target="_blank"><img src="jet-images.png" width="500px"/></a></center>

---
## Artificial Neural Networks
### The Particle Physics Project
* Quantum Chromodynamics
<center><a href="qcd.png" target="_blank"><img src="qcd.png" width="500px"/></a></center>

---
## Artificial Neural Networks
### Signal VS Background

<center><a href="backgroundVSsignal.png" target="_blank"><img src="backgroundVSsignal.png" width="700px"/></a></center>

---
## Artificial Neural Networks
### Signal VS Background

Run this <a href="https://colab.research.google.com/drive/1zauFbl7qwyv4wXFp1K5ldfXxD_1QzO6R?usp=sharing" target="_blank">Jupyter Notebook</a> for performing the Jet Classification.

---

{{<slide background-image="COVID19_CT.jpg">}}
# <span style="color:#ff0000;">COVID19 Chest CT Image Processing</span>

---
## Artificial Neural Networks
### COVID19 Diagnosis

<center><a href="COVIDCT1.png" target="_blank"><img src="COVIDCT1.png" width="500px"/></a></center>

---
## Artificial Neural Networks
### COVID19 Diagnosis

Run this <a href="https://colab.research.google.com/drive/1S1UQxmDHewXrR_C69mZmoXaP57nGH2X6?usp=sharing" target="_blank">Jupyter Notebook</a> for performing the CT Image Classification.
