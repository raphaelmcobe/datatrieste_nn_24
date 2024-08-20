# Autoencoders

---
### Autoencoders

#### Some Definitions:

* Unsupervised learning algorithm that applies backpropagation, setting the target values to be equal to the inputs:
    * Uses $y^{(i)} = x^{(i)}$
    * <em>Semi-supervised</em>
* One (or more) hidden layers that describe a latent space encoding used to represent the input data.
* Two parts:
    * Encoder function $h = f(x)$, and
    * Decoder function that produces the <em>reconstruction</em> $r = g(h)$.
* Designed to be <em>incapable of perfectly copying the data</em>.

{{% fragment %}}
**Question:** Why is simply learning to set $g(f(x)) = x$ everywhere not particularly useful?
{{% /fragment %}}

---

### Autoencoders

#### Some Definitions:

* Special case of feedforward networks
* Trained with the <em>same techniques</em>, such as gradient descent with minibatch, backpropagation, etc.
* Restricted to only approximately copying, i.e., producing data that <em>merely resembles the training data</em>.
* Forced to prioritize only certain _aspects_ of the input:
    * Often learns useful properties of the data, e.g., relevant features.
* Traditionally used for <em>dimensionality reduction</em> or <em>feature learning</em>.

---

### Autoencoders

#### Example:

<p align="justify">
Suppose the inputs $x$ are pixel intensity values of a <em>10×10 image</em> (100 pixels) - $n = 100$, and there are $s_2 = 50$ hidden units in layer $L_2$.
</p>
<p align="justify">
From the definition of Autoencoders, we have $y \in \mathcal{R}^{100}$. Since there are only 50 hidden units, the network is forced to learn a "compressed" representation of the input.
</p>
<p align="justify">
Given only the activation vector of hidden units $a^{(2)} \in \mathcal{R}^{50}$, it must attempt to "reconstruct" the 100-pixel input $x$.
</p>

---

### Autoencoders

#### Some Definitions:

* If there is any underlying structure in the data, such as some of the input attributes being correlated, then this algorithm <em>will be able to discover some of these correlations</em>.
* This simple form of autoencoder will likely learn a low-dimensional representation very similar to PCA
* Can be thought of as a <em>data compression algorithm</em>
* The compression and decompression functions are:
    1. Data-specific,
    2. Lossy, and
    3. Automatically learned from examples rather than designed by us.

---

### Autoencoders

#### Some Definitions:

![PCA vs. Autoencoder](pca-vs-ae.png)

---

### Autoencoders

#### Questions:
<ol>
<li>Why don't Autoencoders represent optimal compression algorithms <em>for general applications</em>?</li>

{{% fragment %}}
<li>Why do they <em>need to be lossy</em>?</li>
{{% /fragment %}}

{{% fragment %}}
<li>Would an autoencoder trained on face images do a good job compressing tree images?</li>
{{% /fragment %}}

---

### Autoencoders

#### So, What Are They Good For?

* <em>Data Compression?</em>
    * Almost impossible to outperform standard algorithms like JPEG, MP3, etc.;
    * We can **improve performance by restricting the type of data** it uses;
        * **This reduces generalization capability.**
    * Generally impractical for real-world data compression/compaction problems:
        * Can only be used on data similar to what it was trained on.

---

### Autoencoders

#### So, What Are They Good For?

* <em>Dimensionality Reduction:</em>
    * If the decoder is linear and the cost function is MSE, an Autoencoder learns to span the same subspace as PCA.
* <em>Denoising:</em>
    * The data is partially corrupted by noise;
    * The model is trained to predict the original, non-corrupted data as its output.

---

### Undercomplete Autoencoders

* Autoencoder whose coding dimension is smaller than the input dimension;
* Forces the autoencoder to capture the most relevant attributes of the training data;
    * Also known as bottlenecks;
* Minimizes the cost function, where $f$ is the encoder function and $g$ is the decoder function, by adjusting parameters $\theta$ and $\phi$:
$$
J(w,w\prime) = \frac{1}{n} \sum_{i=1}^{n} (x^{(i)} - f_w(g_{w\prime}(x^{(i)})))^2
$$
* Nonlinear encoder functions $f$, as well as decoders $g$, can learn a more powerful nonlinear generalization than PCA.

---

### Undercomplete Autoencoders

#### Implementation

General Architecture:

<img src="autoencoder1.png" width="700px">

_Source: [Autoencoder architecture by Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)_

---

### Undercomplete Autoencoders

#### Implementation:

Dimensionality reduction for 3D data:

```python
encoding_dim = 2
input_layer = keras.Input(shape=(3,))
encoded = layers.Dense(encoding_dim, activation="sigmoid")(input_layer)
decoded = layers.Dense(3, activation="sigmoid")(encoded)
autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(loss="mse", optimizer="SGD")
```

---

### Undercomplete Autoencoders

#### Implementation

![Autoencoder architecture](autoencoder2.png)

---

### Undercomplete Autoencoders

#### Implementation

<img src="autoencoder3.png" width="300px">


<img src="autoencoder4.png" width="300px">

---

### Undercomplete Autoencoders

#### What Are They Not Good At?

<img src="autoencoder5.png" width="500px">

---

### Undercomplete Autoencoders

#### What Are They Not Good At?

* Some of the biggest challenges regarding the latent space are:
    * <em>Gaps in the latent space</em>: we don't know what data points might look like in those spaces.
    * <em>Separability in the latent space</em>: there are also regions where the labels are intermixed/randomly scattered.
    * <em>Discrete latent space</em>: we don't have a trained statistical model for an arbitrary input.

---

### Autoencoders

#### Limitations

* Unfortunately, autoencoders **do not learn anything useful** if the encoder and decoder **have too much capacity**.
* This also happens if the latent space has the same dimension as the input.
* Even a linear encoder and decoder can <em>learn to copy the input</em> to the output:
    * <em>Nothing useful is learned about the data distribution.</em>

---

### Regularized Autoencoders

* Uses a loss function that "encourages" the model to have other properties, such as sparsity of the representation and robustness to noise or missing inputs;
* Can be nonlinear and overcomplete, but still learn something useful about the data distribution;
* The two most common Regularized Autoencoders are:
    * **Sparse Autoencoders**: sparsity penalty added to their original cost function; -> Eq. 1
    * **Denoising Autoencoders**: add noise (Gaussian, for example) to the inputs, forcing the model to learn important features;

---

### Regularized Autoencoders

#### Sparse Autoencoders

* Essentially an autoencoder whose training criterion involves sparsity as a penalty term;
* Has a latent dimension larger than the input or output dimensions;
* Typically used to **learn features for another task**, such as classification;
* Think of the penalty simply as a **regularization term**;
* We want to restrict neurons to be **inactive most of the time**;
* Reduces the propensity for overfitting the network;
* It can **no longer copy the input through certain neurons**:
    * in each run, **these neurons may not be the active ones**.

---

### Regularized Autoencoders

#### Sparse Autoencoders

![Sparse Autoencoder](autoencoder6.png)
_Source: Image by Shreya Chaudhary_

---

### Regularized Autoencoders

#### Sparse Autoencoders - Implementation

In Keras, this can be done by adding a regularizer with an `activity_regularizer` to the `Dense` layer:

```python
encoded = layers.Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
```

With the regularization added, the model is less likely to overfit and **can be trained longer**.

---

### Regularized Autoencoders

#### Denoising Autoencoders

* The input is <em>partially corrupted</em> by adding noise or "masking" some values of the input vector stochastically;
* The model is trained to <em>recover the original input</em> (note: not the corrupted one);

<img src="denoising_autoencoder.png" width="400px">

where $\mathcal{M_D}$ defines the mapping of true data samples to noisy or corrupted ones.

---

### Regularized Autoencoders

**Denoising Autoencoders**

![Denoising Autoencoder](autoencoder7.png)

_Source: Image by Lilian Weng_

---

### Regularized Autoencoders

#### Denoising Autoencoders

* Motivated by the fact that humans can easily recognize an object even with partially occluded vision;
* To "repair" the input, the DAE must discover the <em>relationship between the input dimensions to infer the missing parts</em>;
* In images, the model is likely to rely on evidence gathered from a <em>combination of many input dimensions</em> to recover the noise-free version:
    * This creates a good foundation for learning a robust latent representation;
* In the [original DAE paper](https://dl.acm.org/doi/abs/10.1145/1390156.1390294), a fixed proportion of input dimensions is randomly selected, and their values are forced to 0 (similar to dropout?).

---

### Regularized Autoencoders

#### Denoising Autoencoders - Implementation

Example for the MNIST dataset:

```python
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
```

![Digits from MNIST after adding noise](autoencoder8.png)

[_Jupyter Notebook_](https://colab.research.google.com/drive/1KzErreZmln2zvDnEzXPIOqwpo2-8pgZ7?usp=sharing) with the example implementation of Denoising Autoencoder for the MNIST.

---

# Variational Autoencoders

---
### Introduction

### Recap: The problem with the standard autoencoder

<font size="5">

* Besides some efficient applications like denoising autoencoders, they are quite limited.
* The latent space to which they convert their inputs, and where their encoding vectors reside, may not be continuous or allow for easy interpolation.
* For example, training an autoencoder on the MNIST dataset and visualizing the encodings in a 2D latent space reveals the formation of distinct clusters:
</font>

<img src="vae1.png" width="300px">

---

### Variational Autoencoders

#### Recap: The problem with the standard autoencoder

* When building a generative model, we don't want to replicate the input data:
    * Randomly sample from the latent space, or
    * Generate variations in an input image from a continuous latent space;
    * If the space has discontinuities and you sample/generate a variation from there, the decoder will simply produce an unrealistic output;
    * The decoder has no idea how to handle that region of the latent space;
    * During training, it never saw encoded vectors coming from that region of the latent space;

---

### Variational Autoencoders

#### Definitions

* Variational Autoencoders (VAEs) have a fundamentally unique property that separates them from common autoencoders:
    * Their latent spaces are inherently continuous;
    * The continuity of the latent space allows for easy random sampling and interpolation.
* Their encoder does not produce a coding vector of size $n$;
* Instead, it generates two vectors of size $n$:
    * A vector of means, $\mu$, and
    * Another vector of standard deviations, $\sigma$.
    * The mean and the standard deviation of the $i$-th random variable, $X_i$, from which we sample to obtain the sampled encoding passed to the decoder;

---

### Variational Autoencoders

#### Definitions

![Variational Autoencoder with $\mu$ and $\sigma$ vectors](vae2.png)
_Source: <a href="https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf" target=_blank>Variational Autoencoder architecture by Irhum Shafkat</a>_

---

### Variational Autoencoders

#### Example

In a scenario where we have an input signal with 500 features and we intend to reduce this signal to just 30, we could build a VAE as follows:

<img src ="vae3.png" width="300px">

---

### Variational Autoencoders

#### Definitions

<img src="vae4.png" width="500px">

---

### Variational Autoencoders

#### Definitions

<font size="5">

* Stochastic generation of encoding vectors.
    * For the same input, keeping the mean and standard deviation the same, the actual encoding will vary on each pass due to sampling.
* The mean vector controls where the encoding of an input should be centered;
* The standard deviation controls how much the encoding can vary from the mean (the area).

</font>

<img src="vae5.png" width="250px">


---

### Variational Autoencoders

#### Definitions

* Not just a single point in the latent space refers to a sample of that class.
* All nearby points refer to the same within a $\sigma$ radius;
* The goal here is to create a more homogeneous latent space, eliminating discontinuity;
    * The model is now exposed to a certain degree of local variation by varying the encoding of a sample;
    * We want overlap between samples that are also not very similar;
        * Interpolation between classes;

---

### Variational Autoencoders

#### Definitions

* There are no limits to the values that the $\mu$ and $\sigma$ vectors can assume:
    * The encoder can learn to generate very different $\mu$ values for different classes, clustering them and minimizing $\sigma$;
    * It can reach a point that appears as a single point.
* Desirable: Encodings that are as close as possible while still distinct, allowing for smooth interpolation and the possibility of constructing new samples.

---

### Variational Autoencoders

#### Definitions

What we want and what we can achieve:

![Desired latent space](vae6.png)

---

### Variational Autoencoders

#### Definitions - The KL Divergence

<font size="5">

* [Kullback-Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
* Measures how much they diverge from each other;
* For VAEs, the cost by KL is equivalent to the sum of all KL divergences between the component $X_i\sim\mathcal{N}(\mu_i, \sigma_i^2)$ and the standard normal distribution.
    * This measure is minimized when $\mu_i=0$ and $\sigma_i=0$;
* When the divergence is calculated between univariate distributions, it can be simplified to [1]: 
$$
\sum_{i=1}^n \sigma_i^2+\mu_i^2 - \log(\sigma_i^2)-1
$$

_Source: [Deriving the KL divergence loss for VAEs](https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048)_

</font>

---

### Variational Autoencoders

#### Definitions - The KL Divergence

<font size="5">

* Forces the encoder to distribute all encodings uniformly around the center of the latent space;
* Using purely the result of the KL loss results in a latent space with densely placed encodings randomly, near the center of the latent space;
* The decoder finds it impossible to decode anything meaningful from this space;

</font>
<img src="vae7.png" width="250px">


---

### Variational Autoencoders

#### Grouping the information...

<font size="5">

* Use the KL divergence as a penalization mechanism;
* Optimize the composite loss (e.g., reconstruction, or cross-entropy) and the KL divergence;
    * Generate a latent space that maintains the similarity of nearby encodings;
    * Globally, it is densely packed near the origin of the latent space;
    * Balance is achieved by the clustering nature of the reconstruction loss and the dense packing nature of the KL loss;

</font>

<img src="vae8.png" width="250px">
