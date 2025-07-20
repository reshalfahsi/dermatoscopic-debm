# **Generating Dermatoscopic Images With the Deep Energy-Based Model**

<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/dermatoscopic-debm/blob/main/Generating_Dermatoscopic_Images_With_the_Deep_Energy_Based_Model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>


Implicit generative models (e.g., diffusion models and GANs) are a promising avenue in which data generation bypasses the explicit restriction of adhering to the normalized data distribution. In the wake of deep learning, Energy-Based Models have jumped onto this path even harder. In the paper *"Implicit Generation and Modeling with Energy-Based Models"*, a deep learning model is used to model a negative energy function $-E\_\theta(x)$. The energy function defines the data probability distribution $q\_\theta(x)$ and an implicit generator by means of Langevin dynamics. It is optimized to assign low energy to real samples and high energy to generated ones. Next, the learned energy landscape is leveraged by Langevin dynamics for sample generation via stochastic gradient-based updates. The model is trained using contrastive divergence, comparing the energy of real images with that of synthetic samples initialized from noise (kind of similar to what Wasserstein GAN optimizes). The contrastive divergence stems from maximum likelihood estimation (MLE) of $q\_\theta(x)$. By doing so, we basically train a score-based model (reminiscent to diffusion models). Experimental results demonstrate that even a simple convolutional energy network can capture meaningful structure in medical image data and generate realistic dermatoscopic samples. It highlights the feasibility of applying EBMs in medical imaging and underscores their potential as a generative modeling tool in healthcare applications.

## What Is an Energy-Based Model (EBM)?

An EBM models the **probability density** of data $x \in \mathbb{R}^d$ using an **energy function** $E\_\theta(x)$, defined by a neural network with parameters $\theta$. However, in practice, the neural network models the negative energy function. The probability is given by:


$$
q\_\theta(x) = \frac{e^{-E\_\theta(x)}}{Z(\theta)}, \quad \text{where } Z(\theta) = \int e^{-E\_\theta(x)} dx
$$


* The lower the energy $E\_\theta(x)$, the more likely the data point $x$.
* $Z(\theta)$ is the partition function (often intractable).


### Maximum Likelihood Estimation (MLE) in EBMs

The goal of MLE is to find parameters $\theta$ that maximize the log-likelihood of the data:


$$
\mathcal{L}(\theta) = \mathbb{E}\_{x \sim p\_{\text{data}}} \[\log q\_\theta(x) ]
$$


Plug in the definition of $q\_\theta(x)$:


$$
\log q\_\theta(x) = -E\_\theta(x) - \log Z(\theta)
$$


So, the expected log-likelihood becomes:


$$
\mathcal{L}(\theta) = \mathbb{E}\_{x \sim p\_{\text{data}}} \[-E\_\theta(x) ] - \log Z(\theta)
$$


Since $\log Z(\theta)$ depends on $\theta$, we need its gradient when optimizing $\mathcal{L}(\theta)$.


### Gradient of the Log-Likelihood

We differentiate the expected log-likelihood with respect to $\theta$ (i.e., the expected score function $s(x; \theta) := \nabla\_\theta \log q\_\theta(x)$ ):


$$
\nabla\_\theta \mathcal{L}(\theta) = \mathbb{E}\_{x \sim p\_{\text{data}}} \[\nabla\_\theta \log q\_\theta(x) ]
$$


Remember the previous part:


$$
\nabla\_\theta \mathcal{L}(\theta) = \mathbb{E}\_{x \sim p\_{\text{data}}} \[-\nabla\_\theta E\_\theta(x) ] - \nabla\_\theta \log Z(\theta)
$$


The tricky part is $\nabla\_\theta \log Z(\theta)$. Using calculus of variations:

- Under regularity conditions (smoothness, integrability), we can **interchange** the derivative and the integral:


$$
\nabla\_\theta \int e^{-E\_\theta(x)} dx = \int \nabla\_\theta e^{-E\_\theta(x)} dx
$$


- Now, differentiate the integrand:


$$
\nabla\_\theta e^{-E\_\theta(x)} = -e^{-E\_\theta(x)} \nabla\_\theta E\_\theta(x)
$$


- Therefore:


$$
\nabla\_\theta Z(\theta) = \int -e^{-E\_\theta(x)} \nabla\_\theta E\_\theta(x) dx
$$


- Plug this into the earlier equation:


$$
\nabla\_\theta \log Z(\theta) = \frac{1}{Z(\theta)} \int -e^{-E\_\theta(x)} \nabla\_\theta E\_\theta(x) dx
$$


- Recognize the integral as an expectation under the model distribution:


$$
\nabla\_\theta \log Z(\theta) = -\mathbb{E}\_{x \sim q\_\theta} \[\nabla\_\theta E\_\theta(x) ]
$$


So the final MLE gradient becomes:


$$
\nabla\_\theta \mathcal{L}(\theta) = \mathbb{E}\_{x \sim p\_{\text{data}}} \[-\nabla\_\theta E\_\theta(x) ] + \mathbb{E}\_{x \sim q\_\theta} \[\nabla\_\theta E\_\theta(x) ]
$$


This form clearly shows two competing forces:

* Lower energy for real data
* Increase energy (or "repel") for samples drawn from the model’s own distribution


### Sampling from $q\_\theta(x)$ is Hard

The second expectation, $\mathbb{E}\_{x \sim q\_\theta} \[\cdot ]$, requires sampling from the model distribution:


$$
q\_\theta(x) \propto e^{-E\_\theta(x)}
$$


But sampling from $q\_\theta(x)$ is intractable in most high-dimensional settings.


### Enter Contrastive Divergence (CD)

**Contrastive Divergence** approximates this sampling step using **Langevin dynamics** (a kind of stochastic gradient MCMC).

Instead of drawing perfect samples from $q\_\theta(x)$, we:

* Start from random noise (or real data, in CD-k)
* Run $k$ steps of Langevin dynamics. This acts like a noisy gradient descent on the energy surface:


$$
x\_{t+1} = x\_t - \alpha \nabla\_x E\_\theta(x\_t) + \eta\_t, \quad \eta\_t \sim \mathcal{N}(0, \epsilon)
$$


After a few steps, the resulting sample $x\_k$ is used as a proxy for $q\_\theta$. This allows us to compute the approximate gradient:


$$
\nabla\_\theta \mathcal{L}(\theta) \approx \mathbb{E}\_{x \sim p\_{\text{data}}} \[-\nabla\_\theta E\_\theta(x) ] + \mathbb{E}\_{x' \sim q\_\theta} \[\nabla\_\theta E\_\theta(x') ]
$$


Where $q\_\theta$ is the **approximate distribution** from the short-run MCMC.

This approximation is what we call **Contrastive Divergence** (specifically, CD-k for $k$ Langevin steps).


## Implementation Overview

Here’s the plan:

1. Define the negative energy function $-E\_\theta(x)$ as a CNN.
2. Train via **score matching**: minimizing the energy of real images while maximizing energy of synthetic samples (contrastive divergence).
3. Generate new images by Langevin sampling.
4. Store synthetic images in a replay buffer.

Please go to the [notebook](https://colab.research.google.com/github/reshalfahsi/dermatoscopic-debm/blob/main/Generating_Dermatoscopic_Images_With_the_Deep_Energy_Based_Model.ipynb) for the working implementation.

## Result

### Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/loss_plot.png" alt="loss_plot" > <br /> Loss curves of the Deep Energy-Based Model. Fortunately, the losses don't diverge. We must be aware that the model is susceptible to such a phenomenon. Once the event occurs, all we can do is pray to God, fine-tune the last best weight before divergence, and hope for the best. </p>

### Metric Curve

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/metrics_plot.png" alt="metrics_plot" > <br /> Metrics curves shows the expected energy of real and fake samples during training. </p>


### Qualitative Result

Here are the ten examples of generated dermatoscopic images:

#### Sample 1

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_0.png" alt="image_per_step_0" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_0.gif" width=400 alt="image_per_step_animation_0" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 2

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_1.png" alt="image_per_step_1" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_1.gif" width=400 alt="image_per_step_animation_1" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 3

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_2.png" alt="image_per_step_2" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_2.gif" width=400 alt="image_per_step_animation_2" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 4

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_3.png" alt="image_per_step_3" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_3.gif" width=400 alt="image_per_step_animation_3" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 5

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_4.png" alt="image_per_step_4" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_4.gif" width=400 alt="image_per_step_animation_4" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 6

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_5.png" alt="image_per_step_5" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_5.gif" width=400 alt="image_per_step_animation_5" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 7

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_6.png" alt="image_per_step_6" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_6.gif" width=400  alt="image_per_step_animation_6" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 8

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_7.png" alt="image_per_step_7" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_7.gif" width=400 alt="image_per_step_animation_7" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 9

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_8.png" alt="image_per_step_8" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_8.gif" width=400 alt="image_per_step_animation_8" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>

#### Sample 10

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_9.png" alt="image_per_step_9" > <br /> Transition from a random noise to a generated dermatoscopic image  </p>

<p align="center"> <img src="https://github.com/reshalfahsi/dermatoscopic-debm/blob/main/assets/image_per_step_animation_9.gif" width=400 alt="image_per_step_animation_9" > <br /> Animated transition from a random noise to a generated dermatoscopic image  </p>



## **References**

1. [Y. Du and I. Mordatch, "Implicit generation and modeling with energy based models," *Advances in Neural Information Processing Systems*, vol. 32, (2019).](https://arxiv.org/pdf/1903.08689)
2. [Tutorial 8: Deep Energy-Based Generative Models](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)
3. [J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, vol. 10, no. 1, p. 41, (2023).](https://www.nature.com/articles/s41597-022-01721-8)