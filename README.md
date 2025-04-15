# Denoising Diffusion Implicit Models (DDIM)

This repository contains an implementation of Denoising Diffusion Implicit Models (DDIM) for image generation and inversion tasks, however it's edited so that it is compatible with the rectified flow model. Denoising diffusion implicit models (DDIMs) are a generalization of DDPMs via a class of non-Markovian diffusion processes that still lead to the same training objective as DDPMs. These non-Markovian processes can correspond to a deterministic generative process. DDIMs allow us to perform semantic image interpolation directly in the latent space and reconstruct observations with very low error. In addition to that, DDIMs provide a much faster sampling time than DDPMs.

### Introduction

Deep generative models have demonstrated the ability to sample high-quality samples from unknown distributions. In terms of image generation, DDPMs have shown results that are comparable to those of GANs. However, GANs require very specific choices in optimization and architecture to stabilize training, and they could also fail to cover modes of the data distributions. This opened the way to a new class of generative models where we train a neural network to learn how to denoise an image that has been progressively corrupted by Gaussian noise through a forward process that simulates a [[Markov chain]]. The samples are then generated through a Markov chain which starts from white noise, progressively denoising it into an image.

### Background

Given samples from a data distribution $q(x_0)$, we are interested in learning a model distribution $p_\theta(x_0)$ that approximates $q(x_0)$ and is easy to sample from.

DDPMs are latent variable models that have the following form:

$$p_\theta(x_0) = \int p_\theta(x_{0:T})dx_{1:T} \text{, where } p_\theta(x_{0:T}) := \prod_{t=1}^{T} p_\theta^t(x_{t-1}|x_t)$$

Where the parameters $\theta$ are learned to fit the data distribution $q(x_0)$. $x_1, ..., x_T$ are the latent variables in the same sample space as $x_0$.
In this process, we marginalize the joint distribution of the entire latent states to get the approximate distribution $p_\theta$.
We achieve that by maximizing the variational lower bound ([[ELBO]]):

$$\max_\theta E_{q(x_0)}[\log p_\theta(x_0)] \le \max_\theta E_q(x_0, x_1, ..., x_T)[\log p_\theta(x_{0:T}) - \log q(x_{1:T}|x_0)]$$

Where $q(x_{1:T}|x_0)$ is some inference distribution over the latent variables. Unlike VAEs, the latent variables in DDPMs are learned with a fixed inference procedure $q(x_{1:T}|x_0)$ rather than a trainable one.
For example, in the DDPMs paper, we can consider the following Markov chain with Gaussian transitions parameterized by a decreasing sequence $\alpha_{1:T} \in (0, 1]^T$:

$$q(x_{1:T}|x_0):=\prod_{t=1}^{T} q(x_t|x_{t-1}), \text{ where } q(x_t|x_{t-1}):= \mathcal{N}(\sqrt{\frac{\alpha_t}{\alpha_{t-1}}}x_{t-1}, (1 - (\frac{\alpha_t}{\alpha_{t-1}}))I)$$

Where the covariance matrix is ensured to have positive terms on the diagonal. This is called the forward process due to the auto-regressive nature of the sampling procedure ($x_0$ to $x_T$).
The latent variable model $p_\theta(x_{0:T})$ is a Markov chain that samples from $x_T$ to $x_0$, the generative process. It approximates the intractable reverse process $q(x_{t-1}|x_t)$.

A special property of the forward process is that:

$$q(x_t|x_0) = \int q(x_{1:t}|x_t)dx_{1:t-1} = \mathcal{N}(x_t; \sqrt{\alpha_t}x_0, (1-\alpha_t)I )$$

This allows us to express $x_t$ as a linear combination of $x_0$ and noise $\epsilon$:

$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, I)$$

When we set $\alpha_T$ sufficiently close to $0$, $q(x_T|x_0)$ converges to the standard normal distribution for all $x_0$. If all the conditionals are modeled as Gaussians with trainable mean functions and fixed variance, we can simplify the objective to:

$$L_\gamma(\epsilon_\theta) := \sum_{t=1} ^T E_{x_0\sim q(x_0), \epsilon_t \sim \mathcal{N}(0, I)}[||\epsilon_\theta^t(x_t) - \epsilon_t||_2^2]$$

where $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon_t$, and $\epsilon_\theta := \{\epsilon_\theta^{(t)}\}_{t=1}^T$ is a set of $T$ functions, each $\epsilon_\theta^{(t)}: \mathcal{X} \rightarrow \mathcal{X}$ is a function with trainable parameters $\theta^{(t)}$, and $\gamma := [\gamma_1, ..., \gamma_T]$ is a vector of positive coefficients in the objective that are dependent on $\alpha_{1:T}$. In DDPMs, $\gamma = \mathbb{1}$.

The length $T$ of the forward process is an important hyperparameter in DDPMs. A large $T$ allows the reverse process to be close to a Gaussian, so that the generative process modeled after Gaussian conditional distributions becomes a good approximation. This motivates using large $T$ values. However, all $T$ iterations have to be performed sequentially to obtain $x_0$.

### [[Variational Inference]] for Non-Markovian Forward Processes

The generative process is an approximation of the reverse of the inference process (diffusion process). So, if we use different inference processes that are not Markovian, we can reduce the number of iterations required by the generative model.

One important observation regarding the objective $L_\gamma$ is that it depends only on the marginals $q(x_t|x_0)$, but not directly on the joints $q(x_{1:T}|x_0)$. This means that there are many inference distributions (joints) that could result in the same marginals. By exploiting this fact, we could explore different inference processes that are non-Markovian, which could lead to new generative processes.

These non-Markovian inference processes lead to the same objective function as DDPMs.

##### Non-Markovian Forward Processes

Let's consider a family $\mathcal{Q}$ of inference distributions, indexed by a real vector $\sigma \in \mathbb{R}^T_{>0}$:

$$q_\sigma(x_{1:T}|x_0) := q_\sigma(x_T|x_0)\prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$$

Where $q_\sigma(X_T|x_0) = \mathcal{N}(\sqrt{\alpha_T}x_0, (1-\alpha_T)I)$ for $T$, and for all $t > 1$:

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}}x_0 + \sqrt{1 - \alpha_{t-1} - \sigma^2_t} \cdot \frac{x_t-\sqrt{\alpha_t }x_0}{\sqrt{1-\alpha_t}}, \sigma^2_t I)$$

The mean function is chosen to ensure that $q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I) \space \forall t$, so that it defines an inference joint distribution that matches the marginals as desired.
Then the forward process can be derived from Bayes' rule:

$$q_\sigma(x_t|x_{t-1}, x_0) = \frac{q_\sigma(x_{t-1}|x_t, x_0)q_\sigma(x_t|x_0)}{q_\sigma(x_{t-1}|x_0)}$$

which is also Gaussian, but unlike the diffusion process, the forward processes in this case are no longer Markovian, since each $x_t$ could depend on both $x_{t-1}$ and $x_0$. The magnitude of $\sigma_t$ controls how stochastic the forward process is; when $\sigma \rightarrow 0$, we reach an extreme case where, as long as we observe $x_0$ and $x_t$ for some time $t$, then $x_{t-1}$ becomes known and fixed.

##### Generative Process and Unified Variational Inference Objective

Defining a trainable generative process $p_\theta(x_{0:T})$ where each $p_\theta^{(t)}(x_{t-1}|x_t)$ leverages knowledge of $q_\sigma(x_{t-1}|x_t, x_0)$. Intuitively, given a noisy observation $x_t$, we first make a prediction of the corresponding $x_0$, and then use it to obtain a sample $x_{t-1}$ through the reverse conditional distribution $q_\sigma(x_{t-1}|x_t,x_0)$, which we have defined.

For some $x_0 \sim q(x_0)$ and $\epsilon_t \sim \mathcal{N}(0, I)$, $x_t$ can be obtained using $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t} \epsilon_t$. We then attempt to predict $\epsilon_t$ from $x_t$ without knowledge of $x_0$.

By rewriting the equation, we can then predict the denoised observation, which is a prediction of $x_0$ given $x_t$:

$$\mathcal{f_\theta^{t}}(x_t) := (x_t - \sqrt{1-\alpha_t}\epsilon_{\theta}^{(t)}(x_t))/\sqrt{\alpha_t}$$

Then we can define the generative process with a fixed prior $p_\theta(x_T) = \mathcal{N}(0,I)$ and

$$p_\theta^{(t)}(x_{t-1}|x_t) = \begin{cases} \mathcal{N}(f_\theta^1(x_1),\sigma_1^2I) & \quad \text{if } t = 1\\
q_{\sigma}(x_{t-1}|x_t, f_\theta^{(t)}(x_t)) & \quad \text{otherwise}\\
\end{cases}$$

where $q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}}x_0 + \sqrt{1 - \alpha_{t-1} - \sigma^2_t} \cdot \frac{x_t-\sqrt{\alpha_t }x_0}{\sqrt{1-\alpha_t}}, \sigma^2_t I)$ with $x_0$ replaced by $f_\theta^{(t)}(x_t)$. In the case of $t=1$, to ensure that the generative process is supported everywhere, we add some Gaussian noise with covariance $\sigma_1^2I$.

We optimize for $\theta$ with the following variational inference objective (which is a function over $\epsilon_\theta$):

$$J_\sigma(\epsilon_\theta) := \mathbb{E}_{x_{0:T}\sim q_\sigma(x_{0:T})} [\log q_\sigma(x_{1:T}|x_0) - \log p_\theta(x_{0:T})]$$

$$=\mathbb{E}_{x_{0:T}\sim q_\sigma(x_{0:T})} [\log q_\sigma(x_T|x_0) + \sum_{t=2}^T \log q_\sigma(x_{t-1}|x_t, x_0) - \sum_{t=1}^T \log p_\theta^{(t)}(x_{t-1}|x_t) -\log p_\theta(x_T) ]$$

We get the second formula by factorizing $q_\sigma(x_{1:T}|x_0)$ according to

$$q_\sigma(x_{1:T}|x_0) := q_\sigma(x_T|x_0)\prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$$

and $p_\theta(x_{0:T})$ according to

$$p_\theta(x_{0:T}) := \prod_{t=1}^{T} p_\theta^t(x_{t-1}|x_t)$$

From the definition of $J_\theta$, it would appear that a different model has to be trained for every choice of $\sigma$, since it corresponds to a different variational objective and different variational objective.
However, $J_\theta$ is equivalent to $L_\gamma$ for certain weights $\gamma$. (Theorem 1)

#### Sampling for Generalized Generative Processes

With $L_1$-the objective function of DDPMs- we not only optimize for a Markovian generative process, but also a generative process for many other non-Markovian processes parameterized by $\sigma$. That is, we can use pretrained DDPMs as the solution to $J_\sigma$ and focus on finding a generative process that is better at producing samples subject to our needs by changing $\sigma$.

From $p_\theta(x_{1:T})$ (trained model), we can generate a sample $x_{t-1}$ from a sample $x_t$ via:

$$x_{t-1}  = \sqrt{\alpha_{t-1}} \left(\frac{x_t - \sqrt{1 - \alpha_t }\epsilon_{\theta}^{(t)}(x_t)}{\sqrt{\alpha_t}}\right) + \sqrt{1-\alpha_{t-1} - \sigma_t^{2}}\cdot\epsilon_\theta^{(t)}(x_t) + \sigma_t\epsilon_t$$
where $\epsilon_t \sim \mathcal{N}(0,I)$ is standard Gaussian noise independent of $x_t$, and we define $\alpha_0 :=1$.
Different choices of $\sigma$ values result in different generative processes, while using the same model $\epsilon_\theta$. When $\sigma_t = \sqrt{(1-\alpha_{t-1})/(1-\alpha_t)}\sqrt{1 - \alpha_t/\alpha_{t-1}}$ for all $t$, the forward process becomes Markovian, and the generative process becomes a DDPM.

Another special case is when $\sigma_t = 0$ for all $t$ (note this case is not covered in Theorem 1, but it can be practically covered by making $\sigma$ close to $0$): The forward process becomes deterministic given $x_{t-1}$ and $x_0$ except for $t = 1$; the resulting model becomes an implicit probabilistic model, where the samples are generated from the latent variable with a fixed procedure (from $x_T$ to $x_0$).

## Acknowledgments

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
