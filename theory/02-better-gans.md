# Course 2 - Better undersdanding of GANs

## 1. Inception-v3 Architecture

> 🏃 Commonly used as a feature extractor. It is a huge classifier model trained on ImageNet. Thanks to it we can extract meaningful features and compare images with them.

<!-- TODO: Why do we even talk about this? -->

![Feature Extractor Scheme](image-79.png)

> **Fun fact from Wikipedia:** [The original name (Inception)](https://en.wikipedia.org/wiki/Inceptionv3) was codenamed this way after a popular "'we need to go deeper' internet meme" went viral, quoting a phrase from the Inception film of Christopher Nolan.
>
> ![Inception Meme](image-80.png)

<!-- | Big Kernel Size | Small Kernel Size               | When desired? |
| --------------- | ------------------------------- | ------------- |
|                 | Information distributed locally |               | -->

> To use Inception Architecture as a **feature extractor we cut off the classification layer**. In `pytorch` this can be done with the following code:
>
> #### `inception_model.fc = torch.nn.Identity()`
>
> Where `fc` represents the last fully connected layer (classification layer that outputs propabilities per class) and [`nn.Identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html) is a placeholder that returns the input data without any changes.

### Resources related to this topic

- [Papers with code](https://paperswithcode.com/method/inception-v3)
- Towards Data Science ["A Simple Guide to the Versions of the Inception Network"](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) by Bharath Raj. It explains what Inception Model familly is, what problem it solved and goes trought the 3 versions.

## 2. Fréchet Inception Distance (FID)

> 🏃 FID measures the distance between two distributions.

<p align="center">
  <img src="image-81.png" alt="Image">
</p>

$$d(X,Y) = (\mu_X-\mu_Y)^2 + (\sigma_X-\sigma_Y)^2 $$

> The lower the better. A high FID means a lack of diversity or fidelity.

> Introduced in 2017 and has generally superseded inception score as the preferred measure of generative image model performance.

![Alt text](image-1.png)
![Alt text](image-2.png)
![Alt text](image-3.png)
Higher variance = bigger area of the circle
![Alt text](image-4.png)
![Alt text](image-6.png)

- [Kaggle Notebook about FID](https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid)
- [Target Tech](https://www.techtarget.com/searchenterpriseai/definition/Frechet-inception-distance-FID)

### Covariance

![Alt text](image-20.png)

> 🏃 Its a measure of variance of 2 variables.

| Number of variables | Name           |
| ------------------- | -------------- |
| One                 | Variance       |
| Two                 | **Co**variance |
| More                | ???            |

- If **zeros are everywhere except the diagonal** it means the dimensionc are **intependent**.
- If values **other than zeros** are on the **diagonal** it means the dimensions **covary**.

- [Covariance explanation on YouTube](https://www.youtube.com/watch?v=TPcAnExkWwQ) by Normalized Nerd.
- [Covariance explanation on YouTube](https://www.youtube.com/watch?v=qtaqvPAeEJY) by StatQuest.
- [Covariance and Correleation](https://ai-ml-analytics.com/covariance-and-correlation/) explanation by ai-ml-analytics.com

### Multivariate Normal Fréchet Distance

> 🏃 FID is used to **calculate the distance** between real and generated images.

![Alt text](image-7.png)

##### Formula for Univariate Normal Fréchet Distance

$$(\mu x - \mu y)^2 + (\sigma x - \sigma y)^2$$

##### Formula for Multivariate Normal Fréchet Distance

$$\lVert \mu x - \mu y  \rVert_2 + Tr(\Sigma x + \Sigma y - 2 \sqrt{\Sigma x \Sigma y})$$
$\lVert \mu x - \mu y  \rVert_2 $ - Magnitude of the vector.
$Tr$ - trace of a matrix (sum of its diagonal elements of a matrix).
$\Sigma$ - Covariance.
$\mu x$ - mean of the real embeddings.
$\mu y$ - mean of the fake embeddings.
$\Sigma x$ - Covariance matrix of the real embeddings.
$\Sigma y$ - Covariance matrix of the fake embeddings.

Quick notes:

- Real and fake embeddings are two multivariate normal distributions.
- Lower FID = closer distributions (the lower the better)
  There is no strict range for FID. But usually the closer to zero the better.
- Use large sample size to reduce noise

### Problems with FID

- Not all features are captured by the pre-trained inception model.
- Not fast.
- Mean and covariance (first two moments of a distribution) don't cover all aspects of a distribution.
- A large dataset (smaple size) is needed to work properly.
  ![Alt text](image-8.png)

![Alt text](image-9.png)

> **How does FID measure the difference between reals and fakes?**
> By considering the real and fake embeddings as two distributions and comparing the statistics of between those distributions.

### Inception Model Classification

![Alt text](image-10.png)
$p(y | x)$ - measures the distribution $p$ of $y$ given $x$. S
where given your image $x$, what is the distribution over those classes given that classifier?
$P(y)$ - The **marginal label distribution** or $p$ of $y$. It's a distribution over all labels across your entire dataset or across a large sample

### 3. KL Divergence (Relative Entropy)

> 🏃 KL divergence measures difference in information represented by two distributions.

> Measures **how much information** you can get or gain on $p$ of $y$ for a particular $x$, given just $p$ of $y$.
> ![Alt text](image-11.png)

you can think of KL Divergence here as approximately getting how different the conditional label distribution for fidelity is, from the marginal label distribution for diversity, which is their relative entropy from one another.

> ❓ **Why KL Divergence is not called KL Distance?**
> In the opposite direction, the KL divergence is not the same. However, the distances are always the same regardless of direction.

You want one to be high, you want the other to be low. In other words, basically, when fake images each have a distinct label and there's also a diverse range of labels among those fakes.

- You Tube explanation [of divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY) by ritvikmath.
- [Understanding KL Divergence](https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254) by Aparna Dhinakaran.

## 4. Inception Score

> 🏃 Measures the quality and diversity of generated images. The higher the IS the better.

![Alt text](image-12.png)
$$IS  = exp(\mathbb{E}_{x \sim pg} D_{KL}(p(y | x ) || p(y)))$$

- $D_{KL}(p(y | x ) || p(y)))$ - KL Divergence.
- $\mathbb{E}_{x \sim pg}$ - sum and average of all results.
- $p(y|x)$ - conditional probability distribution.
- $p(y)$ - marginal probability distribution.

|           | Lowest possible value | Highest (best) possible value                                                                                               |
| --------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Math      | $$0$$                 | $$\infty$$                                                                                                                  |
| Real Life | $$1$$                 | **Number of classes** (in the case of Inception-v3 it is 1000, beacause it was trained on ImageNet which has 1000 classes). |

- More about Inception Score in this [paper ,,A Note on the Inception Score"](https://arxiv.org/abs/1801.01973) by Shane Barratt, Rishi Sharma.
- Explanation of [Inception Score](https://www.techtarget.com/searchenterpriseai/definition/inception-score-IS) by Stephen J. Bigelow. It also esplains in a nice way **what a probablility distribution is**.

> **Determining the final inception score.**
>
> 1. Process generated images through discriminator to obtain the conditional probability distribution or $p(y|x)$.
> 2. Calculate the marginal probability distribution or $p(y)$.
> 3. Calculate the KL divergence (between $p(y)$ and $p(y|x)$).
> 4. Calculate the sum for classes and calculate an average score for all images (basically repeat the previous steps for all images in the computer-generated set or sample).
> 5. Calculate the average value of all results $\mathbb{E}_{x \sim pg}$ and take its exponent (exp).
>
> This final result is the inception score for the given set of generated (fake) images.

### Problems with Inceprion Score

- Do not cover all useful features (ImageNet is not a representative dataset for every case).
- Can be exploited or gamed. Generate only one realistic image of each class.
- No comparison to real images (makes it only looks at fake images).
- Works with small square images (about 300 x 300 pixels).

### Inception Score vs. Fréchet Inception Distance

| Feature | IS                                 | FID                                |
| ------- | ---------------------------------- | ---------------------------------- |
|         | Evaluates **only on fake** images. | Evaluates on real and fake images. |

Need a summary of FID and IS? Here are two great articles that recap both metrics!

Fréchet Inception Distance (Jean, 2018):
https://nealjean.com/ml/frechet-inception-distance/

GAN — How to measure GAN performance? (Hui, 2018):
https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732

## 5. Sampling and Truncation

### Truncation Trick

![Alt text](image-77.png)
![Alt text](image-15.png)

> **How does truncation trick sampling work, and when do you use it?**
> By sampling at test time from a normal distribution with its tails clipped.

> Notebook C2W3
> ![Alt text](image-76.png)

https://paperswithcode.com/method/truncation-trick

### HYPE

- Intrigued about human evaluation and HYPE (Human eYe Perceptual Evaluation) [paper](https://arxiv.org/abs/1904.01121)

## 6. Precision & Recall (in context of GANs)

- paper [Assessing Generative Models via Precision and Recall](https://proceedings.neurips.cc/paper_files/paper/2018/file/f7696a9b362ac5a51c3dc8f098b73923-Paper.pdf)
  https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732

![PRECISION](image-16.png)
![RECALL](image-17.png)
![SUMMARY](image-18.png)

What is recall related to in the context of GANs evaluation, and what is it a measure of?Recall is related to diversity and measures of how well the generator models all the variation in the reals. Recall measures the intersection over all the reals.Recall is how much overlap there is over all the reals. It is related to diversity because you can see if the generator models all the variation in the reals or not.

- [Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi, Karras, Laine, Lehtinen, and Aila, 2019)](https://arxiv.org/abs/1904.06991)

## Disadvantages of GANs

- Unstable training (Mode Collapse). Now it is sovled thanks to Wasserstein GANs.
- Lack of intristic evaluation metric.
- No density estimation.
- Incerting is not straightforward (it is hard to get the noise vector from the image). Inversion can be usefull for image edition.

## 7. Alternatives for GANs - Variational Autoencoders

Generator ---> Decoder
Discriminator ---> Encoder
![VAE](image-23.png)
![Decoder and Encoder](image-24.png)
![VAE Instruction](image-25.png)
![Evidence Lower Bound](image-26.png)

| Advantages          | Disadvantages             |
| ------------------- | ------------------------- |
| Density estimation. | Lower quality of results. |
| Invertible.         |                           |
| Stable traning.     |                           |

> **What is the core difference between GANs and VAEs?**
> VAEs use autoencoders to encode a real image and then decode them while GAN generators never see the real image in any form and only take the noise as input.

### Evidence Lower Bound (ELBO).

![Alt text](image-27.png)

Knowing that notation, here's the mathematical notation for the ELBO of a VAE, which is the lower bound you want to maximize: $\mathbb{E}\left(\log p(x|z)\right) + \mathbb{E}\left(\log \frac{p(z)}{q(z)}\right)$, which is equivalent to $\mathbb{E}\left(\log p(x|z)\right) - \mathrm{D_{KL}}(q(z|x)\Vert p(z))$

ELBO can be broken down into two parts: the reconstruction loss $\mathbb{E}\left(\log p(x|z)\right)$ and the KL divergence term $\mathrm{D_{KL}}(q(z|x)\Vert p(z))$. You'll explore each of these two terms in the next code and text sections.

### Reconstruction Loss

> Reconstruction loss refers to the distance between the real input image (that you put into the encoder) and the generated image (that comes out of the decoder). Explicitly, the reconstruction loss term is $\mathbb{E}\left(\log p(x|z)\right)$, the log probability of the true image given the latent value.

![Alt text](image-28.png)

- Bernouli Distribution Prepared by: [Reza Bagheri](https://www.linkedin.com/in/reza-bagheri-71882a76/)
  ![Alt text](image-29.png)

![Alt text](image-30.png)

### Coursera suggested resources for VAEs

- [$\beta$-VAEs](https://openreview.net/forum?id=Sy2fzU9gl) showed that you can weight the KL-divergence term differently to reward "exploration" by the model.
- [VQ-VAE-2](https://arxiv.org/pdf/1906.00446.pdf) is a VAE-Autoregressive hybrid generative model, and has been ablbe to generate incredibly diverse images - keeping up with GANs. :)
- [VAE-GAN](https://arxiv.org/abs/1512.09300) is a VAE-GAN hybrid generative model that uses an adversarial loss (that is, the discriminator's judgments on real/fake) on a VAE.

### More resources

- [vae](https://deepgenerativemodels.github.io/notes/vae/)
- [lecture](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec17.pdf)
- [The new contender to GANs: score matching with Langevin Sampling](https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/)

### Summary

![SUMMARY](image-22.png)

## 8. Bias

A Survey on Bias and Fairness in Machine Learning (Mehrabi, Morstatter, Saxena, Lerman, and Galstyan, 2019)
https://arxiv.org/abs/1908.09635

Does Object Recognition Work for Everyone? (DeVries, Misra, Wang, and van der Maaten, 2019):
https://arxiv.org/abs/1906.02659

![bias](image-31.png)
![evaluation bias](image-32.png)

## 9. GANs improvements over past years

![Alt text](image-40.png)

### Stability

![stability text](image-34.png)
![gradient](image-35.png)
![Alt text](image-36.png)
![Alt text](image-37.png)

### Capacity

![Alt text](image-38.png)

### Diversity

![Alt text](image-39.png)

## 10. StyleGAN

### Overview

**Goal:** Improved control over generation. Better quality and increased diversity.
**Paper:** [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
**Other papers:**
[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).
Medium [GAN — StyleGAN & StyleGAN2](https://jonathan-hui.medium.com/gan-stylegan-stylegan2-479bdf256299)
**StyleGAN consists of:**

- Noise Mapping Network
- Progressive Growing
- AdaIN (Adaptive Instance Normalization)
- Style Mixing
- Stochastic Noise

### Generator in StyleGAN

> **Intermediate noise vector $w$** is multiple times injected (AdaIN operation) into the StyleGAN generator.

![Alt text](image-74.png)

### Progressive Growing

> 🏃 Ensures stable training of generator models that can output large high-quality images. art with an small image (generating a small image is an easier task).
> ![Alt text](image-44.png)

During the training we **incrementally add layers** to Generator and Discriminator. This results in increase of the spatial resolution of the generated images. All existing layers remain **trainable** throughout the process.
![Alt text](image-75.png)
**Advantages of Progressive Growing:**

- Reduced training time (most of the iterations are computed with lower resolutions).
- Ability to generate high resolution images.
- Usage only of a single GAN (compared to other approaches).

![Alt text](image-48.png)
![Alt text](image-49.png)
The same alfa apameter in generator and discriminator that decides if rellay on learned parameters.
![Generator Alfa](image-51.png)
![Alt text](image-50.png)
![Alt text](image-52.png)

What is the purpose of progressive growing?
To gradually train the generator by iteratively increasing the resolution of images being generated.

![summary](image-53.png)

### Noise Mapping Network

Thanks to Noise Mapping Network the $z$ vector can be represented in a more disentangled space which enables easier feature control.

![Alt text](image-54.png)
![Alt text](image-55.png)
![Z-Space Entanglment](image-56.png)
![W-Space Less Entangled](image-57.png)
![Alt text](image-58.png)
![Summary](image-59.png)
Why map $z$ vector to $w$ vector?
In order to learn factors of variation and increase control over features in a more disentangled space (where changing one fature does not impact change in other features).

### Adaptive Instance Normalization AdaIN (Random Noise Injection)

#### $ \text{AdaIN}(\boldsymbol{\mathrm{x}}_i, \boldsymbol{\mathrm{y}}) = \boldsymbol{\mathrm{y}}_{s,i} \frac{\boldsymbol{\mathrm{x}}_i - \mu(\boldsymbol{\mathrm{x}}\_i)}{\sigma(\boldsymbol{\mathrm{x}}\_i)} + \boldsymbol{\mathrm{y}}_{b,i} $

_The noise tensor is not entirely random; it is initialized as one random channel that is then multiplied by learned weights for each channel in the image._
![Alt text](image-78.png)

![Outline](image-60.png)
![scheme](image-61.png)
![Alt text](image-62.png)

##### Batch Norm vs. Instance Norm

$\frac{x - \mu(x_i)}{\sigma(x_i)}$
$x_i$ - every value from instance i.

![Batch Norm vs. Instance Norm](image-63.png)

![Alt text](image-64.png)
![scale and bias (shift)](image-65.png)
![Alt text](image-66.png)

> ![step 2](image-67.png)
> So you can think of it as an image that has some kind of content and with different ys and yb values. It will be like Picasso drew it, or like Monet drew that image, but **it will have the same content**, right? It'll still be a face or a puppy field, but it would be a different style. So shifting your values to different ranges means standard deviations will actually get you those different styles

![summary](image-68.png)
What is the purpose of the AdaIN layers?
AdaIN blocks transfer learned style info from the intermediate noise vector $w$ onto the generated image. They also renormalize the statistics so each block overrides the one that came before it.

### Style Mixing

Mixing two different noise vectors that will be inputet into the model. The later the switch, the finer the features that you'll get from $w_2$. This approach improvet the diversity of the results.
![Alt text](image-69.png)

### Stochastic Variation

Adding additional noise to the image.
![schema](image-71.png)
$\lambda$'s are learnend values!
![summary](image-72.png)

> **Why is random noise added throughout StyleGAN?**
> To introduce more randomness into the feature map and increase diversity.Extra noise is injected at several different levels of StyleGAN, and affects the generated image in a different way depending on whether the noise was injected earlier or later.

## 11. Perceptual Path Length (PPL)

> 🏃 Metric that evaluates **how well a generator manages to smoothly interpolate between points in its latent space.**

#### $$PPL_{w} = \mathbb{E}\left[\frac{1}{\epsilon^2} \mathop{d_{\mathrm{LPIPS}}}\left(\mathop{\mathrm{G}}\left(\mathrm{lerp}(w_1, w_2, t\right), \mathop{\mathrm{G}}\left(\mathrm{lerp}(w_1, w_2, t + \epsilon\right)\right)\right]$$

- $\mathbb{E}$ - expect. The symbol is there because the equation is calculated several times in order to estimate PPL.

**Bullet points:**

- Introduced as part of StyleGAN

### Interpolation

> 🏃 Process of smoothly **transitioning** between two different data points in the latent space to create intermediate samples.

#### $$z_{interpolated} = (1 - \alpha) z_1 + \alpha z_2$$

Interpolation is used for:

- Style Mixing
- Data Augmentation
- Exploring the latent space

## 12.

# Papers

| Paper Name                                                         | Description                                                                         | Tags | Year | Code                                                      |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ---- | ---- | --------------------------------------------------------- |
| [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) | First GAN paper. Proposed training two adversarial models in the generation proces. | GAN  | 2014 | [Papers with code](https://paperswithcode.com/method/gan) |

# Glossary

> Only short definitions. The details are explained above.

- Probability Distribution - a numbered list of what the classifier predicts an image could be. All the numbers add to $1$.
- Fidelity / Quality -
- Diversity
- Feature Extractor
- Inception-v3 Architecture - commonly used as a feature extractor. Pretrained on ImageNet.
- Auxiliary Classifier
- Inception Layer
- Feature Layer
- Pixel Distance
- Feature Distance
- Embeddings - Used to calculate the feature distance.
- Fréchet Distance - metric measuring the distance between curves (can be extended to comparing distributions).
- Fréchet Inception Distance
- Multivariate Normal Distribution
- Variance - measures spread in a distribution
- Covariance - measures the spread between two dimensions.
- Covariance matrix
- Latent space - lower-dimensional representation that GANs use to generate stuff.
- latent space distribution prior 𝑝(𝑧) - distribution for the latent space with a mean of 0 and a standard deviation of 1 in each dimension $\mathcal{N}(0, I)$.
- Interpolation - process of smoothly transitioning between two different data points in the latent space to create intermediate samples.

<!-- # TODO: Table with metric - which is better high or low etc. -->

# Math

- $\phi (x)$ - Embedding of x.
  ![Alt text](image.png)
