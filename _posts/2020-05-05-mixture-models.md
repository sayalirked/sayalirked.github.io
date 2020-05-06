---
title: 'Mixture Models'
date: 2020-05-05
excerpt: The post explains the concept of **Mixture Models**. Click [here](https://sayrjked.github.io/posts/2020/05/random-processes/) to read further.
permalink: /posts/2020/05/mixture-models/

tags:
  - Machine Learning
  - Probability
  - Data Science
---

Latent variables are the random variables whose values are not specified in the observed data.
Modeling these latent variables is required to explained the observed data in terms of the unobserved concepts. Since these variables cannot be observed or measured experimentally.
Examples:

1. You fit a model to a dataset of scientific papers which tries to identify different topics in the papers, and for each paper, automatically list the topics that it covers. The model wouldn’t know what to call each topic, but you can attach labels manually by inspecting the frequent words in each topic. You then plot how often each topic is discussed in each year.
2. Reduce your energy consumption: you have a device which measures the total energy usage for your house (as a scalar value) for each hour over the course of a month. You want to decompose this signal into a sum of components which you can then try to match to various devices in your house (e.g. computer, refrigerator, washing machine), so that you can figure out which one is wasting the most electricity.

Let $$x$$ be observed/visible variables, $$z$$ be the latent/hidden variables.
We want to model $$p(x, z|\theta)$$, so marginalizing over $$z$$, we can write $$p(x|\theta) = \sum_z p(x, z|\theta)$$
For estimating unknown model parameters $$\theta$$, we can compute the Maximum Likelihood Estimate (MLE) on visible variables alone.

## Mixture models
===

When latent variables $$z$$ are discrete and observed variables $$x$$ are continuous, mixture modeling can be adopted to solve the problem.
It aims at maximizing the marginal likelihood of observed variables $$p(x) = \int_{z} p(x,z)$$.

- Type 1: Model assumptions: Discrete categorical latent variables $$z \in {1,2, \hdots, k}$$,univariate case: continuous observed variables $$D = \{ x_1, x_2, \hdots, x_n \}$$. The mean $$\mu$$ is different and $$\sigma$$ is same for each component of Gaussians, mixing weights are known.

Marginal probability $$p(z_i)$$:

$$p(z_i) = \prod_{c \in \{ 1,2, \hdots, k \} } \pi_c^{\mathbbm{1}  (z_i=c)}$$

Gaussian conditional probability $$p(x_i |z_i )$$:

$$p(x_i|z_i) = \prod_c \mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)}$$

probability density for one data point $$p(x_i)$$:

$$p(x_i) &=  \sum_{z_i} p(x_i|z_i)p(z_i) = \sum_{z_i} \prod_c \mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} \pi_c^{\mathbbm{1}  (z_i=c)}$$

$$
\begin{eqnarray*}
&\text{Joint density or likelihood for } D = \{ x_1, x_2, \hdots, x_n \}\\
&L = p(D) = \prod_{i=1}^n \mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} \pi_c^{\mathbbm{1}  (z_i=c)}\\
&\text{Log-likelihood}\\
&l = \sum_{i=1}^n \log (\mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} \pi_c^{\mathbbm{1}  (z_i=c)})\\
\end{eqnarray*}
$$


Known variables: observed variables $${x}$$, mixing weights $$p(z_i=1), p(z_i=2), \hdots, p(z_i=k)$$;

Unknown variables: latent variables $$\mu_c, \sigma, c={1,\hdots,k}$$

An elegant and powerful approach E-M algorithm, particularly for latent variables, can be used.

**E Step**: fix parameters $$\mu_{a}, \mu_{b}, \mu_{c},\sigma_{a}, \sigma_{b},\sigma_{c}$$ and compute posterior distribution $$p(z_i = a|x_i ), p(z_i = b|x_i ), p(z_i = c|x_i )$$

**M Step**: fix the posterior distribution $$p(z_i = a|x_i ), p(z_i = b|x_i ), p(z_i = c|x_i )$$ and optimize for $$\mu_{a}, \mu_{b}, \mu_{c}, \sigma_{a}, \sigma_{b},\sigma_{c}$$.

<!-- - Type 2: -->
