---
title: 'Mixture Models'
date: 2020-05-05
excerpt: The post explains the concept of **Mixture Models**. Click [here](https://sayalirked.github.io/posts/2020/05/mixture-models/) to read further.
permalink: /posts/2020/05/mixture-models/

tags:
  - Machine Learning
  - Probability
  - Data Science
---

## Latent variables

Latent variables are the random variables whose values are not specified in the observed data.
Modeling these latent variables is required to explained the observed data in terms of the unobserved concepts. Since these variables cannot be observed or measured experimentally.
Examples:

1. You fit a model to a dataset of scientific papers which tries to identify different topics in the papers, and for each paper, automatically list the topics that it covers. The model wouldn’t know what to call each topic, but you can attach labels manually by inspecting the frequent words in each topic. You then plot how often each topic is discussed in each year.
2. Reduce your energy consumption: you have a device which measures the total energy usage for your house (as a scalar value) for each hour over the course of a month. You want to decompose this signal into a sum of components which you can then try to match to various devices in your house (e.g. computer, refrigerator, washing machine), so that you can figure out which one is wasting the most electricity.

Let $$x$$ be observed/visible variables, $$z$$ be the latent/hidden variables.
We want to model $$p(x, z|\theta)$$, so marginalizing over $$z$$, we can write $$p(x|\theta) = \sum_z p(x, z|\theta)$$
For estimating unknown model parameters $$\theta$$, we can compute the Maximum Likelihood Estimate (MLE) on visible variables alone.

## Mixture models

When latent variables $$z$$ are discrete and observed variables $$x$$ are continuous, mixture modeling can be adopted to solve the problem.
It aims at maximizing the marginal likelihood of observed variables $$p(x) = \int_{z} p(x,z)$$.

- **Type 1**: Model assumptions: Discrete categorical latent variables $$ z \in \{1,2, ..., k\} $$, univariate case: continuous observed variables $$ D = \{ x_1, x_2, ..., x_n \} $$. The mean $$\mu$$ is different and $$\sigma$$ is same for each component of Gaussians, mixing weights are known.

Marginal probability $$p(z_i)$$:

<!-- p(z_i) = \prod_{c \in \{ 1,2, \hdots, k \} } \pi_c^{\mathbbm{1}  (z_i=c)} -->
<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-pzi.gif" alt="eqn-pzi"/></p>

Gaussian conditional probability:

<!-- p({x_i}|{z_i}) = \prod_{c} \mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} -->
<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-pxizi.gif" alt="eqn-pxizi"/></p>

Probability density for one data point $$p(x_i)$$:

<!-- p(x_i) &=  \sum_{z_i} p(x_i|z_i)p(z_i) = \sum_{z_i} \prod_c \mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} \pi_c^{\mathbbm{1}  (z_i=c)} -->

<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-pxi.gif" alt="eqn-pxi"/></p>

Joint density or likelihood for  $$D = \{ x_1, x_2, .., x_n \}$$:
<!-- \begin{align*}
L = p(D) &= \prod_{i=1}^n \mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} \pi_c^{\mathbbm{1}  (z_i=c)}\\
\text{Log-likelihood}\\
l &= \sum_{i=1}^n \log (\mathcal{N}(x_i; \mu_c, \sigma^2)^{\mathbbm{1}  (z_i=c)} \pi_c^{\mathbbm{1}  (z_i=c)})\\
\end{align*} -->

<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-jointloglikeli1.gif" alt="eqn-jointloglikeli1"/></p>



Known variables: observed variables $${x}$$, mixing weights $$p(z_i=1), p(z_i=2), .., p(z_i=k)$$;

Unknown variables: latent variables $$\mu_c, \sigma, c=\{1,..,k\}$$

## Method: Expectation-Maximization algorithm

An elegant and powerful approach E-M algorithm, particularly for latent variables, can be used.

**E Step**:
- fix parameters $$\mu_{a}, \mu_{b}, \mu_{c},\sigma_{a}, \sigma_{b},\sigma_{c}$$ and
- compute posterior distribution
<!-- p({z_i = a}|x_i ), p({z_i = b}|x_i ), p({z_i = c}|x_i ) -->

<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-esteppost1.gif" alt="eqn-esteppost1"/></p>

**M Step**:
- fix the posterior distribution
<!-- p({z_i = a}|x_i ), p({z_i = b}|x_i ), p({z_i = c}|x_i ) \text{ and} -->
<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-msteppost1.gif" alt="eqn-msteppost1"/></p>

- optimize for $$\mu_{a}, \mu_{b}, \mu_{c}, \sigma_{a}, \sigma_{b},\sigma_{c}$$.

MLE estimates are the latent variables $\mu_c, \sigma_c, c={0,1,...,k}$

Using MLE estimates of parameters, following probabilities can be estimated:
<!-- p(z=1|{x}), \hdots, p(z=k|{x}) -->

<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-condzx.gif" alt="eqn-condzx"/></p>


- **Type 2**: Model assumptions: Discrete categorical latent variables $z \in \{1,2, ..., k\}$, multivariate case: continuous observed variables $\textbf{x}$. The mean $\mu$ and $\sigma$ are different for each component of Gaussians, mixing weights are unknown.

Marginal probability $p(\textbf{z})$:
<!-- p(\textbf{z}) = \prod_{c \in \{ 1,2, \hdots, k \} } \pi_c^{\mathbbm{1}  (\textbf{z}=c)} -->
<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-pzibf.gif" alt="eqn-pzibf"/></p>

Conditional probability $p(\textbf{x}|\textbf{z})$:
<!-- \begin{align*}
p(\textbf{x}|\textbf{z}) = \prod_c \mathcal{N}(\textbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)^{\mathbbm{1} (\textbf{z}=c)}
\end{align*} -->

Joint density or likelihood for $\textbf{x}$:
<!-- \begin{align*}
L &= p(D) = \prod_{i=1}^n \{ \pi_{a} \mathcal{N}(\textbf{x}_i; \boldsymbol{\mu}_{a}, \boldsymbol{\Sigma}_{a}) + \pi_{b} \mathcal{N}(\textbf{x}_i; \boldsymbol{\mu}_{b}, \boldsymbol{\Sigma}_{b}) + \pi_{c} \mathcal{N}(\textbf{x}_i; \boldsymbol{\mu}_{c}, \boldsymbol{\Sigma}_{c}) \}\\
\text{Log-likelihood:}\\
l &= \sum_{i=1}^n \log (\pi_{a} \mathcal{N}(\textbf{x}_i; \boldsymbol{\mu}_{a}, \boldsymbol{\Sigma}_{a}) + \pi_{b} \mathcal{N}(\textbf{x}_i; \boldsymbol{\mu}_{b}, \boldsymbol{\Sigma}_{b}) + \pi_{c} \mathcal{N}(\textbf{x}_i; \boldsymbol{\mu}_{c}, \boldsymbol{\Sigma}_{c}))\\
\end{align*} -->
<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-jointloglikeli1bf.gif" alt="eqn-jointloglikeli1bf"/></p>

E-M algorithm can be used.

Known variables: observed variables $\textbf{x}$;

Unkown variables: latent variables $\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c, c={1,...,k}$, mixing weights $p(\textbf{z}_i=1), p(\textbf{z}_i=2), ..., p(\textbf{z}_i=k)$

MLE estimates are the latent variables $\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c$, mixing weights $\pi_c, c={0,1,...,k}$
Using MLE estimates of parameters, following probabilities can be estimated:

<!-- p(\textbf{z}_i=1|\textbf{x}), \hdots, p(\textbf{z}_i=k|\textbf{x}) -->

<p style="text-align:center;"><img src="/images/equations/mixturemodels/eqn-condzxbf.gif" alt="eqn-condzxbf"/></p>

## References:

1. Grosse, R., Machine Learning [CS 2515](http://www.cs.toronto.edu/~rgrosse/courses/csc2515_2019/)
