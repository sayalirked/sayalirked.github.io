---
title: 'Relevance Vector Machine Learning for Bayesian Linear Regression'
date: 2021-08-08
excerpt: The post explains the concept of **Relevance Vector Machine (RVM) Learning**. Click [here](https://sayrjked.github.io/posts/2021/08/relevance-vector-machine/) to read further.
permalink: /posts/2021/08/relevance-vector-machine/

tags:
  - Machine Learning
  - Probability
  - Data Science
---

Relevance Vector Machine (RVM) is a sparse Bayesian learning technique and provides probabilistic estimates of model parameters. Different modeling challenges including noisy data, overfitting, the curse of dimensionality, small experimental datasets exist. Thus, Bayesian relevance learning addresses these challenges and provides many advantages.

Consider a linear regression forward problem as follows: $$\mathbf{d} = \mathbf{K} \boldsymbol{\theta} + \boldsymbol{\eta}$$.

Here, the noise $$\boldsymbol{\eta}$$ is assumed to be Gaussian independent and identically (i.i.d.) random vector, $$\mathbf{K}$$ is the kernel, $$\boldsymbol{\theta}$$ are the unknown stochastic model parameters to be estimated and $$\mathbf{d}$$ is the experimental data.

[Detailed writeup coming soon.]

## References:

1. M. E. Tipping. Sparse bayesian learning and the relevance vector machine. Journal of Machine Learning Research, 1(3):211–244, 2001.

2. C. M. Bishop. Pattern Recognition and Machine Learning. Springer, New York, 2006.

