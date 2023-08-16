The repository contains two instances of optimization using R:

1. Multivariate optimization in R:

This repository contains R code to optimize and finds all local maxima's for the function:
$$z = f(x,y) = \big(1 - \frac{x}{k}\big)\big(1 + \frac{x}{k}\big)\big(1 - \frac{y}{k}\big)\big(1 + \frac{y}{k}\big)\bigg[ -(y+47)sin\bigg(\sqrt{|y + \frac{x}{2} + 47|}\bigg) - xsin\big(\sqrt{|x - (y + 47)|}\big) \bigg]$$
in the interval $-120 < x < 120$ and $-120 < y < 120$.

The code does the following:
* Graph $z = f(x,y)$ in `persp3d` from the package `rgl`.
* Uses Nelder-Mead numerical optimizer to find and mark all local maxima's with a red-dot. `points3d` was used to do this.
* Finds the global maxima in the interval and marks it with a green-dot.

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Function_image.png" width="500" height="500">

2. Optimization of Tukey's robust regression, exponential smoothing for time-series data, maximum likelihood estimator in logistic regression:
