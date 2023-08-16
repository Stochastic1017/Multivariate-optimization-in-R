Tukey Robust Regression
================
S. Sudhir
2023-08-16

# Head of Dataset

Consider the dataset that has the land and farm area in square miles for
all U.S. states:

    ## # A tibble: 50 × 3
    ##    state         land  farm
    ##    <chr>        <dbl> <dbl>
    ##  1 Alabama      50744 14062
    ##  2 Alaska      567400  1375
    ##  3 Arizona     113635 40781
    ##  4 Arkansas     52068 21406
    ##  5 California  155959 39688
    ##  6 Colorado    103718 48750
    ##  7 Connecticut   4845   625
    ##  8 Delaware      1954   766
    ##  9 Florida      53927 14453
    ## 10 Georgia      57906 16094
    ## # ℹ 40 more rows

# Scatterplot of Dataset

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

# Explanation

We want to build a regression model for `farm`, explained by `land`, but
we know Alaska is an outlier (and Texas is a leverage point, that is,
one with an extreme $x$ coordinate). The normal least squares line is
found by choosing the parameters $\beta_0$ and $\beta_1$ that minimize
the sum of squared residuals, i.e.,

$$S(\beta_0,\beta_1) = \frac{1}{n} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$$

An alternative to fitting a least squares line is to fit a line based on
Tukey’s $\rho$ norm, that is, finding the parameters $\beta_0$ and
$\beta_1$ that minimize the Tukey function:

$$\text{Tukey}(\beta_0,\beta_1) = \frac{1}{n} \sum_{i=1}^n \rho(y_i - \beta_0 - \beta_1 x_i)$$

where $\rho(t)$ is given by

$$\rho(t) = \begin{cases}
t^2, &  |t| \leq k \\
2 k |t| - k^2, &  |t| > k
\end{cases}$$

We’ll use gradient-based methods (among others) to minimize
$\text{Tukey}(\beta_0,\beta_1)$, so we’ll need its gradient. We can
differentiate $\rho(t)$ to get

$$\rho\prime(t) = \begin{cases}
2 t, &  |t| \leq k \\
2 k \, \mbox{sign}(t), &  |t| > k
\end{cases}$$

which means that

$$
\frac{\partial}{\partial \beta_0} \text{Tukey}(\beta_0,\beta_1) = - \frac{1}{n} \sum_{i=1}^n \rho\prime (y_i - \beta_0 - \beta_1 x_i)
$$

$$
\frac{\partial}{\partial \beta_1} \text{Tukey}(\beta_0,\beta_1) = - \frac{1}{n} \sum_{i=1}^n x_i \rho\prime (y_i - \beta_0 - \beta_1 x_i)
$$

# Implementation

``` r
## Function for Tukey's rho norm given above
## k: fixed and chosen, t: y[i] - beta0 - beta1*x[i]
rho = function(t, k)
{
  return (ifelse(abs(t) <= k, t^2, 2*k*abs(t) - k^2))
}

## Function for Tukey(beta) given above
## Estimated parameter: beta = c(beta0, beta1)
## Fixed and observable parameters: x, y
Tukey = function(beta, x, y)
{
  obj.fun = 0
  n = length(x)
  
  for (i in 1:n)
  {
    obj.fun = obj.fun + rho(y[i] - beta[1] - beta[2]*x[i], k)
  }
  
  return (1/n * obj.fun)
}

## Derivative of Tukey's rho norm w.r.t t given above
rho.prime = function(t, k) 
{
  return (ifelse(abs(t) <= k, 2*t, 2*k*sign(t)))
}

## Derivative of Tukey function w.r.t beta = c(beta0, beta1)
## returns partial derivative w.r.t beta0 and beta1
Tukey.gr = function(par, x, y)
{
  dT.db0 = 0
  dT.db1 = 0
  n = length(x)
  
  for (i in 1:n)
  {
    dT.db0 = dT.db0 + rho.prime(y[i] - par[1] - par[2] * x[i], k)
    dT.db1 = dT.db1 + (x[i] * rho.prime(y[i] - par[1] - par[2] * x[i], k))
  }
  return (c(-1/n * dT.db0, -1/n * dT.db1))
}
```

## Fitting linear least square regression line using `lm()`

``` r
## Estimates of ordinary least squares regression line using lm()
summary(lm(area$farm ~ area$land))$coefficients[,'Estimate']
```

    ##  (Intercept)    area$land 
    ## 1.845242e+04 1.456762e-01

``` r
## Extracting linear least squares coefficients from lm() summary
beta_0 = summary(lm(area$farm ~ 
                    area$land))$coefficients['(Intercept)', 
                                             'Estimate']

beta_1 = summary(lm(area$farm ~ 
                    area$land))$coefficients['area$land', 
                                             'Estimate']

beta_lm = c(beta_0, beta_1)
```

    ## [1] "The Beta coefficient vector for Normal Least Squares Regression line using lm is (18452.421461, 0.145676)"

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## Estimating $\beta_0$ and $\beta_1$ using the Nelder-Mead method in `optim()` with the initial parameters `c(0 ,0)`

``` r
k = 19000

## Optimizing Tukey using Nelder-Mead optimization (no gradient used)
rlm_nm_opt = optim(par = c(0, 0), 
                fn = Tukey, 
                #gr = Tukey.gr,
                x = area$land, 
                y = area$farm, 
                method = 'Nelder-Mead',
                lower = -Inf,
                upper = Inf)
rlm_nm_opt
```

    ## $par
    ## [1] 0.1149554 0.3988888
    ## 
    ## $value
    ## [1] 447645184
    ## 
    ## $counts
    ## function gradient 
    ##       41       NA 
    ## 
    ## $convergence
    ## [1] 0
    ## 
    ## $message
    ## NULL

    ## [1] "The Beta coefficient vector for Robust Regression line using Nelder-Mead is (0.114955, 0.398889)"

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

## Estimating $\beta_0$ and $\beta_1$ using the BFGS method in `optim()` with the initial parameters `c(0 ,0)`

``` r
## Optimizing Tukey using BFGS optimization (gradient used)
rlm_bfgs_opt = optim(par = c(0, 0), 
                fn = Tukey,
                gr = Tukey.gr,
                x = area$land, 
                y = area$farm, 
                method = 'BFGS',
                lower = -Inf,
                upper = Inf)
rlm_bfgs_opt
```

    ## $par
    ## [1] 5134.2092421    0.3245301
    ## 
    ## $value
    ## [1] 441085793
    ## 
    ## $counts
    ## function gradient 
    ##       35        6 
    ## 
    ## $convergence
    ## [1] 0
    ## 
    ## $message
    ## NULL

    ## [1] "The Beta coefficient vector for Robust Regression line using BFGS is (5134.209242, 0.324530)"

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

## Estimating $\beta_0$ and $\beta_1$ using the CG method in `optim()` with the initial parameters `c(0 ,0)`

``` r
## Optimizing Tukey using CG optimization (gradient used)
rlm_cg_opt = optim(par = c(0, 0), 
                fn = Tukey,
                gr = Tukey.gr,
                x = area$land, 
                y = area$farm, 
                method = 'CG',
                lower = -Inf,
                upper = Inf)
rlm_cg_opt
```

    ## $par
    ## [1] 0.0003340687 0.3989328915
    ## 
    ## $value
    ## [1] 447645463
    ## 
    ## $counts
    ## function gradient 
    ##      856      101 
    ## 
    ## $convergence
    ## [1] 1
    ## 
    ## $message
    ## NULL

    ## [1] "The Beta coefficient vector for Robust Regression line using CG is (0.000334, 0.398933)"

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

## Summary and Conclusion

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

    ##                          method    intercept     slope    values
    ## 1 Robust Regression Nelder-Mead    0.1149554 0.3988888 447645184
    ## 2        Robust Regression BFGS 5134.2092421 0.3245301 441085793
    ## 3          Robust Regression CG    0.3989329 0.3989329 447645463

    ## [1] "The method for which the value of Tukey function is the smallest is Robust Regression BFGS with the value 441085792.843601"

When we create a plot of the $\rho(t)$ function (using `curve()`) over
the interval $t \in (-100000,100000)$, we can see that:

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

The SSE loss function is greatly influenced by outliers due to the
‘square’ component of errors. Small number outliers will have large
magnitude change in SSE and thus influence the regression line as can be
seen by the limegreen color ordinary regression line.

One way to control this type of influence by outliers is to instead use
absolute values rather than squares for our loss function. This allows
the small number of outliers to have a lesser magnitude change in
absolute loss function compared SSE loss function and is thus more
robust. However, the drawback with this is that absolute function are
not differentiable at their minimum (due to sharp turn) and thus
optimization becomes extremely tedious due to non-convergence issues.

Therefore, in order to achieve the robustness of absolute loss function,
as well as the differentiable aspect of SSE, the Tukey’s rho norm comes
in to play as it follows the SSE loss function for small values of $t$
and constant times $|t|$ for large values of $t$ (large and small values
determined by $k$ which is fixed).

Plotting the derivative of Tukey’s rho norm (rho.prime), we can see that
the function is differentiable throughout, and therefore gradient
optimization can be used effectively.

![](Tukey_Robust_Regression_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->
