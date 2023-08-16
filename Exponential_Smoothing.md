Exponential_Smoothing
================
S. Sudhir
2023-08-16

# Exponential smoothing

Consider the `nhtemp` dataset which holds yearly average measurements of
temperature for New Hampshire, from 1912 to 1971

``` r
library(ggplot2)
require(datasets)
df1 = data.frame(year = 1912:1971, avg.temp = nhtemp)
```

    ##   year avg.temp
    ## 1 1912     49.9
    ## 2 1913     52.3
    ## 3 1914     49.4
    ## 4 1915     51.1
    ## 5 1916     49.4
    ## 6 1917     47.9

![](Exponential_Smoothing_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

We want to fit an exponential smoothing model to this data such that
$\hat{Y}_1 = Y_1$ and, for $i = 2, 3, \ldots, n$,

$$\hat{Y}_i = \beta Y_{i-1} + (1-\beta) \hat{Y}_{i-1}$$

where $\beta$ is a constant between 0 and 1.

``` r
## Forecast function as given above
## Estimated parameter: beta
## Fixed and observable parameter: y
forecast = function(beta, y)
{
  yhat = c()
  yhat[1] = y[1] # initializing yhat[1]
  
  for (i in 2:length(y))
  {
    yhat[i] = beta * y[i-1] + (1 - beta) * yhat[i-1] # recursion for yhat[2] on wards
  }
  return (yhat)
}
```

We will choose the parameter estimate $\hat{\beta}$ that minimizes the
mean forecast error

$$FE(\beta) = \frac{1}{n} \sum_{i=2}^n \left( Y_i - \hat{Y}_i \right)^2$$

``` r
## Parameter beta estimate which minimizes mean forecast error
forecast.error = function(beta, y)
{
  yhat = forecast(beta, y)
  return (1/length(y) * sum((y - yhat)^2))
}
```

The derivatives of this function are rather complicated (notice that
$\hat{Y}_i$ is a function of $\beta$), so let’s use a derivative-free
method based on the function `optimize()`.

## Using `optimize()` on the interval $[0,1]$, to find the value of $\beta$ that produces the minimum forecast error.

``` r
## Finding parameter estimator that minimizes forecast error
FE_opt = optimize(f = forecast.error,
                  interval = c(0,1),
                  tol = 0.0001,
                  y = df1$avg.temp)
FE_opt
```

    ## $minimum
    ## [1] 0.1860813
    ## 
    ## $objective
    ## [1] 1.275533

    ## [1] "The value of beta that minimizes forecast error is 0.186081274135334 with the value at 1.27553266171794"

![](Exponential_Smoothing_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

## Reproducing the previous plot, including some other levels of smoothing, with $\beta=0.05$ and $\beta=0.75$

![](Exponential_Smoothing_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->
