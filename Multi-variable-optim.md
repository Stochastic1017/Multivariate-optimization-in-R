Example of multi-dimensional optimization
================
S. Sudhir
2023-08-17

The two-dimensional distribution of pollutant concentration in a channel
can be described by:

$$z = \text{concentration}(x,y) = 7.9 + 0.13x + 0.21y - 0.05x^2 - 0.016y^2 - 0.007xy$$

``` r
# Create two-dimensional function of pollutant concentration in a channel
concentration <- function(x, y)
{
  result = 7.9 + 
           0.13*x +
           0.21*y -
           0.05*x^2 -
           0.016*y^2 -
           0.007*x*y
  
  return(result)
}
```

## Graph the concentration in the region defined by −10 ≤ x ≤ 10 and 0 ≤ y ≤ 20.

![](Multi-variable-optim_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

## Implement Gradient Descent

``` r
# First Derivative of Function
gradient.concentration = function(par) 
{
  x = par[1]
  y = par[2]
  df.dx = 0.13 - 0.1*x - 0.007*y
  df.dy = 0.21 - 0.032*y - 0.007*x
  return(c(df.dx, df.dy))
}

# Adapting gradient.descent() from notes
gradient.descent = function(par, gr, gamma = 10, epsilon = .0001, 
n= 30,verbose = FALSE, ...)
{
  for (i in seq_len(n)) {
    gradient = gr(par, ...)
    par = par + gamma * gradient
    gradient.size = sum(abs(gradient))
    if (verbose) {
      cat(sep = "", "i = ", i, ", par = c(", paste(signif(par, 4), 
                                                   collapse = ","),
          "), gradient = c(", paste(signif(gradient, 4), collapse = ","),
          "), size = ", signif(gradient.size, 4), "\n")
    }
    if (gradient.size < epsilon) {
      break
    }
  }
  return(par)
}

grad.des = gradient.descent(par = c(0,0), gr = gradient.concentration, verbose = TRUE)
```

    ## i = 1, par = c(1.3,2.1), gradient = c(0.13,0.21), size = 0.34
    ## i = 2, par = c(1.153,3.437), gradient = c(-0.0147,0.1337), size = 0.1484
    ## i = 3, par = c(1.059,4.356), gradient = c(-0.009359,0.09194), size = 0.1013
    ## i = 4, par = c(0.995,4.988), gradient = c(-0.006436,0.06318), size = 0.06961
    ## i = 5, par = c(0.9508,5.422), gradient = c(-0.004422,0.04341), size = 0.04783
    ## i = 6, par = c(0.9204,5.721), gradient = c(-0.003039,0.02983), size = 0.03287
    ## i = 7, par = c(0.8996,5.926), gradient = c(-0.002088,0.0205), size = 0.02258
    ## i = 8, par = c(0.8852,6.066), gradient = c(-0.001435,0.01408), size = 0.01552
    ## i = 9, par = c(0.8753,6.163), gradient = c(-0.0009859,0.009677), size = 0.01066
    ## i = 10, par = c(0.8686,6.23), gradient = c(-0.0006774,0.00665), size = 0.007327
    ## i = 11, par = c(0.8639,6.275), gradient = c(-0.0004655,0.004569), size = 0.005035
    ## i = 12, par = c(0.8607,6.307), gradient = c(-0.0003198,0.00314), size = 0.003459
    ## i = 13, par = c(0.8585,6.328), gradient = c(-0.0002198,0.002157), size = 0.002377
    ## i = 14, par = c(0.857,6.343), gradient = c(-0.000151,0.001482), size = 0.001633
    ## i = 15, par = c(0.856,6.353), gradient = c(-0.0001038,0.001019), size = 0.001122
    ## i = 16, par = c(0.8553,6.36), gradient = c(-7.13e-05,0.0006999), size = 0.0007712
    ## i = 17, par = c(0.8548,6.365), gradient = c(-4.899e-05,0.0004809), size = 0.0005299
    ## i = 18, par = c(0.8544,6.368), gradient = c(-3.366e-05,0.0003305), size = 0.0003641
    ## i = 19, par = c(0.8542,6.371), gradient = c(-2.313e-05,0.0002271), size = 0.0002502
    ## i = 20, par = c(0.854,6.372), gradient = c(-1.589e-05,0.000156), size = 0.0001719
    ## i = 21, par = c(0.8539,6.373), gradient = c(-1.092e-05,0.0001072), size = 0.0001181
    ## i = 22, par = c(0.8539,6.374), gradient = c(-7.505e-06,7.367e-05), size = 8.117e-05

``` r
grad.des.max = concentration(grad.des[1], grad.des[1])

print(paste0("Maximum value of ", 
             sprintf(grad.des.max, fmt = '%#.3f'), 
             " found at coordinates (", 
             sprintf(grad.des[1], fmt = '%#.3f'),
             ",", 
             sprintf(grad.des[2], fmt = '%#.3f'),")"))
```

    ## [1] "Maximum value of 8.137 found at coordinates (0.854,6.374)"

![](Multi-variable-optim_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

## Using `optim()` to solve the same problem with method=“Nelder-Mead” and provide fn but not gr

``` r
#placeholder function to use concentration function with vector input
pfc <- function(x)
{     
  return(concentration(x[1], x[2]))
}

#Nelder-Mead Function
out1.pfc = optim(par = c(mean(x),mean(y)), 
                 fn = pfc, 
                 control = list(fnscale = -1), 
                 method = "Nelder-Mead")

#Format and print result
print(paste0("Maximum value of ", 
             sprintf(out1.pfc$value, fmt = '%#.3f'), 
             " found at coordinates (", 
             sprintf(out1.pfc$par[1], fmt = '%#.3f'), 
             ",", 
             sprintf(out1.pfc$par[2], fmt = '%#.3f'),") in ",
             out1.pfc$counts[1], " iterations"))
```

    ## [1] "Maximum value of 8.625 found at coordinates (0.853,6.376) in 51 iterations"

![](Multi-variable-optim_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## Use `optim()` to solve the same problem with method=“BFGS”, which approximates Newton’s method, providing fn and gr

``` r
#BFGS Function
out2.pfc = optim(par = c(mean(x),mean(y)),
                 fn = pfc,  
                 gr = gradient.concentration, 
                 control = list(fnscale=-1), 
                 method = "BFGS")

#Format and print result
print(paste0("Maximum value of ", 
             sprintf(out2.pfc$value, fmt = '%#.3f'), 
             " found at coordinates (", 
             sprintf(out2.pfc$par[1], fmt = '%#.3f'), 
             ",", 
             sprintf(out2.pfc$par[2], fmt = '%#.3f'),") in ",
             out2.pfc$counts[1], " iterations"))
```

    ## [1] "Maximum value of 8.625 found at coordinates (0.853,6.371) in 10 iterations"

![](Multi-variable-optim_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

## Conclusion

The Nelder-Mead method resulted in 51 iterations, whereas the BFGS
method resulted in 10 iterations.

The “Nelder-Mead” method is simple optimization algorithm that does not
require the calculation of gradients, so it can be efficient
low-dimensional functions that are expensive to compute gradients for.

The “BFGS” method is quasi-Newton method that approximates the Hessian
matrix of the function, and it is more powerful for high-dimensional
functions that have a smooth Hessian.

Since we had a higher dimensional case, we had less iteration for “BFGS”
method.
