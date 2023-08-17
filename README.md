# Multivariate Optimization in R

**The repository contains various instances of optimization using R:**
1. Example of single variable one-dimensional optimization
2. Example of multi-dimensional optimization
3. Visual optimization of a multivariate function
4. Optimization for Tukey's robust regression
5. Optimization for exponential smoothing for time-series data
6. Optimization for Maximum likelihood estimator (MLE)
   * MLE for mean and standard deviation of a random sample from $N(\mu, \sigma)$
   * MLE for the parameters $\beta_0$ and $\beta_1$ in logistic regression

# 1. Example of single variable one-dimensional optimization:

An object subject to linear drag due to air resistance is projected
upward at a specified velocity. Its altitude $z$ is modeled as a
function of time $t$ as:

$$z(t) = z_0 + \frac{m}{c}\bigg(v_0 + \frac{mg}{c}\bigg)\bigg(1 - e^{\frac{-c}{m}t}\bigg) - \frac{mg}{c}t$$

where:

$z:$ altitude (in m) above Earth’s surface (which has $z = 0$)

$z_0:$ initial altitude (in m)

$m:$ mass (in kg)

$c:$ Linear Drag Coefficient (in kg/s)

$v_0:$ initial velocity (in m/s) where positive velocity is up

$t:$ time (in s)

``` r
# Create height function with inputs for:

# time: time in seconds
# alt.init: initial altitude in meters
# mass: mass of object in kg
# vel.init: initial velocity in m/s
# drag: drag coefficients in kg/s
# g: gravity default to 9.81 m/s2

altitude.fun <- function(time, alt.init, mass, drag, vel.init, g)
{
  altitude = alt.init + 
             ( (mass/drag) * (vel.init + ((mass*g)/drag)) * (1 - exp((-drag/mass)*time)) ) - 
             ( (mass*g/drag)*time )
  
  altitude.out = ifelse(altitude < 0, 0, altitude) # if object hits ground set value to 0
  return(altitude.out)
}
```

## Graph of function

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Graph_Trejectory.png" width="400" height="300">

## Time when object strikes the ground

``` r
# Running For Loop to find the first instance of time when ball hits the ground, 
# and then optimizing the function from 0 to that time t

for (t in time)
{
  if (altitude.fun(t, alt.init, mass, drag, vel.init, g) == 0)
  {
    min_t = optimize(f = altitude.fun, 
                          interval = c(0,t), 
                          alt.init = 100, 
                          mass = 80, 
                          drag = 15, 
                          vel.init = 55, 
                          g = 9.81, 
                          tol = 0.01)
    break
  }
}

print(paste0("Time it takes for the object to reach the ground is: ", 
             sprintf(min_t$minimum, fmt = '%#.3f')," seconds"))
```

    ## [1] "Time it takes for the object to reach the ground is: 11.614 seconds"

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Trajectory_Ground.png" width="400" height="300">

## Find the object’s maximum height

``` r
# Using optimize to find both time and height where function reaches maximum
max_t = optimize(f = altitude.fun, 
                          interval = c(0,15), 
                          alt.init = 100, 
                          mass = 80, 
                          drag = 15, 
                          vel.init = 55, 
                          g = 9.81, 
                          tol = 0.01,
                          maximum = TRUE)

print(paste0("The maximum height the object reaches is : ", 
             sprintf(max_t$objective, fmt = '%#.3f')," meters")) 
```

    ## [1] "The maximum height the object reaches is : 192.861 meters"

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Trajectory_Height.png" width="400" height="300">

## Find the time at which the object reaches its maximum height

    ## [1] "Time it takes for the object to reach the maximum height is: 3.833 seconds"

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Trejectory_optim.png" width="400" height="300">

# 2. Example of multi-dimensional optimization:

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

## Graph of the concentration in the region defined by −10 ≤ x ≤ 10 and 0 ≤ y ≤ 20.

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Graph_Multivariate.png" width="400" height="300">

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/First_optim.png" width="400" height="300">

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Second_optim.png" width="400" height="300">

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Third_optim.png" width="400" height="300">

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

# 3. Visual optimization of a multivariate function:

Goal is to visualize and find all local maxima's for the function:
$$z = f(x,y) = \big(1 - \frac{x}{k}\big)\big(1 + \frac{x}{k}\big)\big(1 - \frac{y}{k}\big)\big(1 + \frac{y}{k}\big)\bigg[ -(y+47)sin\bigg(\sqrt{|y + \frac{x}{2} + 47|}\bigg) - xsin\big(\sqrt{|x - (y + 47)|}\big) \bigg]$$
in the interval $-120 < x < 120$ and $-120 < y < 120$.

```r
library(rgl)

# Define the function
k <- 120
f <- function(x, y) {
  return ((1 - x/k) * (1 + x/k) * (1 - y/k) * (1 + y/k) *
            (-(y + 47) * sin(sqrt(abs(y + (x/2) + 47))) - x * sin(sqrt(abs(x - (y+47))))))
}

# Create a grid of x and y values
x <- seq(-k, k)
y <- seq(-k, k)
z <- outer(x, y, f)

persp3d(x, y, z, theta = 30, phi = 30, expand = 0.5, col = "blue",
        xlab = "x", ylab = "y", zlab = "z")

# helper function for f() so it can be used with optim()
f_helper <- function(x) {
  y <- x[2]
  x <- x[1]

  f(x, y)
}

highest_par_val = c(0,0)

for (i in seq(-120, 120, 20))
{
  Sys.sleep(1)
  for (j in seq(-120, 120, 20))
   {
    local_max <- optim(c(i,j), fn = f_helper, control = list(fnscale = -1))
    pars <- local_max$par
    value <- local_max$value
    if(pars[1] <= 120 & pars[1] >= -120 & pars[2] <= 120 & pars[2] >= -120 & value < highest_par_val[2])
    {
      points3d(x = pars[1], y = pars[2], z = value, col = "red", size = 10)
    }
    else if(pars[1] <= 120 & pars[1] >= -120 & pars[2] <= 120 & pars[2] >= -120 & value > highest_par_val[2])
    {
      points3d(x = pars[1], y = pars[2], z = value, col = "red", size = 10)
      highest_par_val <- c(pars,value)
    }
  }
}

points3d(x = highest_par_val[1], y = highest_par_val[2], z = highest_par_val[3], col="green", size = 20)
```


The code does the following:
* Graph $z = f(x,y)$ in `persp3d` from the package `rgl`.
* Uses Nelder-Mead numerical optimizer to find and mark all local maxima's with a red-dot. `points3d` was used to do this.
* Finds the global maxima in the interval and marks it with a green-dot.

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Function_image.png" width="500" height="500">

# 4. Tukey's robust regression:

## Head of Dataset

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Scatterplot_farmland.png" width="400" height="300">

## Explanation

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

## Implementation

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

### Fitting linear least square regression line using `lm()`

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/SSE.png" width="400" height="300">

### Estimating $\beta_0$ and $\beta_1$ using the Nelder-Mead method in `optim()` with the initial parameters `c(0 ,0)`

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Nelder_Mead.png" width="400" height="300">

### Estimating $\beta_0$ and $\beta_1$ using the BFGS method in `optim()` with the initial parameters `c(0 ,0)`

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/BFGS.png" width="400" height="300">

### Estimating $\beta_0$ and $\beta_1$ using the CG method in `optim()` with the initial parameters `c(0 ,0)`

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/CG.png" width="400" height="300">

### Summary and Conclusion

    ##                          method    intercept     slope    values
    ## 1 Robust Regression Nelder-Mead    0.1149554 0.3988888 447645184
    ## 2        Robust Regression BFGS 5134.2092421 0.3245301 441085793
    ## 3          Robust Regression CG    0.3989329 0.3989329 447645463

    ## [1] "The method for which the value of Tukey function is the smallest is Robust Regression BFGS with the value 441085792.843601"

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/All_Tukey_Lines.png" width="400" height="300">

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

# 5. Exponential smoothing for time-series data

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Original_TS.png" width="400" height="300">

We want to fit an exponential smoothing model to this data such that
$\hat{Y}_1 = Y_1$ and, for $i = 2, 3, \ldots, n$,

$$Y_i = \beta Y_{i-1} + (1-\beta) Y_{i-1}$$

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

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Smoothing_TS.png" width="400" height="300">

## Reproducing the previous plot, including some other levels of smoothing, with $\beta=0.05$ and $\beta=0.75$

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/All_TS.png" width="400" height="300">

# 6. Optimization for Maximum Likelihood Estimation (MLE):

## 6(a.) MLE for mean and standard deviation of a random sample from $N(\mu, \sigma)$

Here we’ll use optimization to confirm that, given a simple random
sample $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$, the maximum-likelihood
estimates for the unknown mean $\mu$ and standard deviation $\sigma$ are

$$\mu = \frac{1}{n} \sum_{i=1}^n X_i$$

and

$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (X_i - \mu)^2}$$

Since each $X_i \sim N(\mu, \sigma^2)$ has the probability density
function

$$f(x_i; \mu, \sigma) = \frac{1}{\sqrt{2 \pi} \sigma} \exp\left(-\frac{(x_i - \mu)^2}{2 \sigma^2}\right)$$

``` r
## Function for Normal Probability Density Function
norm_pdf = function(x, mean, sd)
{
  return(1/(sqrt(2*pi)*sd) * exp((-1/(2*sd^2))*(x - mean)^2))
}
```

and the $X_i$’s are independent, the density function for the sample is

$$f(x_1, \ldots, x_n; \mu; \sigma) = \prod_{i=1}^n f(x_i; \mu, \sigma) = \left(\frac{1}{\sqrt{2 \pi} \sigma}\right)^n \exp\left(-\frac{1}{{2 \sigma^2}} \sum_{i=1}^n (x_i - \mu)^2\right)$$

If we now consider the sample $(x_1, \ldots, x_n)$ as fixed, then
$f(x_1, \ldots, x_n; \mu; \sigma)$ can be regarded as a function of
$\mu$ and $\sigma$ called the likelihood, $L$:

$$L(\mu, \sigma; x_1, \ldots, x_n) = f(x_1, \ldots, x_n; \mu; \sigma)$$

We want to use optimization to find the $(\mu, \sigma)$ pair that
maximizes $L(\mu, \sigma; x_1, \ldots, x_n)$. However, computing $L$ is
problematic because its product of small numbers often leads to
underflow, in which the product is closer to zero than a computer can
represent with the usual floating-point arithmetic. Taking logarithms
addresses this problem by transforming products of very small positive
numbers to sums of moderate negative numbers. For example, $10^{-10}$ is
very small, but $\log(10^{-10}) \approx -23.03$ is moderate. With this
in mind, the log likelihood $l$ is

$$l(\mu, \sigma; x_1, \ldots, x_n) = \log\left(L(\mu, \sigma; x_1, \ldots, x_n)\right) = n \log\left(\frac{1}{\sqrt{2 \pi} \sigma}\right) -\frac{1}{{2 \sigma^2}} \sum_{i=1}^n (x_i - \mu)^2$$

``` r
## Log Likelihood Function where par = c(mean, sd)
Log.Lik = function(par, x)
{
  n = length(x)
  lik1 = n * log(1/(sqrt(2*pi)*par[2]))
  lik2 =  1/(2*par[2]^2) * sum((x - par[1])^2)
  
  return (-(lik1 - lik2)) # Returning negative so as to find maximum
}
```

Since the logarithm is an increasing function, the maximum of $l$ occurs
at the same location $(\mu, \sigma)$ as the maximum of $L$.

### Using `optim()` with its default Nelder-Mead method to find the estimates of $\mu$ and $\sigma$ that maximize $l$ over the data $x_1, \ldots, x_n =$ `mtcars$mpg`

``` r
## Optimizing Log-Likelihood to find parameter estimates
opt.par = optim(par = c(.5, .5), 
             fn = Log.Lik, 
             x = mtcars$mpg,
             method = 'Nelder-Mead',
             upper = Inf,
             lower = -Inf)

opt.par
```

    ## $par
    ## [1] 20.10088  5.94055
    ## 
    ## $value
    ## [1] 102.3779
    ## 
    ## $counts
    ## function gradient 
    ##       89       NA 
    ## 
    ## $convergence
    ## [1] 0
    ## 
    ## $message
    ## NULL

    ## [1] "The par = (mean, sd) estimations for Log-Likelihood using Nelder-mead is (20.100877, 5.940550)"

### Checking estimates by comparing them to the sample mean and (population) standard deviation.

    ## [1] "The mean and sd estimate using mle and Nelder-Mead optimization is (20.100877, 5.940550)"

    ## [1] "The mean and sd of the data mtcars$mpg is (20.090625, 6.026948)"

<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Comparing%20Normal%20Dist.png" width="400" height="300">

As our estimated parameters fits very closely to the sample mean and
population sd under the normal distribution, we can deduce with
certainty that the MLE for mean and sd of a random distribution is
approximately the sample mean and population sd.

## 6(b.) MLE for the parameters $\beta_0$ and $\beta_1$ in logistic regression:

In simple logistic regression, we have a numeric explanatory variable
$X$ and binary response variable $Y$ that takes one of two values, 0
(failure) or 1 (success). We suppose that
$P(Y=1|X=x) = p(x; \beta_0, \beta_1)$ for some function $p$ of the data
$x$ and two parameters $\beta_0$ and $\beta_1$, so that
$P(Y=0|X=x) = 1 - p(x; \beta_0, \beta_1)$.

Given the data $(x_1, y_1), \ldots, (x_n, y_n)$, where each
$y_i \in \{0, 1\}$, the probability of the data under the model is

$$f(y_1, \ldots, y_n | x_1, \ldots, x_n; \beta_0, \beta_1) = \prod_{i=1}^n p(x_i; \beta_0, \beta_1)^{y_i} (1 - p(x_i; \beta_0, \beta_1))^{1-y_i}$$

A *logistic transformation* maps $p \in [0, 1]$ to
$\log\left(\frac{p}{1-p}\right)$, whose range is the entire real line.
We define $p(x; \beta_0, \beta_1)$ implicitly by requiring its logistic
transformation to be linear:

$$\log\left(\frac{p(x; \beta_0, \beta_1)}{1-p(x; \beta_0, \beta_1)}\right) = \beta_0 + \beta_1 x$$

Solving for $p(x; \beta_0, \beta_1)$ gives

$$p(x; \beta_0, \beta_1) = \frac{1}{1 + \exp(-(\beta_0 + \beta_1 x))}$$

which is called the sigmoid function.

``` r
## Function for p(x;par) where par = c(beta0, beta1)
sigmoid = function(par, x)
{
  return ( 1/(1 + exp(-par[1] - par[2]*x)) )
}
```

The likelihood of $(\beta_1, \beta_1)$ given the data is then

$$L(\beta_0, \beta_1; x_1, \ldots, x_n, y_1, \ldots, y_n) = \prod_{i=1}^n \left(\frac{1}{1 + \exp(-(\beta_0 + \beta_1 x_i))}\right)^{y_i} \left(1 - \frac{1}{1 + \exp(-(\beta_0 + \beta_1 x_i))}\right)^{1-y_i}$$

and the log likelihood is (after a few lines of work)
$$l(\beta_0, \beta_1; x_1, \ldots, x_n, y_1, \ldots, y_n) = -\sum_{i=1}^n \log(1 + \exp(\beta_0 + \beta_1 x_i)) + \sum_{i=1}^n y_i (\beta_0 + \beta_1 x_i)$$

``` r
Log.Lik = function(par, x, y)
{
  p = sigmoid(par, x)
  return ( -sum(y*log(p) + (1-y)*log(1-p)) )
}
```

Consider the `menarche` data frame in the `MASS` package
(`require("MASS"); ?menarche`). It gives proportions of girls at various
ages who have reached menarche.

    ##      Age Total Menarche
    ## 1   9.21   376        0
    ## 2  10.21   200        0
    ## 3  10.58    93        0
    ## 4  10.83   120        2
    ## 5  11.08    90        2
    ## 6  11.33    88        5
    ## 7  11.58   105       10
    ## 8  11.83   111       17
    ## 9  12.08   100       16
    ## 10 12.33    93       29
    ## 11 12.58   100       39
    ## 12 12.83   108       51
    ## 13 13.08    99       47
    ## 14 13.33   106       67
    ## 15 13.58   105       81
    ## 16 13.83   117       88
    ## 17 14.08    98       79
    ## 18 14.33    97       90
    ## 19 14.58   120      113
    ## 20 14.83   102       95
    ## 21 15.08   122      117
    ## 22 15.33   111      107
    ## 23 15.58    94       92
    ## 24 15.83   114      112
    ## 25 17.58  1049     1049

The first row says “0 out of 376 girls with average age 9.21 have
reached menarche.” The tenth row says “29 out of 93 girls with average
age 12.33 have reached menarche.” The last row says “1049 out of 1049
girls with average age 17.58 have reached menarche.”

Here I’ll make a second data frame called `menarche.cases` from
`menarche` that gives one line for each girl in the study indicating her
age and whether (1) or not (0) she has reached menarche.

``` r
success.indices = rep(x=seq_len(nrow(menarche)), times=menarche$Menarche)
success.ages = menarche$Age[success.indices]
success = data.frame(age=success.ages, reached.menarche=1)
failure.indices = rep(x=seq_len(nrow(menarche)), times=menarche$Total - menarche$Menarche)
failure.ages = menarche$Age[failure.indices]
failure = data.frame(age=failure.ages, reached.menarche=0)
menarche.cases = rbind(success, failure)
menarche.cases = menarche.cases[order(menarche.cases$age), ]
rownames(menarche.cases) = NULL # Remove incorrectly ordered rownames; they get restored correctly.
```
       ## # A tibble: 25 × 3
    ##      Age Total Menarche
    ##    <dbl> <dbl>    <dbl>
    ##  1  9.21   376        0
    ##  2 10.2    200        0
    ##  3 10.6     93        0
    ##  4 10.8    120        2
    ##  5 11.1     90        2
    ##  6 11.3     88        5
    ##  7 11.6    105       10
    ##  8 11.8    111       17
    ##  9 12.1    100       16
    ## 10 12.3     93       29
    ## # ℹ 15 more rows

### Use `optim()` with its default Nelder-Mead method to find the estimates of $\beta_0$ and $\beta_1$ that maximize $l$ over the data $x_1, \ldots, x_n, y_1, \ldots, y_n =$ `age`, `reached.menarche` from `menarche.cases`. Check your `optim()` estimates by making a graph with these elements:

``` r
opt.par = optim(par = c(.5, .5),
                fn = Log.Lik,
                gr = NULL,
                x = menarche.cases$age,
                y = menarche.cases$reached.menarche, 
                method = 'Nelder-Mead',
                lower = -Inf,
                upper = Inf)

opt.par
```

    ## $par
    ## [1] -21.220857   1.631518
    ## 
    ## $value
    ## [1] 819.6524
    ## 
    ## $counts
    ## function gradient 
    ##      109       NA 
    ## 
    ## $convergence
    ## [1] 0
    ## 
    ## $message
    ## NULL

The 3918 points (x=age, y=reached.menarche) from `menarche.cases`. Since
there are only 25 ages, these points would overlap a lot. To fix the
overlap, I used `jitter()` to add a little random noise to each vector
of coordinates. For example, `jitter(c(1, 2, 3))` gives something like
`c(1.044804, 1.936708, 2.925454)`.

``` r
## The 3918 points (x=age, y=reached.menarche) from menarche.cases with jitter
## to create random noise for age
x1 = c(jitter(menarche.cases$age))
y1 = c(menarche.cases$reached.menarche)
id1 = rep('menarche.cases', nrow(menarche.cases))
```

The 25 points $(x_i, y_i)$ where $x_i$ is the $i$th age in the original
`menarche` data frame, and $y_i$ is the proportion of girls of that age
who have reached menarche.

``` r
## 25 points (xi, yi) where xi is the ith age in the original menarche data frame, 
## and yi is the proportion of girls of that age who have reached menarche
x2 = c(menarche$Age)
y2 = c()

for (i in 1:nrow(menarche))
{
  y2 = append(y2, 1/menarche[i,]$Total * menarche[i,]$Menarche)
}

id2 = rep('menarch', nrow(menarche))
```
<img src="https://github.com/Stochastic1017/Multivariate-optimization-in-R/blob/main/Images/Fitting%20Logistic%20Curve.png" width="400" height="300">
