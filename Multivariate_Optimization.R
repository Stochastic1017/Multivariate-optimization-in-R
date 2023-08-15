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

for (i in seq(-115,115,20))
{
  Sys.sleep(1)
  for (j in seq(-115,115,20))
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
