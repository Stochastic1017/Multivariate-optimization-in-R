# 1. Example of single variable one-dimensional optimization

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

## Find the time at which the object reaches its maximum height

    ## [1] "Time it takes for the object to reach the maximum height is: 3.833 seconds"

