# HW: Gravity Simulator

The purpose of this homework is to write a parallel gravity simulator using either
* CUDA or
* HIP
(your choice).

There are two levels of this assignment. You may write the basic simulator using the code you are 
given, or you may write a more sophisticated one that will work much faster using better math.
This is difficult, so +100 for anyone who does it. Note that if you want partial credit for the
more difficult version, it is only available if you make a working version of the simple one.

Write a gravity simulator. The basic code for a scalar simulator is given to you.
The code has not been tested. So it is your responsiblity to find any bugs.
The first question is to define "ground truth". How can we know what the correct answer should be?
The answer, is you are going to test the simulation with the earth and sun.

- Look up the mass of the sun and earth
- Look up the mean distance of the earth from the sun $r$
- We will simplify by assuming earth is in a circular orbit
- calculate 1 year in seconds $secperyear = 365.2425 * 24*60*60$
- calculate the distance around the circle $D = 2*pi*r_{earthorbit}$. This is how far the earth travels every year in our slightly simplified universe.
- Calculate the speed of earth
  $v=D / secperyear$
- Create a scenario where the sun has the correct mass at location (0,0,0) initial velocity (0,0,0)
- For this simulation you may consider the sun to be the center of the universe, and it should not move much. The reason: The sun has $10^6$ times more mass than earth.
- start the earth at (r,0,0) with velocity (0,v,0)
- run the algorithm with a small timestep. You can try dt=1000 sec and dt=100 sec. After one year, earth should be approximately back in its starting position. If this is the case, your code is working and you can proceed to the parallelization stage.
- If you check carefully, you will notice that after one year, the sun has started slowly moving. You might wonder why. It's because the earth spent half a year accelerating the sun in a direction. The next half a year decelerated it. By the end, the sun has stopped moving, but it has moved. If this bothers you, find out the $maximumvelocity$ of the sun and start it with $velocity=-maximumvelocity/2$

Load in at least 10 bodies to test your code. You may read in my data file, or create your own. For this test, the number of operations is:

$numtimesteps = secperyear / dt$

To fully test your GPU if it has a lot of cores, create a simulation with many bodies $n=1000$ will do. There are $O(n^2)$ interactions between each body and every other. With $10^6/2$ interactions, there is more than enough for each CPU to be kept quite busy. 

In order to debug the code, display position and velocity every so often. This can be once per day (every $24*60*60=86400$ seconds)

The required submission for this homework is source code (solarsystem.cu for NVIDIA) and an input file with your simulation initialization. Your program should run with:

```bash
./sim solarsys.dat
```

output should look like:

```bash
t=0
sun     0,0,0            0,0,0
earth   1.52469e+9,0,0   0,32420,0
t=315360
...
```

Extra credit: +100 for successfully implementing this using
Runge-Kutta-Fehlberg (RKF45) see [RKF45](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method)
This is considerably more difficult than the code provided but will dramatically improve performance with bigger timesteps

+20 for generating a nice graphic in MATLAB or python to show what's happening
