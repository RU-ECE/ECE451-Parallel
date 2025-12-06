# HW: Gravity Simulator

The goal of this homework is to write a **parallel gravity (N-body) simulator** using either **CUDA** or **HIP** (your
choice).

There are two levels to this assignment:

1. A **basic simulator** using the provided scalar code.
2. A **more sophisticated simulator** that uses better numerical methods and runs much faster (harder, **+100 points**).

> If you attempt the more difficult version, you can only receive partial credit for it if you also have a working
> version of the simple one.

---

## 1. Basic Gravity Simulator (Earth–Sun Test)

You are given basic code for a scalar gravity simulator. The code has **not** been tested, so it is your responsibility
to:

- Find and fix bugs.
- Verify that the simulation behaves reasonably.

To define a **“ground truth”**, you will test your simulator on the **Earth–Sun system**.

### 1.1. Setup and Parameters

Do the following:

- Look up the **mass of the Sun**.
- Look up the **mass of the Earth**.
- Look up the **mean distance** of the Earth from the Sun, call this distance $r$.
- Assume the Earth is in a **circular orbit**.

Compute:

- One year in seconds:

  $$\text{secperyear} = 365.2425 \cdot 24 \cdot 60 \cdot 60$$

- The distance traveled in one orbit (circumference of the orbit):

  $$D = 2\pi r_{\text{earthorbit}}$$

  This is how far the Earth travels in one year in our simplified model.

- The orbital speed of Earth:

  $$v = \frac{D}{\text{secperyear}}$$

### 1.2. Initial Conditions

Create a simulation scenario with:

- **Sun**
	- Mass: correct solar mass.
	- Initial position: $(0, 0, 0)$.
	- Initial velocity: $(0, 0, 0)$.
	- For this simulation you may treat the Sun as nearly fixed at the center, because it has roughly $10^6$ times more
	  mass than the Earth.

- **Earth**
	- Initial position: $(r, 0, 0)$.
	- Initial velocity: $(0, v, 0)$.

### 1.3. Running the Simulation

- Use a **small timestep**, for example:
	- $\text{dt} = 1000$ seconds, and/or
	- $\text{dt} = 100$ seconds.
- Run the simulation for **one year** of simulated time:

  $$\text{numtimesteps} = \frac{\text{secperyear}}{\text{dt}}$$

- After one year, the Earth should be **approximately back at its starting position**.  
  If this is the case (within reasonable error), your basic code is probably working, and you can proceed to
  parallelization.

### 1.4. Sun’s Motion (Drift)

If you examine the results carefully, you may notice that after one year, the **Sun has started to move slowly**.

Why?

- The Earth pulls on the Sun.
- For about half the year, it accelerates the Sun in one direction.
- For the next half, it decelerates it.
- By the end of the year, the Sun’s **velocity** may be near zero again, but its **position** has shifted.

If this bothers you:

- Find the **maximum velocity** the Sun attains during the run, call it `maximumvelocity`.
- Initialize the Sun with:

  $$\text{velocity} = -\frac{\text{maximumvelocity}}{2}$$

to partially cancel the net drift.

---

## 2. Many-Body Tests

Once the Earth–Sun test works, move on to **larger systems**.

- Load at least **10 bodies** to test your code.
	- You may use an instructor-provided data file or create your own.
- Use the same formula for the number of timesteps:

  $$\text{numtimesteps} = \frac{\text{secperyear}}{\text{dt}}$$

To fully test a GPU with many cores, run a simulation with **many bodies**, for example:

- $n = 1000$ bodies.

There are $O(n^2)$ interactions, because each body interacts with every other body. With about $\frac{10^6}{2}$
interactions, there is more than enough work to keep each core busy.

---

## 3. Debugging and Output

To help debug the code:

- Periodically display the **position** and **velocity** of each body.
- A reasonable choice is once per simulated day:

  $$24 \cdot 60 \cdot 60 = 86400 \text{ seconds}$$

---

## 4. Program Interface and Example Output

Your program should:

- Read an input file describing the initial conditions (e.g. positions, velocities, masses).
- Run the simulation.
- Print selected timesteps in a clear, regular format.

Example output style:

```text
t=0
sun 0,0,0 0,0,0
earth 1.52469e+9,0,0 0,32420,0

t=315360
...
```

Use a similar clear format for all bodies and timesteps you choose to print.

---

## 5. Extra Credit

You can earn extra credit by extending your simulator:

* **+100 points**

  Implement the simulation
  using [Runge–Kutta–Fehlberg (RKF45)](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method). This
  is considerably more difficult than using the basic integrator, but it allows **much larger timesteps** while
  maintaining good accuracy and can dramatically improve performance.

* **+20 points**

  Generate a **nice graphic** in MATLAB or Python that shows what is happening:
	* For example, plot the orbits/trajectories in 2D or 3D over time.
