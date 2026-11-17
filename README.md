Details of members
 
 1. Avi Gupta MS26BTECH11007
 2. Yuvraj Singh B. MS26BTECH11039
 3. Snigdha Das MS26BTECH11032
 4. Aryan Bhardwaj MS26BTECH11007

 Sequence of videos:
 
 1st video by Avi Gupta

 2nd video by Yuvraj Singh Bachchawat

 3rd video by Snigdha Das

 4th video by Aryan Bhardwaj


 Roles and responsibilities:

 1. Avi Gupta: Particle and NumPy initialization, tests, user interface.
 2. Yuvraj Singh: Force calculation and integration algorithm design and optimization.
 3. Snigdha Das: Implementation, energy conservation and calculation.
 4. Aryan: Visualization, bug fixing and testing.

Acknowledgement of AI usage:
1. Gemini - Used for optimization and code refinement and UI rehaul

##

## Overview
This is our submission for the FOSC project, a toy to simulate the motion of gas particles with a mutual attractive force.

## Methods
The "atoms" are confined in an $l \times l \times l$ box, which is subdivided into cells of volume $k \times k \times k$, with each cell having cell indices $x_{cell}, y_{cell}, z_{cell}$.

An atom is defined by the following data scheme:

$$
\begin{bmatrix}
x \\
y \\
z \\
v_x \\
v_y \\
v_z \\
x_{\text{cell}} \\
y_{\text{cell}} \\
z_{\text{cell}} \\
\end{bmatrix}
$$

$\text{where}$

$$
x,y,z \rightarrow \text{Components of position in cartesian coordinates} \\
v_i \rightarrow \text{Component of velocity in i direction} \\
x_{\text{cell}}, y_{\text{cell}}, z_{\text{cell}} \rightarrow \text{The indices of the cell containing the atom}
$$

Also, $m$ is the mass of the atom.

A single vector is chosen to represent the atom so that the vectorization capabilities of NumPy can be used to speed up calculations.

The cell index $x_{cell}$ for a point is obtained by:

$$ x_{cell} = \left[ \frac{x}{k} \right]$$

The algorithm for calculating the forces on a particle hinges on the variation inverse square law, i.e

$$ F \propto \frac{1}{r^2} $$

where F is the force between any two chosen atoms and r is the separation between them.

As distance increases, the force decreases significantly. This means the forces beyond a certain distance, which we'll call the cutoff radius $r_{cutoff}$, are very small and can be neglected.

So, we choose the cell edge size $k$ such that the cutoff radius only extends to cells with cell indices 2 units away. 

This allows us to heavily optimize computation where we only calculate the forces of particles with the aforementioned cells instead of the those entire  box volume. This gives huge gains in performance as division is a very computationally expensive operation.

The force between two particles is approximated as:

$$ F \approx - k'\times (r - r_{0}) $$

where r is the distance between the two particles, $r_0$ is the natural interaction distance, and $k'$ is the force constant.

So for every step, for each particle, we select only the particles within 2 cells of the particle's cell.

We select and compute the forces exerted on the particles and sum them together, and use them to update the velocity of the particle.

The total potential energy is computed by summing the pairwise interaction energies of each particle with all other particles.

Sliders are given for $k'$ and $r_{cutoff}$.

## Possible improvements

The use of a Verlet neighbour list would speed up the process of computing forces incredibly.

There are also optimizations in the way NumPy is used here that would be incredibly helpful in both computation cost and speed.

There are also probably better ways to calculate the dynamics of the system, like the use of Verlet integration instead of the basic stepping algorithm used here.