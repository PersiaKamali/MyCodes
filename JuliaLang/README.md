# Approximating a Van der Pol oscillator with Gaussian-Mixture Low-Rank Networks

This project is a Julia implementation of the algorithm written in Sec. 6 of the paper by [M.Beiran et al](https://pnicompneurojc.github.io/papers/Beiran%202021.pdf). 

See also the original [code](https://github.com/emebeiran/low-rank2020/tree/main) written in Python.

## Brief Description

M. Beiran et al. show that a gaussian mixture network of rank R receiving a costant input is a universal approximator of R-dimensional dynamical systems.
In this project, we test this theory by using a 2-rank network to approximate the Van der Pol oscillator, a two-dimensional nonlinear dynamical system defined as

````math
\frac{dx}{dt} = y, \\
\frac{dy}{dt} = \mu(1-x^2)y - x
````

To do this we fix the following paramaters of the network:
* The number of neurons in the network, $N$;
* The number of gaussian populations in the network, $P$;
* The fraction of neurons in each population, $\bold{\alpha} = (\alpha_1, \alpha_2, ..., \alpha_P)$;
* Mean and variance of the right connectivity patterns $\textbf{m}^{(\bold{r})} = \{m_i^{(r)}\}_{i=1,...,N}$ for $r=1,2$ for each population, that is, respectively, $a_{m_r}^{(p)}$ and $\sigma_{I^2}^{(p)}$; 
* Mean and variance of the external input $I$, $a_I^{(p)}$ and $\sigma_{I^2}^{(p)}$.

The means of the connectivity patterns $\bold{m}^{(r)}$ and the mean of the input $I$ are assigned randomly by a uniform distribution in $(a,b)$, while the variances are sampled by an exponential distribution with paramater $\theta$.

Finally, the last parameters needed for the network are the mean and variance per population of the left connectivity patterns $\bold{n}^{(r)}$, for $r=1,2$, which are determined using linear regression, as described in Sec.6 of M. Beiran et al. A regularisation can be implemented by introducing a ridge parameter $\beta$.

Thus, $P$, $\bold\alpha$, $a_{m_r}^{(p)}$, $\sigma_{m_r^2}^{(p)}$, $a_I^{(p)}$, $\sigma_{I^2}^{(p)}$ and $\beta$ are hyperparameters of the algorithm that computes the left connectivy patterns. Of course, these can be tuned in order to attain the desired accuracy of approximating the targeted dynamical system.

The project is made of three main modules:

* `RunSimulation.jl`: runs the simulation by computing the path of the dynamical system in three ways: 
	1. Integrating the Van der Pol equations directly;
	2. Integrating equation 3.4 of M. Beiran et al. using the mean field parameters;
	3. Integrating equation 3.4 in M. Beiran et al, where the parameters are computed by averaging over the entire network.

* `FindNetworkPar.jl`: contains functions that execute the linear regression for computing the statistics of the left connectivity patterns and that sample from a multivariate gaussian distribution the pattern loadings $(\bold{m}_i, \bold{n}_i, I_i)$ for each neuron $i \in [N]$. It also contains functions that compute the energy and the vector field of the dynamical systems in two ways:
	1. Using equation 3.4 with the mean field parameters;
	2. Using equation 3.4 with parameters obtained by averaging over the entire network.
* `DynSys.jl`: contains the main functions for computing the Van der Pol vector field and energy.


## Example

By running the following commands in the`julia REPL`,

```julia
include("RunSimulation.jl")
```

The code will run the algorithm and output three graphs that display the three paths in $(X,Y)$ space computed in the different ways mentioned above.

## Considerations and Improvements

As one can observe by running the code, the mean field solution replicates fairly well the path of Van der Pol oscillator but the same cannot be said for the finite size solution. This could be due to finite size effects: one needs a larger number of neurons per population in order to start seeing mean field effects.

The code can be definitely improved: in particular, one could write it in a more general form such that one can easily re-use the `FindNetworkPar.jl` for any dynamical system of any size $R$. To do this it would be better to use julia `Matrices` instead of types like `Vector{Vector{Float64}}` or the `struct neuron` for the variables and re-write the code in order to access the matrices along columns. 
It could be intersting to find a block-matrix representation of the matrix $W$ in the`find_parameters`function, in order to speed up the inversion of the matrix, which is a potential bottleneck especially if it is very big. Also an additional improvement to the code would be to add some exception handling.  
