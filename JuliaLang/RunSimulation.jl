module RunSimulation

include("FindNetworkPar.jl")
include("DynSys.jl")

using .FindNetworkPar
using .DynSys: VanderPol, VP_field
using Random, Distributions
using CairoMakie, Interpolations

#Network Parameters
N = 30000
R = 2
P = 15
alpha = fill(2000/N, P)

#Simulation Parameters
dt = 0.02
tot_time = range(0, stop=40, step=dt)
nsteps = length(tot_time)
seed = 123
beta = 0.5
num_points = 30
xmin, xmax = -3, 3
ymin, ymax = -3, 3
xs = range(-4, 4, length=100)
ys = range(-4, 4, length=100)

#parameters for hyperparamaters
a, b = -2., 2.
theta_sM = 1. 
theta_sI = 1.

#the solutions of the dynamics
solT = zeros(nsteps, R)
solMF = zeros(nsteps, R)
solN = zeros(nsteps, R)

solT0 = [1.,1.]
solMF0 = [1.,1.]
solN0  = [1.,1.]

solT[1,:] = solT0
solMF[1,:] = solMF0
solN[1,:] = solN0

#Set the seed
Random.seed!(seed)

#sample hyperparameters: muM, muI, sigmaM, sigmaI
muM = [ rand(Uniform(a,b), R) for _=1:P]
muI =  rand(Uniform(a,b), P)
sigmaM = [rand(Exponential(theta_sM), R) for _=1:P]
sigmaI = rand(Exponential(theta_sI), P)

#get set points
data_points = create_grid_2D(num_points, xmin, xmax, ymin, ymax)

#compute parameters: muN, sigmaMN
muN, sigmaMN = find_parameters(data_points, muM, muI, sigmaM, sigmaI, beta, alpha)
#compute network
neuron_network, muMP, muNP, muIP, sigMNP, sigNP, sigMP, sigIP, bigSigP = get_network_parameters(muM, muN, muI, sigmaMN, sigmaM, sigmaI, alpha, N)

#get fields

mfU, mfV, mfE = MF_field(xs, ys, muM, muN, muI, sigmaMN, sigmaM, sigmaI, alpha)
nU, nV, nE = Net_field(xs, ys, neuron_network, N, P, alpha)
X, Y, E, U, V = VP_field(xs, ys)

#get paths
#VanderPolSlt
for it in 1:(nsteps-1)
	solT[it+1,:] .= solT[it,:] + dt.*(VanderPol(solT[it,:]))
end
MF_solution!(solMF, muM, muN, muI, sigmaMN, sigmaM, sigmaI, alpha, dt)
Num_solution!(solN, neuron_network, N, dt, alpha)



###################################PLOTTING RESULTS##############################################

#solT
function VanderPol2D(x,y)         
	return Point2f(y, 1.0*(1-x^2)*y - x)
end 

f1 = Figure()
ax = Axis(f1[1,1],
	limits = (minimum(xs), maximum(xs), minimum(ys), maximum(ys)),
	title= "Dynamics of the Van der Pol oscillator",
	xlabel = "x",
	ylabel = "y")
heatmap!(ax, xs, ys, log10.(E))
streamplot!(ax, VanderPol2D, xs, ys)
Colorbar(f1[1,2], limits= (-2, 2), colormap = :viridis, label = "log10(Energy)")
lines!(ax, solT[1200:end, 1], solT[1200:end, 2], color= :red, linewidth=1)
rect = Rect(-3, -3, 6, 6)
poly!(ax, rect, color=:transparent, strokecolor=:red, strokewidth=2)

#solMF
function splot_VanderPol(u,v, axi)
	nx, ny = size(u)
	x, y = 1:nx, 1:ny
	intu, intv = linear_interpolation((x,y), u), linear_interpolation((x,y), v)
	f(x) = Point2f(intu(x...), intv(x...))
	return streamplot!(axi, f, x, y, colormap=:magma)
end

f2 = Figure()
ax2 = Axis(f2[1,1],
	limits = (minimum(xs), maximum(xs), minimum(ys), maximum(ys)),
	title = "Mean Field slt of Van der Pol oscillator",
	xlabel = "x",
	ylabel = "y")
heatmap!(ax2, xs, ys, log10.(mfE))
#splot_VanderPol(mfU, mfV, ax2)
Colorbar(f2[1,2], limits=(-2,2), colormap= :viridis, label = "log10(Energy)")
lines!(ax2, solMF[:, 1], solMF[:,2], color= :red, linewidth=1, label = "Mean field path")
lines!(ax2, solT[1200:end,1], solT[1200:end,2], color= :red, linewidth=1, linestyle = :dash, label = "Van der Pol")
axislegend(position = :rb) 

#solN
function VanderPol_NN(x, y)
	ix = findfirst(isequal(x), xs)
	iy = findfirst(isequal(y), ys)
	return Point2f(nU[ix,iy], nV[ix,iy])
end

f3 = Figure()
ax3 = Axis(f3[1,1],
	limits = (minimum(xs), maximum(xs), minimum(ys), maximum(ys)),
	title = "Finite size slt of Van der Pol oscillator",
	xlabel = "x",
	ylabel = "y")
heatmap!(ax3, xs, ys, log10.(nE))
#splot_VanderPol(nU, nV, ax3)
Colorbar(f3[1,2], limits=(-2,2), colormap= :viridis, label = "log10(Energy)" )
lines!(ax3, solN[:,1], solN[:,2], color= :red, linewidth=1, label = "Finite size solution")
lines!(ax3, solT[1200:end,1], solT[1200:end,2], color= :red, linewidth=1, linestyle = :dash, label = "Van der Pol")
Colorbar(f3[1,2], limits=(-2,2), colormap= :viridis)
axislegend(position = :rb) 



display(f1)
display(f2)
display(f3)

end
