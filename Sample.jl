module Sample

using Random, Distributions
export sample_parameters

function sample_parameters(P, seed)
	
	muM = rand(Uniform(-2.,2.), 2, P)
	muI = rand(Uniform(-2.,2.), P)

	sigmaM = rand(Exponential(1.), 2, P)
	sigmaI = rand(Exponential(1.), P)

	return muM, muI, sigmaM, sigmaI
end
end
