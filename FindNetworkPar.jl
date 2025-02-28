module FindNetworkPar

using QuadGK
using LinearAlgebra, Statistics
using Random, Distributions

export neuron, MF_field, Net_field, create_grid_2D, find_parameters, get_network_parameters, MF_solution!, Num_solution!, create_grid_2D_2

mutable struct neuron
	m::Vector{Float64}
	n::Vector{Float64}
	I::Float64
end

function Phi(mu, delta)
	f(x)  = tanh(mu + sqrt(delta)*x)*exp(-x^2/2)*1/sqrt(2*pi)
	integral, err = quadgk(f, -Inf, Inf)
	return integral
end

function Phi_prime(mu, delta) 
	g(x)  = sqrt(delta)*(1-tanh(mu + sqrt(delta)*x)^2)*exp(-x^2/2)*1/sqrt(2*pi)
	integral, err = quadgk(g, -Inf, Inf)
	return integral
end

function VP_field(x, y, mmu=1.)
	G_x = y
	G_y = mmu*(1 - x^2)*y - x
	return [G_x, G_y]
end

function MF_path(sol, muM, muN, muI, sigMN, sigM, sigI, alpha)
	
	s = zeros(size(sol))
	numPop = length(alpha) 
	for iP in 1:numPop
		mu = muI[iP] + dot(muM[iP], sol)
		delta = sigI[iP] + dot(sigM[iP], sol.^2)
		P1 =  Phi_prime(mu, delta)
		P0 = muN[iP].*Phi(mu,delta)       
		s .+= alpha[iP].*(P0 .+ (sigMN[iP]*sol).*P1)
	end

	s .= s .- sol
	return s
end

function MF_field(xs, ys, muM, muN, muI, sigMN, sigM, sigI, alpha)
	
	size = length(xs)
	U = zeros(size,size)
	V = zeros(size,size)
	E = zeros(size,size)
	numPop = length(alpha)

	for (ix, x) in enumerate(xs)
		for (iy, y) in enumerate(ys)
			v = zeros(2)
			for iP in 1:numPop
				Mu = muI[iP] + x*muM[iP][1] + y*muM[iP][2]
				Delta = sigI[iP] + sigM[iP][1]*x^2 + sigM[iP][2]*y^2
                		P1 =  Phi_prime(Mu, Delta)
				P0 = muN[iP].*Phi(Mu,Delta)
				v .+= alpha[iP].*(P0 .+ sigMN[iP]*[x,y].*P1)
			end
			U[ix,iy] = v[1] - x
			V[ix, iy] = v[2] - y
		end
	end
	E .= sqrt.(U.^2 +V.^2)
	return U, V, E
end

function Net_path(sol, nn, N, alpha)
	
	numPop = size(alpha,1)
	nP = Int.(alpha.*N)
	s = zeros(2)
	s[1] = mean( nn[i][j].n[1]*tanh(nn[i][j].I + sol[1]* nn[i][j].m[1] + sol[2]*nn[i][j].m[2])  for i in 1:numPop for j in 1:nP[i])
	s[2] = mean( nn[i][j].n[2]*tanh(nn[i][j].I + sol[1]*nn[i][j].m[1] + sol[2]*nn[i][j].m[2]) for i in 1:numPop for j in 1:nP[i])
	s[1] = s[1] -sol[1]
	s[2] = s[2] -sol[2]
	return s
end

function Net_field(xs, ys, nn, N, numPop, alpha)

	size = length(xs)
	U = zeros(size, size)
	V = zeros(size, size)
	E = zeros(size, size)
	nP = Int.(alpha.*N)

	for (ix, x) in enumerate(xs)
		for (iy, y) in enumerate(ys)
			
			U[ix, iy] = - x + mean( nn[i][j].n[1]*tanh(nn[i][j].I + x * nn[i][j].m[1] + y * nn[i][j].m[2]) for i=1:numPop for j=1:nP[i] )
            		
			V[ix, iy] = - y + mean(nn[i][j].n[2]*tanh(nn[i][j].I + x * nn[i][j].m[1] + y * nn[i][j].m[2]) for i=1:numPop for j=1:nP[i])
		end
	end
	E .= sqrt.(U.^2+V.^2)
	return U, V, E
end

function create_grid_2D(num_points, x_min, x_max, y_min, y_max)
	
	data_points = [ Vector{Float64}(undef, 2) for _ = 1:num_points^2 ]

	vx = LinRange(x_min, x_max, num_points)
	vy = LinRange(y_min, y_max, num_points)
	
	n = 0
	for x in vx
		for (i, y) in enumerate(vy)
			data_points[i+n] = [x, y]
		end
		n += num_points
	end
	return data_points
end

function create_grid_2D_2(num_pointsaxes, xmin, xmax, ymin, ymax)

	return Base.Iterators.product(range(xmin, xmax, num_pointsaxes), range(ymin, ymax, num_pointsaxes))
end

#function that implements the algorithm for computing the left pattern's statistics
function find_parameters(data_points, mM, mI, sigM, sigI, beta, alpha::Vector)
	
	num_pop = length(alpha) 
	r = size(mM, 1)
	num_points = length(data_points)
	
	rr = r*(r+1)

	G = zeros(r*num_points)
	W = zeros(r*num_points, rr*num_pop)

	for (i,p) in enumerate(data_points)
 		
		ip_s = 1 + (i - 1)*r
		ip_e = ip_s + r - 1
			
		G[ip_s:ip_e] .= VP_field(p[1], p[2]) .+ p
		
		for s in 1:num_pop

			mu = mI[s] + sum(mM[:, s].*p) 
			delta = sigI[s] + sum(sigM[:,s].*p.^2)
			
			phi0 = Phi(mu, delta)
			phi1 = Phi_prime(mu, delta)

			is_s = 1 + (s - 1)*rr
			is_e = is_s + rr - 1
			
			W[ip_s:ip_e, is_s:is_e] .= alpha[s]*[(phi0*I)(r) [phi1*collect(p)'; zeros(r)'] [zeros(r)'; phi1*p'] ]
		end
	end
	
	sol = W'*G
	C = W'*W + beta^2*I(rr*num_pop)
	X =  inv(C) * W' * G
	
	sigMN = [ Matrix{Float64}(undef, r, r) for _ = 1:num_pop ]
	mN = [ Vector{Float64}(undef, r) for _ = 1:num_pop ]

	for i in 1:num_pop

		i_sN = 1 + (i-1)*rr 
		i_eN = i_sN + r - 1
		
		i_sS = i_eN + 1 
		i_eS = i_sN + rr - 1

		mN[i] .= X[i_sN:i_eN]
		sigMN[i] .= reshape(X[i_sS:i_eS], r, r)'
	end

	return mN, sigMN 
end


function find_parameters(data_points, mM, mI, sigM, sigI, beta, alpha=1.)
	 
	r = length(mM)
	num_points = length(data_points)
	
	rr = r*(r+1)

	G = Vector{Float64}(undef, r*num_points)
	W = Matrix{Float64}(undef, r*num_points, rr)
	
	for (i,p) in enumerate(data_points)
 		
		ip_s = 1 + (i - 1)*r
		ip_e = ip_s + r - 1
		
		G[ip_s:ip_e] = VP_field(p[1], p[2]) + p
		
		mu = mI + dot(mM, p) 
		delta = sigI + dot(sigM, p.^2)
			
		phi0 = Phi(mu, delta)
		phi1 = Phi_prime(mu, delta)
		
		W[ip_s:ip_e, :] = alpha.*[(phi0*I)(r) [phi1.*p'; zeros(r)'] [zeros(r)'; phi1.*p'] ]
	end
	
	sol = W'*G
	C = W'*W + (beta^2*I)(rr)
	X =  inv(C) * W' * G
	
	sigMN = Matrix{Float64}(undef, r, r)
	mN = Vector{Float64}(undef, r)
	
	mN = X[1:r]
	sigMN = reshape(X[(r+1):rr], r, r)'

	return mN, sigMN 
end

#function that samples the loading patterns of each neuron
function get_network_parameters(mM, mN, mI, sigMN, sigM, sigI, alpha, N)

	r = length(mM[1])
	num_Pop = length(alpha)

	bigSig = zeros(2r+1, 2r+1)
	
	numNP = Int.(alpha.*N)
	neuron_network = []
	muMP = [ zeros(r) for _=1:num_Pop ]
	muNP = [ zeros(r) for _=1:num_Pop ]
	muIP = zeros(num_Pop)
	
	bigMean = zeros(2r+1)
	bigSig_P = [ zeros(2r+1, 2r+1) for _=1:num_Pop ]
	sigMP = [ zeros(r) for _=1:num_Pop ]
	sigNP = [ zeros(r) for _=1:num_Pop ]
	sigIP  = zeros(num_Pop)
	sigMNP = [ zeros(r,r) for _=1:num_Pop ]

	for j in 1:num_Pop
		neuron_network = [ [ neuron(Vector{Float64}(undef, r), Vector{Float64}(undef, r), 0.0) for _=1:numNP[j] ] for _ = 1:num_Pop ]
	end

	for i in 1:num_Pop

		bigSig[end, end] = sigI[i]
		
		diagS = diagm([ sigM[i]; fill(1.1, r) ])
		Binds = CartesianIndices(diagS)
		Ainds = CartesianIndices((1:2r, 1:2r))

		copyto!(bigSig, Ainds, diagS, Binds)

		Binds = CartesianIndices(sigMN[i])
		Ainds = CartesianIndices((1:r,(r+1):(2r)))
 		
		copyto!(bigSig, Ainds, sigMN[i], Binds)

		bigSig_S = Symmetric(bigSig)
		bigSig .= bigSig_S

		#make bigSig positive definite as much possible
		cnt1 = 0
		val = minimum(eigvals(bigSig))
		
		while val<1e-07
			if cnt1<200
				diagN = 1.2.*diag(bigSig[(r+1):2r, (r+1):2r]) 
				bigSig[diagind(bigSig)[(r+1):2r]] .= diagN 	

			else
				diagN = 1.02.*diag(bigSig[(r+1):2r, (r+1):2r])
				bigSig[diagind(bigSig)[(r+1):2r]] .= diagN
			end
				
			val = minimum(eigvals(bigSig))
			cnt1 += 1

		end

		bigMean .= [ mM[i]; mN[i]; mI[i] ]
		error = 1e8
		cnt2 = 0
		sample_save = zeros(size(bigMean,1),numNP[i])
		rs_corr = similar(bigSig)
		dif = similar(bigSig)
		while error>0.3 && cnt2 < 500 

			cnt2 += 1

			sample = rand(MvNormal(bigMean, bigSig), numNP[i])
			mean_s = mean(sample, dims=2)
			sig_s =  cov(sample, dims=2) 
			
			rs_corr .= bigSig
			rs_corr[abs.(rs_corr).<1e-10] .= 1e-10
			dif .= (sig_s .- rs_corr)./rs_corr
			
			dif[abs.(dif).>1e8] .= 0.
			dif[diagind(dif)[(r+1):2r]] .= zeros(r)
			
			error2 = std(dif) + std(bigMean .- mean_s)
			if error2<error
				error=error2
				sample_save .= sample
			end
		end
		for j in 1:numNP[i]
			#saving neuron patterns into the network
			neuron_network[i][j].m .= sample_save[1:r,j] 
			neuron_network[i][j].n .= sample_save[(r+1):2r, j] 
			neuron_network[i][j].I = sample_save[2r+1, j] 
			
			#computing network statistics
			muMP[i] .+= sample_save[1:r, j]
			muNP[i] .+= sample_save[(r+1):2r, j]
			muIP[i] += sample_save[2r+1, j]
			
 		end
		
		muMP[i] .= muMP[i] ./ numNP[i]
		muNP[i] .= muNP[i] ./ numNP[i]
		muIP[i] = muIP[i]/numNP[i]

		corr = cov(sample_save, dims=2)
		bigSig_P[i] .= corr
		sigMP[i] .= diag(corr)[1:r]
		sigNP[i] .= diag(corr)[(r+1):2r]
		sigIP[i] = corr[end,end]
		sigMNP[i] .= corr[1:r, (r+1):2r]
	end
	return neuron_network, muMP, muNP, muIP, sigMNP, sigNP, sigMP, sigIP, bigSig_P
end

#functions that integrate the path
function MF_solution!(sol, muM, muN, muI, sigMN, sigM, sigI, alpha, dt)
	
	tot_time = size(sol,1)

	for it in 1:(tot_time-1)
		sol[it+1, :] .= sol[it,:] .+ dt.*MF_path(sol[it,:], muM, muN, muI, sigMN, sigM, sigI,alpha)
	end
	return sol
end

function Num_solution!(sol, neuron_network, N, dt, alpha)

	tot_time = size(sol,1)
	
	for it in 1:(tot_time-1)
		sol[it+1, :] .=  sol[it,:] .+ dt.*Net_path(sol[it,:], neuron_network, N, alpha) 
	end
	return sol
end

end


