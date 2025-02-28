module DynSys

function VanderPol(v, mmu=1.)

	g_x = v[2]
	g_y = mmu*(1 - v[1]^2) * v[2] - v[1]
	
	return [g_x, g_y]
end

function VP_field( xs, ys, mmu=1.0)
	X, Y = Iterators.product(xs, ys) |> collect |> x->(getindex.(x, 1), getindex.(x, 2))
	
	size = length(xs)
	U = zeros(size,size)
	V = zeros(size,size)
	
	for (ix, x) in enumerate(xs)
		for (iy, y) in enumerate(ys)
			U[ix, iy] = y
			V[ix, iy] = -x + mmu*(1-x^2)*y
		end
	end
	E = sqrt.(U.^2+V.^2)
	return X, Y, E, U, V
end

end
