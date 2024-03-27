using DrWatson
@quickactivate "ElectrochemCO2ReductionGold"
using LiquidElectrolytes, ExtendableGrids, PreallocationTools, LessUnitful
using VoronoiFVM, ForwardDiff

@kwdef mutable struct ElectrolyteData <: AbstractElectrolyteData
    "Number of ionic species."
    nc::Int
	
    "Number of surface species"
    na::Int
	
    "Potential index in species list."
    iϕ::Int = nc + na + 1

    "Pressure index in species list"
    ip::Int = nc + na + 2

    "Mobility coefficient"
    D::Vector{Float64} = fill(2.0e-9 * ufac"m^2/s", nc)

    "Charge numbers of ions"
    z::Vector{Int}

    "Molar weight of solvent"
    M0::Float64 = 18.0153 * ufac"g/mol"

    "Molar weight of ions"
    M::Vector{Float64} = fill(M0, nc)

    "Molar volume of solvent"
    v0::Float64 = 1 / (55.4 * ufac"M")

    "Molar volumes of ions"
    v::Vector{Float64} = fill(v0, nc)

    "Solvation numbers"
    κ::Vector{Float64} = fill(0.0, nc)

    "Bulk ion concentrations"
    c_bulk::Vector{Float64} = fill(0.1 * ufac"M", nc)

    "Bulk voltage"
    ϕ_bulk::Float64 = 0.0 * ufac"V"

    "Bulk pressure"
    p_bulk::Float64 = 1.0e-2

    "Bulk boundary number"
    Γ_bulk::Int = 2

    "Working electrode voltage"
    ϕ_we::Float64 = 0.0 * ufac"V"

    "Working electrode  boundary number"
    Γ_we::Int = 1

    "Temperature"
    T::Float64 = (273.15 + 25) * ufac"K"

    "Molar gas constant scaled with temperature"
    RT::Float64 = ph"R" * T

    "Faraday constant"
    F::Float64 = ph"N_A*e"

    "Dielectric permittivity of solvent"
    ε::Float64 = 78.49

    "Dielectric permittivity of vacuum"
    ε_0::Float64 = ph"ε_0"

    "Pressure scaling factor"
    pscale::Float64 = 1.0e9

    "Local electroneutrality switch"
    eneutral::Bool = false

    """
    [Flux caculation scheme](@id fluxes)
    This allows to choose between
    - `:μex` (default): excess chemical potential (SEDAN) scheme, see [`sflux`](@ref)
    - `:act` : scheme based on reciprocal activity coefficients, see [`aflux`](@ref)
    - `:cent` : central scheme, see [`cflux`](@ref).
    """
    scheme::Symbol = :μex

    """
    Regularization parameter used in [`rlog`](@ref)
    """
    epsreg::Float64 = 1.0e-20

    """
    Species weights for norms in solver control.
    """
    weights::Vector{Float64} = [v..., zeros(na)..., 1.0, 1.0]
	
	"Number of reactions in the bulk"
	nr::Int64
	
	"Stoichiometric coefficients for the bulk reactions"
	γ::Matrix{Int64}

	"Stoichiometric coefficents of water for the bulk reactions"
	γH₂O::Vector{Int64}
	
	"Forward rate constants for the bulk reactions"
	kf::Vector{Float64}

	"Backward rate constants for the bulk reactions"
	kb::Vector{Float64}
	
	"relative chemical potential in the standard state"
	μ°::Vector{Float64}

	"Relative Chemical potential of water in the standard state"
	μ°H₂O::Float64

	"Adsorption rate constants"
	k°ₐ::Vector{Float64}

	"Symmetry factors for adsorbtion reactions"
	βₐ::Vector{Union{Missing, Float64}} = fill(missing, nc)

	"Kinetic barriers for the adsorbtion reactions"
	ℬₐ::Vector{Float64} = fill(0.0, nc)
	
	"Number of surface reactions"
	nrₛ::Int64

	"Stoichiometric coefficients in the surface reactions"
	γₛ::Matrix{Int64}
	
	"Stoichiometric coefficients of water for the surface reactions"
	γₛH₂O::Vector{Int64}
	
	"relative chemical potential in the standard state on the surface"
	μ°ₛ::Vector{Float64}

	"Surface rate constants"
	k°ₛ::Vector{Float64}
	
	"Symmetry factors for surface reactions"
	βₛ::Vector{Union{Missing, Float64}} = fill(missing, nrₛ)

	"Kinetic barriers for the surface reactions"
	ℬₛ::Vector{Float64} = fill(0.0, nrₛ)

	# not yet needed
	# "reference molar density of metal ions in the electrode"
	# cM°::Float64 = 5.86e28 / ph"N_A" * ufac"m"^(-3)
	
	# "molar density of free electrons in the electrode"
	# ce::Float64 = 1 * cM°

	"molar area of the metal ions"
	aM::Float64 = 7.1233e8
	
	cMₛ::Float64 = 1.42e8 * ufac"mol/cm^2"
	
	# "molar chemical potential of the free electrons in the metal electrode"
	# μe::Float64 = (3/(8*π))^(2/3) * ph"h"^2 / (2 * ph"m_e") * ph"N_A"^(5/3) * ce^(2/3)
	
	"molar surface chemical potential of the electrons on the metal electrode surface"
	μeₛ::Float64 = 4.5071 * ufac"eV" * ph"N_A"

	"number of available surface sites per metal atom"
	ωM::Int = 1
	
	"number of adsorption sites per molecule"
	ω::Vector{Int64} = fill(1, na)

	"hbond-adsorbate correction constants"
	hbond_consts::Vector{@NamedTuple{a::Float64, b::Float64}}
end

@enum ActivityType pressureconstrained latticeconstrained

const nc = 7
const na = 8

const nr = 5
const nrₛ= 2

begin # enumeration of the involved species
	const iK⁺ 		= 1
	const iH⁺ 		= 2
	const iHCO₃⁻ 	= 3
	const iCO₃²⁻ 	= 4
	const iCO₂ 		= 5
	const iOH⁻ 		= 6
	const iCO 		= 7
	const iCOOH 	= 8  # only in adsorbed phase, not bulk phase
end

const pH = 6.8

begin # geometrical parameters
	const hmin 		= 1.0e-6 	* ufac"μm"
	const hmax 		= 1.0  		* ufac"μm" 
	const nref  	= 0
	const L 		= 60.0 		* ufac"μm" 
	const Γ_we 		= 1
	const Γ_bulk 	= 2
end

begin # definition of charge numbers
	const z = zeros(Int64, nc)
	z[iK⁺] = 1
	z[iH⁺] = 1
	z[iHCO₃⁻] = -1
	z[iCO₃²⁻] = -2
	z[iOH⁻] = -1
	z[iCO] = 0
	z[iCO₂] = 0
end

begin # bulk reaction coefficents
	const γ = zeros(Int64, nr, nc)
	const γH₂O = zeros(Int64, nr)
	# CO₂ + OH⁻ <--> HCO₃⁻
	γ[1, iCO₂] = -1
	γ[1, iOH⁻] = -1
	γ[1, iHCO₃⁻] = 1
	

	# HCO₃⁻ + OH⁻ <--> CO₃²⁻ + H₂O
	γ[2, iHCO₃⁻] = -1
	γ[2, iOH⁻] = -1
	γ[2, iCO₃²⁻] = 1
	γH₂O[2] = 1

	# H₂O + CO₂ <--> HCO₃⁻ + H⁺
	γ[3, iCO₂] = -1
	γ[3, iH⁺] = 1
	γ[3, iHCO₃⁻] = 1
	γH₂O[3] = -1

	# HCO₃⁻ <--> CO₃²⁻ + H⁺
	γ[4, iHCO₃⁻] = -1
	γ[4, iH⁺] = 1
	γ[4, iCO₃²⁻] = 1

	# H₂O <--> H⁺ + OH⁻
	γ[5, iH⁺] = 1
	γ[5, iOH⁻] = 1
	γH₂O[5] = -1
end

begin # bulk rate constants
	const ke = [4.44e7 / (ufac"mol/dm^3"), 4.66e3 / (ufac"mol/dm^3"), 4.44e-7 * (ufac"mol/dm^3"), 4.66e-5 / (ufac"mol/dm^3"), 1.0e-14 * (ufac"mol/dm^3")^2]
	const kf = [5.93e3 / (ufac"mol/ dm^3 / s"), 1.0e8 / (ufac"mol / dm^3 / s"), 3.7e-2 / ufac"s", 59.44e3 / (ufac"mol / dm^3 / s"), 2.4e-5 * (ufac"mol / dm^3 / s")]
	const kb = kf ./ ke
end

begin # bulk diffusion coefficents
	const D = zeros(Float64, nc)
	D[iK⁺] = 9.310e-9
	D[iH⁺] = 1.957e-9
	D[iHCO₃⁻] = 1.185e-9
	D[iCO₃²⁻] = 0.923e-9
	D[iCO₂] = 1.91e-9
	D[iOH⁻] = 5.273e-9
	D[iCO] = 2.23e-9
end

begin # bulk concentrations
	const c_bulk = zeros(Float64, nc)
	c_bulk[iH⁺] = 10^(-pH) * ufac"mol / dm^3"
	c_bulk[iHCO₃⁻] = 0.091 * ufac"mol / dm^3"
	c_bulk[iCO₃²⁻] = 2.68e-5 * ufac"mol / dm^3"
	c_bulk[iCO₂] = 0.033 * ufac"mol / dm^3"
	c_bulk[iOH⁻] = 10^(pH - 14) * ufac"mol / dm^3"
	c_bulk[iCO] = 0.0001 * ufac"mol / dm^3"
	c_bulk[iK⁺] = -sum([c_bulk[i] * z[i] for i in 1:nc if i != iK⁺]) / z[iK⁺] # electroneutrality condition
end

begin # surface reaction coefficents
	const γₛ = zeros(Int64, nrₛ, na)
	const γₛH₂O = zeros(Int64, nrₛ)

	# CO2(ads) + H₂O(l) + e⁻ <-> COOH(ads) + OH⁻(aq)
	γₛ[1, iCO₂] = -1
	γₛ[1, iCOOH] = 1
	γₛ[1, iOH⁻] = 1
	γₛH₂O[1] = -1

	# COOH(ads) + H₂O(l) + e⁻ <-> CO(ads) + OH⁻(aq) + H₂O(l)
	γₛ[2, iCOOH] = -1
	γₛ[2, iCO] = 1
	γₛ[2, iOH⁻] = 1
	γₛH₂O[2] = 0 # !?

	# # CO2(ads) + H⁺ + e⁻ <-> COOH(ads)
	# γₛ[3, iCO₂] = -1
	# γₛ[3, iH⁺] = -1
	# γₛ[3, iCOOH] = 1

	# # COOH(ads) +  H⁺ + e⁻ <-> CO(ads) + H₂O(l)
	# γₛ[4, iCOOH] = -1
	# γₛ[4, iH⁺] = -1
	# γₛ[4, iCO] = 1
	# γₛH₂O[4] = 1
end

const k°ₛ = fill(1.0e13 / ufac"s", nrₛ)

const k°ₐ = fill(1.0e13 / ufac"s", nc)
k°ₐ[iCO] = 1.0e8 / ufac"s"

const βₛ = convert(Vector{Union{Float64, Missing}}, fill(missing, nrₛ))
βₛ[2] = 0.5

const βₐ = convert(Vector{Union{Float64, Missing}}, fill(missing, nc))

const S = 1.0e-5 / ph"N_A" * (1.0e10)^2 * ufac"mol / m^2"

const hbond_consts = [
	(; a = 0.0, b = 0.0),
	(; a = 0.0, b = 0.0),
	(; a = 0.0, b = 0.0),
	(; a = 0.0, b = 0.0),
	(; a = 0.0297720125 / (ufac"μA/cm^2"), b = -0.000286600929 / (ufac"μA/cm^2")^2),
	(; a = 0.0, b = 0.0),
	(; a = -0.00942574086 / (ufac"μA/cm^2"), b = -0.000189106972 / (ufac"μA/cm^2")^2),
	(; a = 0.00226896383 / (ufac"μA/cm^2"), b = -9.0295682e-05 / (ufac"μA/cm^2")^2),
]

const voltages = (-1.5:0.1:0.0)

function simulate(; μ°, μ°ₛ, μ°H₂O, μ°TS, κ, v, activitytype, p_bulk = 0.0, pscale = 1.0e9, hmin = 1.0e-6 * ufac"μm", max_round = 1000, maxiters = 100, reltol = 1.0e-10, abstol = 1.0e-10, tol_round = 1.0e-10, tol_mono = 1.0e-10, damp_initial = 0.1, damp_growth = 1.1)

	# grid
	grid =  let
		X = geomspace(0, L, hmin, hmax)
		simplexgrid(X)
	end

	# reaction
	#reaction(f, u, node, data) = reaction(f, u, node, data)

	# celldata
	celldata = let 
		ℬₛ = zeros(Float64, nrₛ)
		ℬₛ[2] = μ°TS - μ°ₛ[iCOOH] - μ°H₂O

		ℬₐ = zeros(Float64, nc)
		
		ElectrolyteData(; nc, na, z, c_bulk, D, μ°, μ°H₂O, κ, v, p_bulk, pscale, nr, γ, γH₂O, kf, kb, nrₛ, μ°ₛ, γₛ, γₛH₂O, k°ₛ, ℬₛ, βₛ, k°ₐ, ℬₐ, βₐ, hbond_consts)
	end


	cell = PNPSystem(grid; bcondition = halfcellbc, reaction, celldata, unknown_storage=:dense)

	solver_control = (; max_round, maxiters, reltol, abstol, tol_round, tol_mono, damp_initial, damp_growth)

	result = ivsweep(cell; voltages, store_solutions=true, solver_control...)
	return result
end



begin # reaction
	const 𝔞educts_cache = DiffCache(zeros(5), 17)
	const 𝔞products_cache = DiffCache(zeros(5), 17)
	const R_cache = DiffCache(zeros(5), 17)

	function reaction(f, u, node, data)
		(; iϕ, ip, p_bulk, pscale, nc, v0, v, nr, γ, γH₂O, kf, kb, μ°, μ°H₂O, RT) = data
		Tv = eltype(u)

		𝔞educts = get_tmp(𝔞educts_cache, u[iϕ])
		𝔞products = get_tmp(𝔞products_cache, u[iϕ])
		R = get_tmp(R_cache, u[iϕ])

		𝔞educts .= 1.0
		𝔞products .= 1.0
		R .= 0.0
		
		p = (u[ip]) * pscale - p_bulk
		@views c₀, cbar = c0_barc(u[1:nc], data)

		c = 1/(1-v[iK⁺]*(u[iK⁺]))
		for ic in 1:nc
			for ir in 1:nreactions(data)
				if γ[ir, ic] < 0
					𝔞educts[ir] *= (c * u[ic])^(-γ[ir, ic])
				elseif γ[ir, ic] > 0
					𝔞products[ir] *= (c * u[ic])^γ[ir, ic]
				end
			end
		end

		for ir in 1:NReactions
		# if γH₂O[ir] < 0
		# 	𝔞educts[ir] *= (c₀)^(-γH₂O[ir])
		# elseif γH₂O[ir] > 0
		# 	𝔞products[ir] *= (c₀)^(γH₂O[ir])
		# end
			R[ir] = kf[ir] * 𝔞educts[ir] - kb[ir] * 𝔞products[ir]
		end

		for ic in 1:nc # production rates of the dissolved
			for ir in 1:nr
				f[ic] -= γ[ir, ic] * R[ir]
			end
		end
		nothing
	end
end


begin # surface reaction
	const Nₛ_cache = DiffCache(zeros(7), 17)
	const Rₛ_cache = DiffCache(zeros(4), 17)
	const Δᵣμᶿₛ_cache = DiffCache(zeros(4), 17)
	const 𝔞ₛeducts_cache = DiffCache(ones(4), 17)
	const 𝔞ₛproducts_cache = DiffCache(ones(4), 17)

	function we_breaction(f,
		u::VoronoiFVM.BNodeUnknowns{Tval, Tv, Tc, Tp, Ti}, 
		bnode,
		data::ElectrolyteData
	) where {Tval, Tv, Tc, Tp, Ti}
		(; iϕ, ip, pscale, na, nc, nrₛ, μ°, μ°ₛ, γₛ, k°ₛ, k°ₐ, γₛH₂O, μ°H₂O, ℬₛ, ℬₐ, βₛ, βₐ, ω, z, hbond_consts, ϕ_we, RT, v0, v, p_bulk) = data

		Nₛ = get_tmp(Nₛ_cache, u[iϕ]) # normal surface flux due to adsorption
		Rₛ = get_tmp(Rₛ_cache, u[iϕ]) # surface reaction rates
		Δᵣμᶿₛ = get_tmp(Δᵣμᶿₛ_cache, u[iϕ])
		𝔞ₛeducts = get_tmp(𝔞ₛeducts_cache, u[iϕ])
		𝔞ₛproducts = get_tmp(𝔞ₛproducts_cache, u[iϕ])

		Nₛ .= 0.0 # normal surface flux due to adsorption
		Rₛ .= 0.0 # surface reaction rates
		Δᵣμᶿₛ .= 0.0
		𝔞ₛeducts .= 1.0
		𝔞ₛproducts .= 1.0

		@views c0, cbar = c0_barc(u[1:nc], data)
		@views cVₛ, cbarₛ = cVₛ_barcₛ(u[nc+1:nc+na], data)

		p = (u[ip]) * pscale - p_bulk
		σ = 0.0 #ForwardDiff.value(f[iϕ]) #20.0 * ufac"μA"/ufac"cm"^2
		
		for ia in 1:na
			(; a, b) = hbond_consts[ia]
			Δμσ = (a * σ + b * σ^2) * ph"e * N_A"
			𝔞ₛ = (ia == 2 | ia == 6) ? u[ia] / cbar : u[nc + ia] / cbarₛ * (cbarₛ / cVₛ)^ω[ia]
			for ir in 1:nrₛ
				Δᵣμᶿₛ[ir] += (ia == 2 | ia == 6) ? γₛ[ir, ia] * (μ°[ia] + v[ia] * p) : γₛ[ir, ia] * (μ°ₛ[ia] + Δμσ)
				if γₛ[ir, ia] < 0
					𝔞ₛeducts[ir] *= 𝔞ₛ^(-γₛ[ir, ia])
				elseif γₛ[ir, ia] > 0
					𝔞ₛproducts[ir] *= 𝔞ₛ^γₛ[ir, ia]
				end
			end
			
			if ia <= nc # adsorption rates
				𝔞ₛ = u[nc + ia] / cbarₛ * (cbarₛ / cVₛ)^ω[ia]
				𝔞 = u[ia] / cbar
				Δᵣμᶿₐ = μ°ₛ[ia] + Δμσ - μ°[ia] - v[ia] * p
				β = ismissing(βₐ[ia]) ? (Δᵣμᶿₐ > 0 ? 1.0 : 0.0) : βₐ[ia]
				Nₛ[ia] = k°ₐ[ia] * (exp(-1 / RT * (ℬₐ[ia] + β * Δᵣμᶿₐ)) * 𝔞 
								-exp(-1 / RT * (ℬₐ[ia] - (1-β) * Δᵣμᶿₐ)) * 𝔞ₛ) 
			end
		end

		for ir in 1:nrₛ
			Δᵣμᶿₛ[ir] += γₛH₂O[ir] * (μ°H₂O + v0 * p)
			if γₛH₂O[ir] > 0
				𝔞ₛproducts[ir] *= (c0/cbar)^γₛH₂O[ir]
			elseif γₛH₂O[ir] < 0
				𝔞ₛproducts[ir] *= (c0/cbar)^(-γₛH₂O[ir])
			end
			β = ismissing(βₛ[ir]) ? (Δᵣμᶿₛ[ir] > 0 ? 1.0 : 0.0) : βₛ[ir]
			
			Rₛ[ir] = k°ₛ[ir] * (
				exp(-1/RT * (ℬₛ[ir] + β * Δᵣμᶿₛ[ir])) * 𝔞ₛeducts[ir]
			-exp(-1/RT * (ℬₛ[ir] - (1-β) * Δᵣμᶿₛ[ir])) * 𝔞ₛproducts[ir]
			)
		end

		Rₛ[2] *= c0/cbar
		
		for ia in 1:na # production rates of the surface species
			if (ia == 2 | ia == 6)
				for ir in 1:nrₛ
					f[ia] -= γₛ[ir, ia] * Rₛ[ir]
				end
			else
				for ir in 1:nrₛ
					f[nc + ia] 	-= γₛ[ir, ia] * Rₛ[ir]
				end
			end
			if ia <= nc
				f[nc + ia] 	-= Nₛ[ia]
				f[ia] 		-= -Nₛ[ia]
			end
		end
	end

end

function halfcellbc(
	f,
	u::VoronoiFVM.BNodeUnknowns{Tval, Tv, Tc, Tp, Ti}, 
	bnode,
	data
) where {Tval, Tv, Tc, Tp, Ti}
	(; Γ_we, Γ_bulk, ϕ_we, iϕ) = data

	boundary_dirichlet!(f, u, bnode; species = iϕ, region = Γ_we, value = ϕ_we)
	bulkbcondition(f, u, bnode, data; region = Γ_bulk)
	if bnode.region == Γ_we
		we_breaction(f, u, bnode, data)
	end
	nothing
end


begin ################### Utility Functions ###########################
	function cVₛ_barcₛ(cₛ, data)
		(; ω, ωM, aM) = data
		cVₛ = ωM / aM - sum(ω .* cₛ)
		c̅ₛ  = sum(cₛ) + cVₛ
		cVₛ, c̅ₛ
	end

	function surface_chemical_potential(cₛ, c̅ₛ, cVₛ, ω, data)
		(; RT) = data
		RT * rlog(cₛ/c̅ₛ, data) - ω * RT * rlog(cVₛ/c̅ₛ, data)
	end

	function surfacefraction(ia, cₛ, data)
		(; na, ω, ωM, aM) = data
		cVₛ = ωM / aM - sum(ω .* cₛ)
		c̅ₛ  = sum(cₛ) + cVₛ
		ia == 0 ? cVₛ / c̅ₛ : cₛ[ia] / c̅ₛ
	end
end