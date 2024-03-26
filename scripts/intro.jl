using DrWatson
@quickactivate :ElectrochemCO2ReductionGold
using LessUnitful
using JLD2


const μ° = zeros(Float64, 1, nc)
μ°[:, iK⁺]     = [0.0         ] .* ph"N_A"
μ°[:, iH⁺]     = [-2.94939e-21] .* ph"N_A"
μ°[:, iHCO₃⁻]  = [0.0         ] .* ph"N_A"
μ°[:, iCO₃²⁻]  = [0.0         ] .* ph"N_A"
μ°[:, iCO₂]    = [-4.12434e-20] .* ph"N_A"
μ°[:, iOH⁻]    = [1.70899e-20 ] .* ph"N_A"
μ°[:, iCO]     = [-1.89406e-20] .* ph"N_A"

const μ°ₛ = zeros(Float64, 1, na)
μ°ₛ[:, iK⁺]    = [1.0e-17     ] .* ph"N_A"
μ°ₛ[:, iH⁺]    = [1.0e-17     ] .* ph"N_A"
μ°ₛ[:, iHCO₃⁻] = [1.0e-17     ] .* ph"N_A"
μ°ₛ[:, iCO₃²⁻] = [1.0e-17     ] .* ph"N_A"
μ°ₛ[:, iCO₂]   = [1.47112e-19 ] .* ph"N_A"
μ°ₛ[:, iOH⁻]   = [1.0e-17     ] .* ph"N_A"
μ°ₛ[:, iCO]    = [-1.43503e-21] .* ph"N_A"
μ°ₛ[:, iCOOH]  = [7.52748e-20 ] .* ph"N_A"

const μ°H₂O    = [1.41405e-20] .* ph"N_A"
const μ°TS     = [1.95795e-19] .* ph"N_A"

const κ = zeros(Float64, 4, nc)
κ[:, iK⁺]      = [10    , 10    , 10, 10]
κ[:, iH⁺]      = [15    , 10    , 20, 25]
κ[:, iHCO₃⁻]   = [10    , 10    , 10, 10] 
κ[:, iCO₃²⁻]   = [10    , 10    , 10, 10]
κ[:, iCO₂]     = [0     , 0     , 0,  0]
κ[:, iOH⁻]     = [10    , 10    , 10, 10]
κ[:, iCO]      = [0     , 0     , 0,  0]

const v = zeros(Float64, 1, nc)
v[:, iK⁺]      = [0.20     ] ./ (55.4 * ufac"M")
v[:, iH⁺]      = [0.07    ] ./ (55.4 * ufac"M")
v[:, iHCO₃⁻]   = [2.0      ] ./ (55.4 * ufac"M")
v[:, iCO₃²⁻]   = [2.0      ] ./ (55.4 * ufac"M")
v[:, iCO₂]     = [1.93     ] ./ (55.4 * ufac"M")
v[:, iOH⁻]     = [0.07     ] ./ (55.4 * ufac"M")
v[:, iCO]      = [1.81     ] ./ (55.4 * ufac"M")


function test_κdependence()
    
    ds = dict_list(Dict(
            :μ°    => [μ°[1,:]],
            :μ°ₛ   => [μ°ₛ[1,:]],
            :μ°TS  => μ°TS[1],
            :μ°H₂O => μ°H₂O[1],
            :v     => [v[1, :]],
            :κ     => collect(eachrow(κ[4:4,:])),
            :activitytype => pressureconstrained,
    ))

    map(enumerate(ds)) do (i, d)
        solutiongrid, r = simulate(; d...)
        d[:voltages] = r.voltages
        d[:j_we] = r.j_we
        d[:solutions] = r.solutions
        d[:solutiongrid] = solutiongrid
        wsave(datadir("sims", "solvationtest_4.jld2"), d)
    end
    readdir(datadir("sims"))
end

test_κdependence()

