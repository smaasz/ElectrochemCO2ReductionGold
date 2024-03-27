using DrWatson
@quickactivate :ElectrochemCO2ReductionGold
using ExtendableGrids
using Plots, LessUnitful, Printf

species  = Vector{@NamedTuple{name::String, color::Symbol}}(undef, nc)
species[iK⁺]    = (; name = "K⁺", color = :orange)
species[iH⁺]    = (; name = "H⁺", color = :gray)
species[iHCO₃⁻] = (; name = "HCO₃⁻", color = :brown)
species[iCO₃²⁻] = (; name = "CO₃²⁻", color = :violet)
species[iCO₂]   = (; name = "CO₂", color = :red)
species[iOH⁻]   = (; name = "OH⁻", color = :green)
species[iCO]    = (; name = "CO", color = :blue)

function cplot(solutiongrid, solutions, v, iv)        
    title = @sprintf("Φ_we=%+1.2f [V vs. SHE]", v)
    p = plot(; title, xlabel = "Distance from electrode [μm]", ylabel = "Concentration [mol / dm³]", xscale = :log10, yscale = :log10, xlimits = (1.0e-6, 60), ylimits=(1.0e-10, 1.0e2))
    addcplot!(p, solutiongrid, solutions, v, iv)
end

function addcplot!(p, solutiongrid, solutions, v, iv)
    for (ic, solutionrow) in enumerate(eachrow(solutions[iv][1:nc, :])) 
        (; name, color) = species[ic]
        plot!(p, solutiongrid[XCoordinates] ./ ufac"μm" .+ 1.0e-14, solutionrow ./ ufac"mol / dm^3"; color, label=name)
    end
end

function cmovie(solutiongrid, solutions, vs)
    anim = @animate for (iv, v) in enumerate(vs)
        title = @sprintf("Φ_we=%+1.2f [V vs. SHE]", v)
        p = plot(; title, xlabel = "Distance from electrode [μm]", ylabel = "Concentration [mol / dm³]", xscale = :log10, yscale = :log10, xlimits = (1.0e-6, 60), ylimits=(1.0e-10, 1.0e2))
        addcplot!(p, solutiongrid, solutions, v, iv)
    end every 1
    return anim
end


for datafile in readdir(datadir("sims", "solvationtest"))
    d = wload(datadir("sims", "solvationtest", datafile))
    solutiongrid = d["solutiongrid"]
    
    # plot IV curve
    p = plot(d["voltages"], [j[iOH⁻] ./ S .* ph"N_A * e" ./ ufac"mA / cm^2" for j in d["j_we"]], title = "IV Sweep", xlabel = "Voltage [V]", ylabel = "Current [mA / cm²]")
    savefig(p, plotsdir(splitext(datafile)[1] * ".png"))

    # animation of concentrations as functions of the distance to the electrode
    anim = cmovie(solutiongrid, d["solutions"], d["voltages"])
    gif(anim, plotsdir(splitext(datafile)[1] * ".gif"); fps = 10)
end

