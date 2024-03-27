#module DiffusiveResolvedMTK
using ModelingToolkit, DynamicQuantities
using IfElse
import ChainRulesCore

using ModelingToolkit: t, D, scalarize
using DifferentialEquations
using ExtendableGrids: simplexgrid


######### Some functions ###################

C0_func(C, v, v0, κ) = 1 / v0 - sum([(vi / v0 + κi) * Ci for (Ci, vi, κi) in zip(C, v, κ)])

rlog(x; eps_reg) = IfElse.ifelse(x < eps_reg, log(eps_reg) + (x - eps_reg) / eps_reg, log(x))

dμex_func(γk, γl, RT, eps_reg) = IfElse.ifelse(γk > γl, rlog(γk / γl; eps_reg) * RT, -rlog(γl / γk; eps_reg) * RT)

function γfunc(p, p°, c0, cbar, M, M0, v, v0, κ, RT) 
    Mrel = M / M0
    vbar = v + κ * v0
    vrel = vbar - Mrel * v0
    return exp(vrel * (p - p°) / (RT)) * (cbar / c0)^Mrel# * (1/cbar)
end

function bernoulli_horner(x)
    y = x / 47_900_160
    y = x * y
    y = x * (-1 / 1_209_600 + y)
    y = x * y
    y = x * (1 / 30_240 + y)
    y = x * y
    y = x * (-1 / 720 + y)
    y = x * y
    y = x * (1 / 12 + y)
    y = x * (-1 / 2 + y)
    y = 1 + y
end

## Bernoulli thresholds optimized for Float64
const bernoulli_small_threshold = 0.25
const bernoulli_large_threshold = 40.0

function fbernoulli_p(x)
    return IfElse.ifelse(
        x < -bernoulli_large_threshold,
        -x,
        IfElse.ifelse(
            x > bernoulli_large_threshold,
            0.0,
            IfElse.ifelse(
                abs(exp(x) - 1.0) > bernoulli_small_threshold,
                x / (exp(x) - 1.0),
                bernoulli_horner(x)
            )
        )
    )
end
function fbernoulli_n(x)
    return IfElse.ifelse(
        x < -bernoulli_large_threshold,
        0.0,
        IfElse.ifelse(
            x > bernoulli_large_threshold,
            x,
            IfElse.ifelse(
                abs(exp(x) - 1.0) > bernoulli_small_threshold,
                x / (1.0 - 1.0 / (exp(x) - 1.0)),
                x + bernoulli_horner(x)
            )
        )
    )
end

function sedan_flux(p_c, n_c, dμex, dϕ, z, F, RT, DC)
    x = z * dϕ * F / RT + dμex / RT
    p_b = fbernoulli_p(x)
    n_b = fbernoulli_n(x)
    return DC * (n_b * p_c - p_b * n_c)
end

function rexp(x; trunc)
    IfElse.ifelse(
        x < -trunc,
        IfElse.ifelse(
            -x < trunc,
            1.0 / exp(-x),
            1.0 / (exp(trunc) * (-x - trunc + 1))
        ),
        IfElse.ifelse(
            x < trunc,
            exp(x),
            exp(trunc) * (x - trunc + 1)
        )
    )
end

rrate(R0, β, A, RT, trunc) = R0 * (rexp(-β * A / RT; trunc) - rexp((1 - β) * A / RT; trunc))

###################### Constants #######################

@constants begin
    R       = 8.31446261815324      ,[description = "molar gas constant", unit = u"m^2 * kg / s^2 / K / mol"]
    F       = 96485.33212331001     ,[description = "Faraday constant", unit = u"s * A / mol"]
    eps_0   = 8.8541878128e-12      ,[description = "permittivity of the vaccum", unit = u"A^2 * s^4 / kg / m^3"]
    p°      = 0.0                   ,[description = "standard pressure", unit = u"Pa"]
    C°      = 1.0 * 1.0e3           ,[unit = u"mol / m^3"]
    T       = 298.0                 ,[description = "standard temperature", unit = u"K"]
    v0      = 1.0 / 55.4 * 1.0e-3   ,[description = "molar volume of water at standard conditions", unit = u"m^3 / mol"]
    M0      = 18.0153 * 1.0e3       ,[description = "molar mass of water at standard conditions", unit = u"kg / mol"]
end


###################### Models ##########################

@connector Electrolyte begin
    @parameters begin
        # R       = 8.31446261815324                  ,[unit = u"m^2 * kg / s^2 / K / mol"]
        # p°      = 0.0                               ,[unit = u"Pa"]
        # F       = 96485.33212331001                 ,[unit = u"s * A / mol"]
        # eps_0   = 8.8541878128e-12                  ,[unit = u"A^2 * s^4 / kg / m^3"]
        # eps_r   = 1.0                               ,[unit = u"1"]
        # z[1:4]  = [1, 2, 3, -2]                     ,[unit = u"1"]
        # v0      = 1.0 / 55.4                        ,[unit = u"dm^3 / mol"]
        # v[1:4]  = [1.0, 1.0, 1.0, 1.0] ./ 55.4,     ,[unit = u"dm^3/mol"]
        # κ[1:4]  = [0, 0, 0, 0]                      ,[unit = u"1"]
        # C°      = 1.0                               ,[unit = u"mol / dm^3"]
        # Gf[1:4] = [0.0, 0.0, 0.0, 0.0]              ,[unit = u"J / mol"]
        # M0      = 18.0153                           ,[unit = u"g / mol"]
        # M[1:4]  = [1.0, 1.0, 1.0, 1.0] .* 18.0153   ,[unit = u"g / mol"]
        # T       = 298.0,                            ,[unit = u"K"]
        # RT      = R * T                             ,[unit = u"m^2 * kg / s^2 / mol"]

        κ₁       ,[description = "solvation number", unit=u"1"]
        κ₂       ,[description = "solvation number", unit=u"1"]
        κ₃       ,[description = "solvation number", unit=u"1"]
        κ₄       ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v₁       ,[unit = u"m^3 / mol"]
        v₂       ,[unit = u"m^3 / mol"]
        v₃       ,[unit = u"m^3 / mol"]
        v₄       ,[unit = u"m^3 / mol"]
        Gf₁      ,[unit = u"J / mol"]
        Gf₂      ,[unit = u"J / mol"]
        Gf₃      ,[unit = u"J / mol"]
        Gf₄      ,[unit = u"J / mol"]
        M₁       ,[unit = u"kg / mol"]
        M₂       ,[unit = u"kg / mol"]
        M₃       ,[unit = u"kg / mol"]
        M₄       ,[unit = u"kg / mol"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        DC₁      ,[unit = u"m^2 / s"]
        DC₂      ,[unit = u"m^2 / s"]
        DC₃      ,[unit = u"m^2 / s"]
        DC₄      ,[unit = u"m^2 / s"]
        z₁       ,[description = "charge numbers", unit = u"1"]
        z₂       ,[description = "charge numbers", unit = u"1"]
        z₃       ,[description = "charge numbers", unit = u"1"]
        z₄       ,[description = "charge numbers", unit = u"1"]
    end
    @variables begin
        C₁(t)           ,[description = "concentration", unit = u"mol / m^3"]
        C₂(t)           ,[description = "concentration", unit = u"mol / m^3"]
        C₃(t)           ,[description = "concentration", unit = u"mol / m^3"]
        C₄(t)           ,[description = "concentration", unit = u"mol / m^3"]
        p(t)            ,[description = "pressure", unit = u"Pa"]
        ϕ(t)            ,[description = "electrical potential", unit = u"V"]

        Nflux₁(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₂(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₃(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₄(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Dflux(t)        ,[description = "electric flux", unit = u"A * s", connect = Flow]
        sflux(t)        ,[description = "stress gradient flux", unit = u"kg / s^2", connect = Flow]
    end
    @equations begin
        Nflux₁ ~ 0.0
        Nflux₂ ~ 0.0
        Nflux₃ ~ 0.0
        Nflux₄ ~ 0.0
        Dflux ~ 0.0
        sflux ~ 0.0
    end
end

@connector ElectrolyteInterface begin
    @parameters begin
        κ₁       ,[description = "solvation number", unit=u"1"]
        κ₂       ,[description = "solvation number", unit=u"1"]
        κ₃       ,[description = "solvation number", unit=u"1"]
        κ₄       ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v₁       ,[unit = u"m^3 / mol"]
        v₂       ,[unit = u"m^3 / mol"]
        v₃       ,[unit = u"m^3 / mol"]
        v₄       ,[unit = u"m^3 / mol"]
        Gf₁      ,[unit = u"J / mol"]
        Gf₂      ,[unit = u"J / mol"]
        Gf₃      ,[unit = u"J / mol"]
        Gf₄      ,[unit = u"J / mol"]
        M₁       ,[unit = u"kg / mol"]
        M₂       ,[unit = u"kg / mol"]
        M₃       ,[unit = u"kg / mol"]
        M₄       ,[unit = u"kg / mol"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        DC₁      ,[unit = u"m^2 / s"]
        DC₂      ,[unit = u"m^2 / s"]
        DC₃      ,[unit = u"m^2 / s"]
        DC₄      ,[unit = u"m^2 / s"]
        z₁       ,[description = "charge numbers", unit = u"1"]
        z₂       ,[description = "charge numbers", unit = u"1"]
        z₃       ,[description = "charge numbers", unit = u"1"]
        z₄       ,[description = "charge numbers", unit = u"1"]
    end
    @variables begin
        C₁(t)           ,[description = "concentration", unit = u"mol / m^3"]
        C₂(t)           ,[description = "concentration", unit = u"mol / m^3"]
        C₃(t)           ,[description = "concentration", unit = u"mol / m^3"]
        C₄(t)           ,[description = "concentration", unit = u"mol / m^3"]
        p(t)            ,[description = "pressure", unit = u"Pa"]
        ϕ(t)            ,[description = "electrical potential", unit = u"V"]

        Nflux₁(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₂(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₃(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₄(t)       ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Dflux(t)=0.0    ,[description = "electric flux", unit = u"A * s", connect = Flow]
        sflux(t)        ,[description = "stress gradient flux", unit = u"kg / s^2", connect = Flow]
    end
end

@mtkmodel OneElectrolyteInterface begin
    @parameters begin
        κ₁       ,[description = "solvation number", unit=u"1"]
        κ₂       ,[description = "solvation number", unit=u"1"]
        κ₃       ,[description = "solvation number", unit=u"1"]
        κ₄       ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v₁       ,[unit = u"m^3 / mol"]
        v₂       ,[unit = u"m^3 / mol"]
        v₃       ,[unit = u"m^3 / mol"]
        v₄       ,[unit = u"m^3 / mol"]
        Gf₁      ,[unit = u"J / mol"]
        Gf₂      ,[unit = u"J / mol"]
        Gf₃      ,[unit = u"J / mol"]
        Gf₄      ,[unit = u"J / mol"]
        M₁       ,[unit = u"kg / mol"]
        M₂       ,[unit = u"kg / mol"]
        M₃       ,[unit = u"kg / mol"]
        M₄       ,[unit = u"kg / mol"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        DC₁      ,[unit = u"m^2 / s"]
        DC₂      ,[unit = u"m^2 / s"]
        DC₃      ,[unit = u"m^2 / s"]
        DC₄      ,[unit = u"m^2 / s"]
        z₁       ,[description = "charge numbers", unit = u"1"]
        z₂       ,[description = "charge numbers", unit = u"1"]
        z₃       ,[description = "charge numbers", unit = u"1"]
        z₄       ,[description = "charge numbers", unit = u"1"]
    end
    @components begin
        ei = ElectrolyteInterface(κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
    end
    @constants begin
        eps_reg = 1.0e-20, [unit = u"1"]
    end
    @variables begin
        C0(t),          [description = "concentration of the solvent", unit = u"mol / m^3"]
        Cbar(t),        [description = "total concentration", unit = u"mol / m^3"]
        qf(t),          [description = "free charge density", unit = u"C / m^3"]
        γ₁(t),          [description = "activity coefficient", unit = u"1"]
        γ₂(t),          [description = "activity coefficient", unit = u"1"]
        γ₃(t),          [description = "activity coefficient", unit = u"1"]
        γ₄(t),          [description = "activity coefficient", unit = u"1"]
        μ₁(t),          [description = "chemical potential", unit = u"J / mol"]
        μ₂(t),          [description = "chemical potential", unit = u"J / mol"]
        μ₃(t),          [description = "chemical potential", unit = u"J / mol"]
        μ₄(t),          [description = "chemical potential", unit = u"J / mol"]
    end
    @equations begin
        C0 ~ C0_func([ei.C₁, ei.C₂, ei.C₃, ei.C₄], [v₁, v₂, v₃, v₄], v0, [κ₁, κ₂, κ₃, κ₄])
        Cbar ~ C0 + sum([Ci for Ci in [ei.C₁, ei.C₂, ei.C₃, ei.C₄]]) #+ ei.C[1] + ei.C[2] + ei.C[3] + ei.C[4]
        qf ~ sum([F * Ci * zi for (Ci, zi) in zip([ei.C₁, ei.C₂, ei.C₃, ei.C₄], [z₁, z₂, z₃, z₄])])

        γ₁ ~ γfunc(ei.p, p°, C0, Cbar, M₁, M0, v₁, v0, κ₁, RT)
        γ₂ ~ γfunc(ei.p, p°, C0, Cbar, M₂, M0, v₂, v0, κ₂, RT)
        γ₃ ~ γfunc(ei.p, p°, C0, Cbar, M₃, M0, v₃, v0, κ₃, RT)
        γ₄ ~ γfunc(ei.p, p°, C0, Cbar, M₄, M0, v₄, v0, κ₄, RT)
        μ₁ ~ Gf₁ + RT * rlog(γ₁ * ei.C₁ / Cbar; eps_reg=eps_reg)
        μ₂ ~ Gf₂ + RT * rlog(γ₂ * ei.C₂ / Cbar; eps_reg=eps_reg)
        μ₃ ~ Gf₃ + RT * rlog(γ₃ * ei.C₃ / Cbar; eps_reg=eps_reg)
        μ₄ ~ Gf₄ + RT * rlog(γ₄ * ei.C₄ / Cbar; eps_reg=eps_reg)
    end
end


@mtkmodel ElectrolyteControlVolume begin
    @extend OneElectrolyteInterface(; κ₁, κ₂, κ₃, κ₄, eps_r, v₁, v₂, v₃, v₄, Gf₁, Gf₂, Gf₃, Gf₄, M₁, M₂, M₃, M₄, DC₁, DC₂, DC₃, DC₄, RT, z₁, z₂, z₃, z₄) #(; κ, eps_r, v, Gf, M, DC, RT, z)
    @parameters begin
        V, [description = "Volume", unit = u"m^3"]
    end
    @variables begin
        C₁(t) = 1000.0  ,[description = "concentration", unit = u"mol / m^3"]
        C₂(t) = 100.0   ,[description = "concentration", unit = u"mol / m^3"]
        C₃(t) = 100.0   ,[description = "concentration", unit = u"mol / m^3"]
        C₄(t) = 750.0   ,[description = "concentration", unit = u"mol / m^3"]
        p(t)            ,[description = "pressure", unit = u"Pa"]
        ϕ(t)            ,[description = "electrical potential", unit = u"V"]
        der_C₁(t)       ,[unit = u"mol / m^3 / s"]
        der_C₂(t)       ,[unit = u"mol / m^3 / s"]
        der_C₃(t)       ,[unit = u"mol / m^3 / s"]
        der_C₄(t)       ,[unit = u"mol / m^3 / s"]
    end
    @equations begin
        C₁ ~ ei.C₁
        C₂ ~ ei.C₂
        C₃ ~ ei.C₃
        C₄ ~ ei.C₄ 
        p ~ ei.p
        ϕ ~ ei.ϕ

        der_C₁ ~ ei.Nflux₁ / V
        der_C₂ ~ ei.Nflux₂ / V
        der_C₃ ~ ei.Nflux₃ / V
        der_C₄ ~ ei.Nflux₄ / V
        
        D(C₁) ~ der_C₁
        D(C₂) ~ der_C₂
        D(C₃) ~ der_C₃
        D(C₄) ~ der_C₄

        qf ~ ei.Dflux / V
        0 ~ ei.sflux
    end
end

@mtkmodel ElectrolyteControlVolumeBoundary begin
    @constants begin
        eps_reg = 1.0e-20, [unit = u"1"]
    end
    @parameters begin
        A           ,[description = "Area of the control volume boundary part", unit = u"m^2"]
        h           ,[description = "distance between collocation points", unit = u"m"]
        κ₁       ,[description = "solvation number", unit=u"1"]
        κ₂       ,[description = "solvation number", unit=u"1"]
        κ₃       ,[description = "solvation number", unit=u"1"]
        κ₄       ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v₁       ,[unit = u"m^3 / mol"]
        v₂       ,[unit = u"m^3 / mol"]
        v₃       ,[unit = u"m^3 / mol"]
        v₄       ,[unit = u"m^3 / mol"]
        Gf₁      ,[unit = u"J / mol"]
        Gf₂      ,[unit = u"J / mol"]
        Gf₃      ,[unit = u"J / mol"]
        Gf₄      ,[unit = u"J / mol"]
        M₁       ,[unit = u"kg / mol"]
        M₂       ,[unit = u"kg / mol"]
        M₃       ,[unit = u"kg / mol"]
        M₄       ,[unit = u"kg / mol"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        DC₁      ,[unit = u"m^2 / s"]
        DC₂      ,[unit = u"m^2 / s"]
        DC₃      ,[unit = u"m^2 / s"]
        DC₄      ,[unit = u"m^2 / s"]
        z₁       ,[description = "charge numbers", unit = u"1"]
        z₂       ,[description = "charge numbers", unit = u"1"]
        z₃       ,[description = "charge numbers", unit = u"1"]
        z₄       ,[description = "charge numbers", unit = u"1"]
    end
    @components begin
        n = ElectrolyteInterface(κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
        p = ElectrolyteInterface(κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
    end
    @variables begin
        dϕ(t)       ,[unit = u"V"]
        dp(t)       ,[unit = u"Pa"]   
        dμex₁(t)    ,[unit = u"J / mol"]
        dμex₂(t)    ,[unit = u"J / mol"]
        dμex₃(t)    ,[unit = u"J / mol"]
        dμex₄(t)    ,[unit = u"J / mol"]

        p_C0(t)     ,[unit = u"mol / m^3"]
        n_C0(t)     ,[unit = u"mol / m^3"]

        p_Cbar(t)   ,[unit = u"mol / m^3"]
        n_Cbar(t)   ,[unit = u"mol / m^3"]

        p_qf(t)     ,[unit = u"A * s / m^3"]
        n_qf(t)     ,[unit = u"A * s / m^3"]
        
        p_γ₁(t)     ,[unit = u"1"]
        p_γ₂(t)     ,[unit = u"1"]
        p_γ₃(t)     ,[unit = u"1"]
        p_γ₄(t)     ,[unit = u"1"]
        n_γ₁(t)     ,[unit = u"1"]
        n_γ₂(t)     ,[unit = u"1"]
        n_γ₃(t)     ,[unit = u"1"]
        n_γ₄(t)     ,[unit = u"1"]

        Nflux₁(t)   ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₂(t)   ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₃(t)   ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Nflux₄(t)   ,[description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Dflux(t)    ,[description = "electric flux", unit = u"A * s", connect = Flow]
        sflux(t)    ,[description = "stress gradient flux", unit = u"kg / s^2", connect = Flow]
    end
    @equations begin
        p_C0 ~ C0_func([p.C₁, p.C₂, p.C₃, p.C₄], [v₁, v₂, v₃, v₄], v0, [κ₁, κ₂, κ₃, κ₄])
        n_C0 ~ C0_func([n.C₁, n.C₂, n.C₃, n.C₄], [v₁, v₂, v₃, v₄], v0, [κ₁, κ₂, κ₃, κ₄])

        p_Cbar ~ p_C0 + sum([p.C₁, p.C₂, p.C₃, p.C₄])
        n_Cbar ~ n_C0 + sum([n.C₁, n.C₂, n.C₃, n.C₄])

        p_qf ~ sum([F * Ci * zi for (Ci, zi) in zip([p.C₁, p.C₂, p.C₃, p.C₄], [z₁, z₂, z₃, z₄])])
        n_qf ~ sum([F * Ci * zi for (Ci, zi) in zip([n.C₁, n.C₂, n.C₃, n.C₄], [z₁, z₂, z₃, z₄])])

        p_γ₁ ~ γfunc(p.p, p°, p_C0, p_Cbar, M₁, M0, v₁, v0, κ₁, RT)
        p_γ₂ ~ γfunc(p.p, p°, p_C0, p_Cbar, M₂, M0, v₂, v0, κ₂, RT)
        p_γ₃ ~ γfunc(p.p, p°, p_C0, p_Cbar, M₃, M0, v₃, v0, κ₃, RT)
        p_γ₄ ~ γfunc(p.p, p°, p_C0, p_Cbar, M₄, M0, v₄, v0, κ₄, RT)

        n_γ₁ ~ γfunc(n.p, p°, n_C0, n_Cbar, M₁, M0, v₁, v0, κ₁, RT)
        n_γ₂ ~ γfunc(n.p, p°, n_C0, n_Cbar, M₂, M0, v₂, v0, κ₂, RT)
        n_γ₃ ~ γfunc(n.p, p°, n_C0, n_Cbar, M₃, M0, v₃, v0, κ₃, RT)
        n_γ₄ ~ γfunc(n.p, p°, n_C0, n_Cbar, M₄, M0, v₄, v0, κ₄, RT)

        dϕ ~ p.ϕ - n.ϕ
        dp ~ p.p - n.p
        dμex₁ ~ dμex_func(p_γ₁, n_γ₁, RT, eps_reg)
        dμex₂ ~ dμex_func(p_γ₂, n_γ₂, RT, eps_reg)
        dμex₃ ~ dμex_func(p_γ₃, n_γ₃, RT, eps_reg)
        dμex₄ ~ dμex_func(p_γ₄, n_γ₄, RT, eps_reg)
        
        Nflux₁ ~ p.Nflux₁
        Nflux₂ ~ p.Nflux₂
        Nflux₃ ~ p.Nflux₃
        Nflux₄ ~ p.Nflux₄

        Nflux₁ ~ -n.Nflux₁
        Nflux₂ ~ -n.Nflux₂
        Nflux₃ ~ -n.Nflux₃
        Nflux₄ ~ -n.Nflux₄

        Nflux₁ ~ A / h * sedan_flux(p.C₁, n.C₁, dμex₁, dϕ, z₁, F, RT, DC₁)
        Nflux₂ ~ A / h * sedan_flux(p.C₂, n.C₂, dμex₂, dϕ, z₂, F, RT, DC₂)
        Nflux₃ ~ A / h * sedan_flux(p.C₃, n.C₃, dμex₃, dϕ, z₃, F, RT, DC₃)
        Nflux₄ ~ A / h * sedan_flux(p.C₄, n.C₄, dμex₄, dϕ, z₄, F, RT, DC₄)

        Dflux ~ p.Dflux
        Dflux ~ -n.Dflux
        Dflux ~ A / h * (-eps_r * eps_0 * dϕ)

        sflux ~ p.sflux
        sflux ~ -n.sflux
        sflux ~ A / h * (dp + (p_qf + n_qf) / 2 * dϕ)
    end
end

@mtkmodel ElectrolyteBulkBoundary begin
    @extend OneElectrolyteInterface(; κ₁, κ₂, κ₃, κ₄, eps_r, v₁, v₂, v₃, v₄, Gf₁, Gf₂, Gf₃, Gf₄, M₁, M₂, M₃, M₄, DC₁, DC₂, DC₃, DC₄, RT, z₁, z₂, z₃, z₄)
    @parameters begin
        ϕ_bulk = 0.0,  [description = "electric potential in the bulk", unit = u"V"]
        p_bulk = 0.0,  [description = "pressure in the bulk", unit = u"Pa"]
        
        A,             [description = "area", unit = u"m^2"]
        h,             [description = "distance between collocation points", unit = u"m"]
        #C_bulk[1:4] = [0.001, 0.01, 0.1, 1],[description = "bulk concentrations", unit = u"mol / dm^3"]
        C_bulk₁,   [description = "bulk concentrations", unit = u"mol / m^3"]
        C_bulk₂,   [description = "bulk concentrations", unit = u"mol / m^3"]
        C_bulk₃,   [description = "bulk concentrations", unit = u"mol / m^3"]
        C_bulk₄,   [description = "bulk concentrations", unit = u"mol / m^3"]
        ϵ = 1.0e-10,   [unit = u"1"]
    end
    @variables begin
        dϕ(t),          [unit = u"V"]
        dp(t),          [unit = u"Pa"]
        dC₁(t),     [unit = u"mol / m^3"]
        dC₂(t),     [unit = u"mol / m^3"]
        dC₃(t),     [unit = u"mol / m^3"]
        dC₄(t),     [unit = u"mol / m^3"]
    end
    @equations begin
        dϕ ~ ϕ_bulk - ei.ϕ
        dp ~ p_bulk - ei.p
        dC₁ ~ C_bulk₁ - ei.C₁
        dC₂ ~ C_bulk₂ - ei.C₂
        dC₃ ~ C_bulk₃ - ei.C₃
        dC₄ ~ C_bulk₄ - ei.C₄

        ei.Nflux₁ ~ -A / h * (DC₁ * dC₁)
        ei.Nflux₂ ~ -A / h * (DC₂ * dC₂)
        ei.Nflux₃ ~ -A / h * (DC₃ * dC₃)
        ei.Nflux₄ ~ -A / h * (DC₄ * dC₄)
        ei.Dflux  ~ -A / h * (-eps_r * eps_0 * dϕ)
        ei.sflux  ~ -A / h * (dp + qf * dϕ) 
    end
end

@mtkmodel ElectrodeElectrolyteBoundary begin
    @extend OneElectrolyteInterface(; κ₁, κ₂, κ₃, κ₄, eps_r, v₁, v₂, v₃, v₄, Gf₁, Gf₂, Gf₃, Gf₄, M₁, M₂, M₃, M₄, DC₁, DC₂, DC₃, DC₄, RT, z₁, z₂, z₃, z₄)
    @constants begin
        β           = 0.5,                  [description = "transfer coefficient"]
        trunc       = 20.0
    end
    @parameters begin
        sc₁             ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
        sc₂             ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
        sc₃             ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
        sc₄             ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
        R0 = 1.0e-6     ,[description = "exchange rate constant", unit = u"mol / m^2 / s"]
        A               ,[description = "area", unit = u"m^2"]
        h               ,[description = "distance between collocation points", unit = u"m"]
        ϕ_we            ,[description = "electric potential at working electrode", unit = u"V"]
        ϵ = 1.0e-10
    end
    @variables begin
        RR(t),  [description = "reaction rate", unit = u"mol / m^2 / s"]
        dG(t),  [description = "change in Gibbs free energy per reaction unit", unit = u"J / mol"]
        dϕ(t),  [unit = u"V"]
    end
    @equations begin
        dϕ ~ ei.ϕ - ϕ_we 
        dG ~ sc₁ * μ₁ + sc₂ * μ₂ + sc₃ * μ₃ + sc₄ * μ₄ - F * dϕ #sum([sci * μi for (sci, μi) in zip(sc, μ)]) - F * dϕ
        
        RR ~ rrate(R0, β, -dG, RT, trunc)
        
        ei.Nflux₁ ~ sc₁ * RR * A
        ei.Nflux₂ ~ sc₁ * RR * A
        ei.Nflux₃ ~ sc₁ * RR * A
        ei.Nflux₄ ~ sc₁ * RR * A
        
        #ei.Dflux ~ A / h * (-eps_r * eps_0 * dϕ)
        ei.Dflux ~ A / h * (-eps_r * eps_0 * dϕ)
        ei.sflux ~ 0.0
    end
end

# @mtkmodel System begin
#     @parameters begin
#         ϕ_we     ,[description = "electric potential at working electrode", unit = u"V"]
#         ϕ_bulk   ,[description = "electric potential in the bulk", unit = u"V"]
#         p_bulk   ,[description = "pressure in the bulk", unit = u"Pa"]
#         C_bulk₁  ,[description = "bulk concentrations", unit = u"mol / m^3"]
#         C_bulk₂  ,[description = "bulk concentrations", unit = u"mol / m^3"]
#         C_bulk₃  ,[description = "bulk concentrations", unit = u"mol / m^3"]
#         C_bulk₄  ,[description = "bulk concentrations", unit = u"mol / m^3"]
#         A        ,[description = "crossectional area", unit = u"m^2"]
#         κ₁       ,[description = "solvation number", unit=u"1"]
#         κ₂       ,[description = "solvation number", unit=u"1"]
#         κ₃       ,[description = "solvation number", unit=u"1"]
#         κ₄       ,[description = "solvation number", unit=u"1"]
#         eps_r    ,[unit = u"1"]
#         v₁       ,[unit = u"m^3 / mol"]
#         v₂       ,[unit = u"m^3 / mol"]
#         v₃       ,[unit = u"m^3 / mol"]
#         v₄       ,[unit = u"m^3 / mol"]
#         Gf₁      ,[unit = u"J / mol"]
#         Gf₂      ,[unit = u"J / mol"]
#         Gf₃      ,[unit = u"J / mol"]
#         Gf₄      ,[unit = u"J / mol"]
#         M₁       ,[unit = u"kg / mol"]
#         M₂       ,[unit = u"kg / mol"]
#         M₃       ,[unit = u"kg / mol"]
#         M₄       ,[unit = u"kg / mol"]
#         RT       ,[unit = u"m^2 * kg / s^2 / mol"]
#         DC₁      ,[unit = u"m^2 / s"]
#         DC₂      ,[unit = u"m^2 / s"]
#         DC₃      ,[unit = u"m^2 / s"]
#         DC₄      ,[unit = u"m^2 / s"]
#         z₁       ,[description = "charge numbers", unit = u"1"]
#         z₂       ,[description = "charge numbers", unit = u"1"]
#         z₃       ,[description = "charge numbers", unit = u"1"]
#         z₄       ,[description = "charge numbers", unit = u"1"]
#         sc₁      ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
#         sc₂      ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
#         sc₃      ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
#         sc₄      ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
#     end
#     @components begin
#         e = Electrolyte(;κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         eeb = ElectrodeElectrolyteBoundary(; A=A, h=1.5e-3, ϕ_we=ϕ_we, sc₁=sc₁, sc₂=sc₂, sc₃=sc₃, sc₄=sc₄, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         cv1 = ElectrolyteControlVolume(; V=1.0e-3, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         b12 = ElectrolyteControlVolumeBoundary(; A=A, h=1.0e-3, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         cv2 = ElectrolyteControlVolume(; V=1.0e-3, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         b23 = ElectrolyteControlVolumeBoundary(; A=A, h=1.0e-3, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         cv3 = ElectrolyteControlVolume(; V=1.0e-3, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#         ebb = ElectrolyteBulkBoundary(; A=A, h=1.5e-3, C_bulk₁=C_bulk₁, C_bulk₂=C_bulk₂, C_bulk₃=C_bulk₃, C_bulk₄=C_bulk₄, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
#     end
#     @equations begin
#         connect(eeb.ei, cv1.ei)
#         connect(cv1.ei, b12.n)
#         connect(b12.p, cv2.ei)
#         connect(cv2.ei, b23.n)
#         connect(b23.p, cv3.ei)
#         connect(cv3.ei, ebb.ei)
#         connect(cv3.ei, e)
#     end
# end

function testsystem(grid;
    ϕ_we   ,# = defaults["ϕ_we"],
    ϕ_bulk ,# = defaults["ϕ_bulk"],
    p_bulk ,# = defaults["p_bulk"],
    C_bulk₁,# = defaults["C_bulk₁"],
    C_bulk₂,# = defaults["C_bulk₂"],
    C_bulk₃,# = defaults["C_bulk₃"],
    C_bulk₄,# = defaults["C_bulk₄"],
    κ₁     ,# = defaults["κ₁"],
    κ₂     ,# = defaults["κ₂"],
    κ₃     ,# = defaults["κ₃"],
    κ₄     ,# = defaults["κ₄"],
    A      ,# = defaults["A"],
    eps_r  ,# = defaults["eps_r"],
    v₁     ,# = defaults["v₁"],
    v₂     ,# = defaults["v₂"],
    v₃     ,# = defaults["v₃"],
    v₄     ,# = defaults["v₄"],
    Gf₁    ,# = defaults["Gf₁"],
    Gf₂    ,# = defaults["Gf₂"],
    Gf₃    ,# = defaults["Gf₃"],
    Gf₄    ,# = defaults["Gf₄"],
    M₁     ,# = defaults["M₁"],
    M₂     ,# = defaults["M₂"],
    M₃     ,# = defaults["M₃"],
    M₄     ,# = defaults["M₄"],
    RT     ,# = defaults["RT"],
    DC₁    ,# = defaults["DC₁"],
    DC₂    ,# = defaults["DC₂"],
    DC₃    ,# = defaults["DC₃"],
    DC₄    ,# = defaults["DC₄"],
    z₁     ,# = defaults["z₁"],
    z₂     ,# = defaults["z₂"],
    z₃     ,# = defaults["z₃"],
    z₄     ,# = defaults["z₄"],
    sc₁    ,# = defaults["sc₁"],
    sc₂    ,# = defaults["sc₂"],
    sc₃    ,# = defaults["sc₃"],
    sc₄    ,# = defaults["sc₄"],
    name, 
)
    hs = grid[CellVolumes]  
    Vs = [[hs[1] / 2]; [hs[i] / 2 + hs[i+1] / 2 for i in eachindex(hs[1:end-1])]; [hs[end] / 2]] .* A
    ps = @parameters begin
        ϕ_we     = ϕ_we    ,[description = "electric potential at working electrode", unit = u"V"]
    #     ϕ_bulk   = ϕ_bulk  ,[description = "electric potential in the bulk", unit = u"V"]
    #     p_bulk   = p_bulk  ,[description = "pressure in the bulk", unit = u"Pa"]
    #     C_bulk₁  = C_bulk₁ ,[description = "bulk concentrations", unit = u"mol / m^3"]
    #     C_bulk₂  = C_bulk₂ ,[description = "bulk concentrations", unit = u"mol / m^3"]
    #     C_bulk₃  = C_bulk₃ ,[description = "bulk concentrations", unit = u"mol / m^3"]
    #     C_bulk₄  = C_bulk₄ ,[description = "bulk concentrations", unit = u"mol / m^3"]
    #     A        = A       ,[description = "crossectional area", unit = u"m^2"]
    #     κ₁       = κ₁      ,[description = "solvation number", unit=u"1"]
    #     κ₂       = κ₂      ,[description = "solvation number", unit=u"1"]
    #     κ₃       = κ₃      ,[description = "solvation number", unit=u"1"]
    #     κ₄       = κ₄      ,[description = "solvation number", unit=u"1"]
    #     eps_r    = eps_r   ,[unit = u"1"]
    #     v₁       = v₁      ,[unit = u"m^3 / mol"]
    #     v₂       = v₂      ,[unit = u"m^3 / mol"]
    #     v₃       = v₃      ,[unit = u"m^3 / mol"]
    #     v₄       = v₄      ,[unit = u"m^3 / mol"]
    #     Gf₁      = Gf₁     ,[unit = u"J / mol"]
    #     Gf₂      = Gf₂     ,[unit = u"J / mol"]
    #     Gf₃      = Gf₃     ,[unit = u"J / mol"]
    #     Gf₄      = Gf₄     ,[unit = u"J / mol"]
    #     M₁       = M₁      ,[unit = u"kg / mol"]
    #     M₂       = M₂      ,[unit = u"kg / mol"]
    #     M₃       = M₃      ,[unit = u"kg / mol"]
    #     M₄       = M₄      ,[unit = u"kg / mol"]
    #     RT       = RT      ,[unit = u"m^2 * kg / s^2 / mol"]
    #     DC₁      = DC₁     ,[unit = u"m^2 / s"]
    #     DC₂      = DC₂     ,[unit = u"m^2 / s"]
    #     DC₃      = DC₃     ,[unit = u"m^2 / s"]
    #     DC₄      = DC₄     ,[unit = u"m^2 / s"]
    #     z₁       = z₁      ,[description = "charge numbers", unit = u"1"]
    #     z₂       = z₂      ,[description = "charge numbers", unit = u"1"]
    #     z₃       = z₃      ,[description = "charge numbers", unit = u"1"]
    #     z₄       = z₄      ,[description = "charge numbers", unit = u"1"]
    #     sc₁      = sc₁     ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
    #     sc₂      = sc₂     ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
    #     sc₃      = sc₃     ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
    #     sc₄      = sc₄     ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
    end

    @named eeb = ElectrodeElectrolyteBoundary(; A=A, h=hs[1], ϕ_we=ϕ_we, sc₁=sc₁, sc₂=sc₂, sc₃=sc₃, sc₄=sc₄, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
    cv = ODESystem[]
    for (i, V) in enumerate(Vs[2:end-1])
        push!(cv, ElectrolyteControlVolume(; V=V, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄, name=Symbol(:cv, i)))
    end    
    #@named cv[1:N]  = ElectrolyteControlVolume(; V=1.0e-9, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
    b = ODESystem[]
    for (i, h) in enumerate(hs[2:end-1])
        push!(b, ElectrolyteControlVolumeBoundary(; A=A, h=h, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄, name=Symbol(:b, i)))
    end
    #@named b[1:N-1] = ElectrolyteControlVolumeBoundary(; A=A, h=h[i], κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)
    @named ebb = ElectrolyteBulkBoundary(; A=A, h=hs[end], C_bulk₁=C_bulk₁, C_bulk₂=C_bulk₂, C_bulk₃=C_bulk₃, C_bulk₄=C_bulk₄, κ₁=κ₁, κ₂=κ₂, κ₃=κ₃, κ₄=κ₄, eps_r=eps_r, v₁=v₁, v₂=v₂, v₃=v₃, v₄=v₄, Gf₁=Gf₁, Gf₂=Gf₂, Gf₃=Gf₃, Gf₄=Gf₄, M₁=M₁, M₂=M₂, M₃=M₃, M₄=M₄, DC₁=DC₁, DC₂=DC₂, DC₃=DC₃, DC₄=DC₄, RT=RT, z₁=z₁, z₂=z₂, z₃=z₃, z₄=z₄)

    eqs = [
        connect(eeb.ei, cv[1].ei),
        [connect(cv.ei, b.n) for (cv, b) in zip(cv[1:end-1], b)]...,
        [connect(b.p, cv.ei) for (b, cv) in zip(b, cv[2:end])]...,
        connect(cv[end].ei, ebb.ei)
    ]
    ODESystem(eqs, t, [], ps; systems=[eeb, cv..., b..., ebb], name)
end


defaults = Dict(
    "ϕ_we" => -0.0,
    "ϕ_bulk" => 0.0,
    "p_bulk" => 0.0,
    "C_bulk₁" => 1.0 * 1.0e3, #[1.0, 0.1, 0.1, 0.75] .* 1.0e3,
    "C_bulk₂" => 0.1 * 1.0e3,
    "C_bulk₃" => 0.1 * 1.0e3,
    "C_bulk₄" => 0.75 * 1.0e3,
    "κ₁" => 0.0,
    "κ₂" => 0.0,
    "κ₃" => 0.0,
    "κ₄" => 0.0,
    "A" => 1.0,
    "eps_r" => 1.0,
    "v₁" => 1.0 / 55.2 * 1.0e-3,
    "v₂" => 1.0 / 55.2 * 1.0e-3,
    "v₃" => 1.0 / 55.2 * 1.0e-3,
    "v₄" => 1.0 / 55.2 * 1.0e-3,
    "Gf₁" => 0.0,
    "Gf₂" => 0.0,
    "Gf₃" => 0.0,
    "Gf₄" => 0.0,
    "M₁" => 1.0 * 18.0153 * 1.0e-3,
    "M₂" => 1.0 * 18.0153 * 1.0e-3,
    "M₃" => 1.0 * 18.0153 * 1.0e-3,
    "M₄" => 1.0 * 18.0153 * 1.0e-3,
    "RT" => 8.31446261815324 * 298.0,
    "DC₁" => 1.0 * 1.0e-9,
    "DC₂" => 1.0 * 1.0e-9,
    "DC₃" => 1.0 * 1.0e-9,
    "DC₄" => 1.0 * 1.0e-9,
    "z₁" => 1,
    "z₂" => 2,
    "z₃" => 3,
    "z₄" => -2,
    "sc₁" => 0,
    "sc₂" => -1,
    "sc₃" => 1,
    "sc₄" => 0,
)

grid = simplexgrid(geomspace(0.0, 1.0e-6, 1.0e-9, 1.0e-7))
@named testsys = testsystem(grid;
ϕ_we    = defaults["ϕ_we"],
ϕ_bulk  = defaults["ϕ_bulk"],
p_bulk  = defaults["p_bulk"],
C_bulk₁ = defaults["C_bulk₁"],
C_bulk₂ = defaults["C_bulk₂"],
C_bulk₃ = defaults["C_bulk₃"],
C_bulk₄ = defaults["C_bulk₄"],
κ₁      = defaults["κ₁"],
κ₂      = defaults["κ₂"],
κ₃      = defaults["κ₃"],
κ₄      = defaults["κ₄"],
A       = defaults["A"],
eps_r   = defaults["eps_r"],
v₁      = defaults["v₁"],
v₂      = defaults["v₂"],
v₃      = defaults["v₃"],
v₄      = defaults["v₄"],
Gf₁     = defaults["Gf₁"],
Gf₂     = defaults["Gf₂"],
Gf₃     = defaults["Gf₃"],
Gf₄     = defaults["Gf₄"],
M₁      = defaults["M₁"],
M₂      = defaults["M₂"],
M₃      = defaults["M₃"],
M₄      = defaults["M₄"],
RT      = defaults["RT"],
DC₁     = defaults["DC₁"],
DC₂     = defaults["DC₂"],
DC₃     = defaults["DC₃"],
DC₄     = defaults["DC₄"],
z₁      = defaults["z₁"],
z₂      = defaults["z₂"],
z₃      = defaults["z₃"],
z₄      = defaults["z₄"],
sc₁     = defaults["sc₁"],
sc₂     = defaults["sc₂"],
sc₃     = defaults["sc₃"],
sc₄     = defaults["sc₄"],
)

# @mtkbuild test = System(; 
#     ϕ_we    = defaults["ϕ_we"],
#     ϕ_bulk  = defaults["ϕ_bulk"],
#     p_bulk  = defaults["p_bulk"],
#     C_bulk₁ = defaults["C_bulk₁"],
#     C_bulk₂ = defaults["C_bulk₂"],
#     C_bulk₃ = defaults["C_bulk₃"],
#     C_bulk₄ = defaults["C_bulk₄"],
#     κ₁      = defaults["κ₁"],
#     κ₂      = defaults["κ₂"],
#     κ₃      = defaults["κ₃"],
#     κ₄      = defaults["κ₄"],
#     A       = defaults["A"],
#     eps_r   = defaults["eps_r"],
#     v₁      = defaults["v₁"],
#     v₂      = defaults["v₂"],
#     v₃      = defaults["v₃"],
#     v₄      = defaults["v₄"],
#     Gf₁     = defaults["Gf₁"],
#     Gf₂     = defaults["Gf₂"],
#     Gf₃     = defaults["Gf₃"],
#     Gf₄     = defaults["Gf₄"],
#     M₁      = defaults["M₁"],
#     M₂      = defaults["M₂"],
#     M₃      = defaults["M₃"],
#     M₄      = defaults["M₄"],
#     RT      = defaults["RT"],
#     DC₁     = defaults["DC₁"],
#     DC₂     = defaults["DC₂"],
#     DC₃     = defaults["DC₃"],
#     DC₄     = defaults["DC₄"],
#     z₁      = defaults["z₁"],
#     z₂      = defaults["z₂"],
#     z₃      = defaults["z₃"],
#     z₄      = defaults["z₄"],
#     sc₁     = defaults["sc₁"],
#     sc₂     = defaults["sc₂"],
#     sc₃     = defaults["sc₃"],
#     sc₄     = defaults["sc₄"],
# )

# u0init = [
#     test.cv1.C₁ => 1000.0,   
#     test.cv2.C₁ => 1000.0, 
#     test.cv3.C₁ => 1000.0, 
#     test.cv1.C₂ => 100.0, 
#     test.cv2.C₂ => 100.0, 
#     test.cv3.C₂ => 100.0, 
#     test.cv1.C₃ => 100.0, 
#     test.cv2.C₃ => 100.0, 
#     test.cv3.C₃ => 100.0, 
#     test.cv1.C₄ => 750.0, 
#     test.cv2.C₄ => 750.0, 
#     test.cv3.C₄ => 750.0, 
#     test.ebb.ei.Dflux => 0.0
# ]

# prob = SteadyStateProblem(
#     test, 
#     u0init,
#     [test.ϕ_we=>-1.0]
# )

#@named testsys = testsystem() 
testsys = structural_simplify(testsys)
prob = SteadyStateProblem(testsys, [], [testsys.ϕ_we => -2.1e-2])

sol = solve(prob, DynamicSS(Rodas5P()))

# @mtkmodel ModelA begin
#     @parameters begin
#         p[1:3]
#         q
#     end
#     @variables begin
#         x(t)[1:3]
#     end
#     @equations begin
#         D(x[1]) ~ p[1] * x[1] + q
#         D(x[2]) ~ p[2] * x[2] + q
#         D(x[3]) ~ p[3] * x[3] + q
#     end
# end

# @mtkmodel ModelB begin
#     @parameters begin
#         p[1:3]
#         q
#     end
#     @components begin
#         A = ModelA(p=p, q=q)
#     end
# end

# @mtkbuild B = ModelB(; p=[1.0, 2.0, 3.0], q=2.0)


# export ElectrolyteControlVolume
# export ElectrolyteControlVolumeBoundary
# export ElectrolyteBulkBoundary
# export ElectrodeElectrolyteBoundary

# end