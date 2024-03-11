#module DiffusiveResolvedMTK
using ModelingToolkit, DynamicQuantities
using IfElse
import ChainRulesCore

using ModelingToolkit: t, D, scalarize


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

        κ[1:4]   ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v[1:4]   ,[unit = u"m^3 / mol"]
        Gf[1:4]  ,[unit = u"J / mol"]
        M[1:4]   ,[unit = u"kg / mol"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        DC[1:4]  ,[unit = u"m^2 / s"]
    end
    @variables begin
        C(t)[1:4],      [description = "concentration", unit = u"mol / m^3"]
        p(t),           [description = "pressure", unit = u"Pa"]
        ϕ(t),           [description = "electrical potential", unit = u"V"]

        Nflux(t)[1:4],  [description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Dflux(t),       [description = "electric flux", unit = u"A * s", connect = Flow]
        sflux(t),       [description = "stress gradient flux", unit = u"kg / s^2", connect = Flow]
    end
    @equations begin
        Nflux[1] ~ 0.0
        Nflux[2] ~ 0.0
        Nflux[3] ~ 0.0
        Nflux[4] ~ 0.0
        Dflux ~ 0.0
        sflux ~ 0.0
    end
end

@connector ElectrolyteInterface begin
    @parameters begin
        κ[1:4]   ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v[1:4]   ,[unit = u"m^3 / mol"]
        Gf[1:4]  ,[unit = u"J / mol"]
        M[1:4]   ,[unit = u"kg / mol"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        DC[1:4]  ,[unit = u"m^2 / s"]
        z[1:4]   ,[description = "charge numbers", unit = u"1"]
    end
    @variables begin
        C(t)[1:4],      [description = "concentration", unit = u"mol / m^3"]
        p(t),           [description = "pressure", unit = u"Pa"]
        ϕ(t),           [description = "electrical potential", unit = u"V"]

        Nflux(t)[1:4],  [description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Dflux(t),       [description = "electric flux", unit = u"A * s", connect = Flow]
        sflux(t),       [description = "stress gradient flux", unit = u"kg / s^2", connect = Flow]
    end
end

@mtkmodel OneElectrolyteInterface begin
    @parameters begin
        κ[1:4]   ,[description = "solvation number", unit=u"1"]
        eps_r    ,[unit = u"1"]
        v[1:4]   ,[unit = u"m^3 / mol"]
        Gf[1:4]  ,[unit = u"J / mol"]
        M[1:4]   ,[unit = u"kg / mol"]
        DC[1:4]  ,[unit = u"m^2 / s"]
        RT       ,[unit = u"m^2 * kg / s^2 / mol"]
        z[1:4]   ,[description = "charge numbers", unit = u"1"]
    end
    @components begin
        ei = ElectrolyteInterface(; κ, eps_r, v, Gf, M, DC, RT, z)
    end
    @constants begin
        eps_reg = 1.0e-20, [unit = u"1"]
    end
    @variables begin
        C0(t),          [description = "concentration of the solvent", unit = u"mol / m^3"]
        Cbar(t),        [description = "total concentration", unit = u"mol / m^3"]
        qf(t),          [description = "free charge density", unit = u"C / m^3"]
        γ(t)[1:4],      [description = "activity coefficient", unit = u"1"]
        μ(t)[1:4],      [description = "chemical potential", unit = u"J / mol"]
    end
    @equations begin
        C0 ~ C0_func(ei.C, v, v0, κ)
        Cbar ~ C0 + sum([Ci for Ci in ei.C]) #+ ei.C[1] + ei.C[2] + ei.C[3] + ei.C[4]
        qf ~ sum([F * Ci * zi for (Ci, zi) in zip(ei.C, z)])

        γ[1] ~ γfunc(ei.p, p°, C0, Cbar, M[1], M0, v[1], v0, κ[1], RT)
        γ[2] ~ γfunc(ei.p, p°, C0, Cbar, M[2], M0, v[2], v0, κ[2], RT)
        γ[3] ~ γfunc(ei.p, p°, C0, Cbar, M[3], M0, v[3], v0, κ[3], RT)
        γ[4] ~ γfunc(ei.p, p°, C0, Cbar, M[4], M0, v[4], v0, κ[4], RT)
        μ[1] ~ Gf[1] + RT * rlog(γ[1] * ei.C[1] / Cbar; eps_reg=eps_reg)
        μ[2] ~ Gf[2] + RT * rlog(γ[2] * ei.C[2] / Cbar; eps_reg=eps_reg)
        μ[3] ~ Gf[3] + RT * rlog(γ[3] * ei.C[3] / Cbar; eps_reg=eps_reg)
        μ[4] ~ Gf[4] + RT * rlog(γ[4] * ei.C[4] / Cbar; eps_reg=eps_reg)
    end
end


@mtkmodel ElectrolyteControlVolume begin
    @extend OneElectrolyteInterface(; κ, eps_r, v, Gf, M, DC, RT, z)
    @parameters begin
        V, [description = "Volume", unit = u"m^3"]
    end
    @variables begin
        C(t)[1:4],      [description = "concentration", unit = u"mol / m^3"]
        p(t),           [description = "pressure", unit = u"Pa"]
        ϕ(t),           [description = "electrical potential", unit = u"V"]
        der_C(t)[1:4],  [unit = u"mol / m^3 / s"]
    end
    @equations begin
        C[1] ~ ei.C[1]
        C[2] ~ ei.C[2]
        C[3] ~ ei.C[3]
        C[4] ~ ei.C[4]
        p ~ ei.p
        ϕ ~ ei.ϕ

        der_C[1] ~ ei.Nflux[1] / V
        der_C[2] ~ ei.Nflux[2] / V
        der_C[3] ~ ei.Nflux[3] / V
        der_C[4] ~ ei.Nflux[4] / V
        
        D(C[1]) ~ der_C[1]
        D(C[2]) ~ der_C[2]
        D(C[3]) ~ der_C[3]
        D(C[4]) ~ der_C[4]

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
        κ[1:4]      ,[description = "solvation number", unit=u"1"]
        eps_r       ,[unit = u"1"]
        v[1:4]      ,[unit = u"m^3 / mol"]
        Gf[1:4]     ,[unit = u"J / mol"]
        M[1:4]      ,[unit = u"kg / mol"]
        RT          ,[unit = u"m^2 * kg / s^2 / mol"]
        DC[1:4]     ,[unit = u"m^2 / s"]
        z[1:4]      ,[description = "charge numbers", unit = u"1"]
    end
    @components begin
        n = ElectrolyteInterface(; κ=scalarize(κ), eps_r=scalarize(eps_r), v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
        p = ElectrolyteInterface(; κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
    end
    @variables begin
        dϕ(t),          [unit = u"V"]
        dp(t),          [unit = u"Pa"]   
        dμex(t)[1:4],   [unit = u"J / mol"]

        p_C0(t),           [unit = u"mol / m^3"]
        n_C0(t),           [unit = u"mol / m^3"]

        p_Cbar(t),         [unit = u"mol / m^3"]
        n_Cbar(t),         [unit = u"mol / m^3"]

        p_qf(t),           [unit = u"A * s / m^3"]
        n_qf(t),           [unit = u"A * s / m^3"]
        
        p_γ(t)[1:4],       [unit = u"1"]
        n_γ(t)[1:4],       [unit = u"1"]

        Nflux(t)[1:4],  [description = "molar mass flux", unit = u"mol / s", connect = Flow]
        Dflux(t),       [description = "electric flux", unit = u"A * s", connect = Flow]
        sflux(t),       [description = "stress gradient flux", unit = u"kg / s^2", connect = Flow]
    end
    @equations begin
        p_C0 ~ C0_func(p.C, v, v0, κ)
        n_C0 ~ C0_func(n.C, v, v0, κ)

        p_Cbar ~ p_C0 + p.C[1] + p.C[2] + p.C[3] + p.C[4]
        n_Cbar ~ n_C0 + n.C[1] + n.C[2] + n.C[3] + n.C[4]

        p_qf ~ sum([F * Ci * zi for (Ci, zi) in zip(p.C, z)])
        n_qf ~ sum([F * Ci * zi for (Ci, zi) in zip(n.C, z)])

        p_γ[1] ~ γfunc(p.p, p°, p_C0, p_Cbar, M[1], M0, v[1], v0, κ[1], RT)
        p_γ[2] ~ γfunc(p.p, p°, p_C0, p_Cbar, M[2], M0, v[2], v0, κ[2], RT)
        p_γ[3] ~ γfunc(p.p, p°, p_C0, p_Cbar, M[3], M0, v[3], v0, κ[3], RT)
        p_γ[4] ~ γfunc(p.p, p°, p_C0, p_Cbar, M[4], M0, v[4], v0, κ[4], RT)

        n_γ[1] ~ γfunc(n.p, p°, n_C0, n_Cbar, M[1], M0, v[1], v0, κ[1], RT)
        n_γ[2] ~ γfunc(n.p, p°, n_C0, n_Cbar, M[2], M0, v[2], v0, κ[2], RT)
        n_γ[3] ~ γfunc(n.p, p°, n_C0, n_Cbar, M[3], M0, v[3], v0, κ[3], RT)
        n_γ[4] ~ γfunc(n.p, p°, n_C0, n_Cbar, M[4], M0, v[4], v0, κ[4], RT)

        dϕ ~ p.ϕ - n.ϕ
        dp ~ p.p - n.p
        dμex[1] ~ dμex_func(p_γ[1], n_γ[1], RT, eps_reg)
        dμex[2] ~ dμex_func(p_γ[2], n_γ[2], RT, eps_reg)
        dμex[3] ~ dμex_func(p_γ[3], n_γ[3], RT, eps_reg)
        dμex[4] ~ dμex_func(p_γ[4], n_γ[4], RT, eps_reg)
        
        Nflux[1] ~ p.Nflux[1]
        Nflux[2] ~ p.Nflux[2]
        Nflux[3] ~ p.Nflux[3]
        Nflux[4] ~ p.Nflux[4]

        Nflux[1] ~ -n.Nflux[1]
        Nflux[2] ~ -n.Nflux[2]
        Nflux[3] ~ -n.Nflux[3]
        Nflux[4] ~ -n.Nflux[4]

        Nflux[1] ~ A / h * sedan_flux(p.C[1], n.C[1], dμex[1], dϕ, z[1], F, RT, DC[1])
        Nflux[2] ~ A / h * sedan_flux(p.C[2], n.C[2], dμex[2], dϕ, z[2], F, RT, DC[2])
        Nflux[3] ~ A / h * sedan_flux(p.C[3], n.C[3], dμex[3], dϕ, z[3], F, RT, DC[3])
        Nflux[4] ~ A / h * sedan_flux(p.C[4], n.C[4], dμex[4], dϕ, z[4], F, RT, DC[4])

        Dflux ~ p.Dflux
        Dflux ~ -n.Dflux
        Dflux ~ A / h * (-eps_r * eps_0 * dϕ)

        sflux ~ p.sflux
        sflux ~ -n.sflux
        sflux ~ A / h * (dp + (p_qf + n_qf) / 2 * dϕ)
    end
end

@mtkmodel ElectrolyteBulkBoundary begin
    @extend OneElectrolyteInterface(; κ, eps_r, v, Gf, M, DC, RT, z)
    @parameters begin
        ϕ_bulk = 0.0,  [description = "electric potential in the bulk", unit = u"V"]
        p_bulk = 0.0,  [description = "pressure in the bulk", unit = u"Pa"]
        
        A,             [description = "area", unit = u"m^2"]
        h,             [description = "distance between collocation points", unit = u"m"]
        #C_bulk[1:4] = [0.001, 0.01, 0.1, 1],[description = "bulk concentrations", unit = u"mol / dm^3"]
        C_bulk[1:4],   [description = "bulk concentrations", unit = u"mol / m^3"]
        ϵ = 1.0e-10,   [unit = u"1"]
    end
    @variables begin
        dϕ(t),          [unit = u"V"]
        dp(t),          [unit = u"Pa"]
        dC(t)[1:4],     [unit = u"mol / m^3"]
    end
    @equations begin
        dϕ ~ ϕ_bulk - ei.ϕ
        dp ~ p_bulk - ei.p
        dC[1] ~ C_bulk[1] - ei.C[1]
        dC[2] ~ C_bulk[2] - ei.C[2]
        dC[3] ~ C_bulk[3] - ei.C[3]
        dC[4] ~ C_bulk[4] - ei.C[4]

        ei.Nflux[1] ~ A / h * (DC[1] * dC[1] / ϵ)
        ei.Nflux[2] ~ A / h * (DC[2] * dC[2] / ϵ)
        ei.Nflux[3] ~ A / h * (DC[3] * dC[3] / ϵ)
        ei.Nflux[4] ~ A / h * (DC[4] * dC[4] / ϵ)
        ei.Dflux ~ A / h * (-eps_r * eps_0 * dϕ / ϵ)
        ei.sflux ~ A / h * (dp + qf * dϕ) / ϵ
    end
end

@mtkmodel ElectrodeElectrolyteBoundary begin
    @extend OneElectrolyteInterface(; κ, eps_r, v, Gf, M, DC, RT, z)
    @constants begin
        β           = 0.5,                  [description = "transfer coefficient"]
        trunc       = 20.0
    end
    @parameters begin
        sc[1:4]         ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
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
        dϕ ~ ϕ_we - ei.ϕ
        dG ~ sc[1] * μ[1] + sc[2] * μ[2] + sc[3] * μ[3] + sc[4] * μ[4] - F * dϕ #sum([sci * μi for (sci, μi) in zip(sc, μ)]) - F * dϕ
        
        RR ~ rrate(R0, β, -dG, RT, trunc)
        
        ei.Nflux[1] ~ 0.0
        ei.Nflux[2] ~ -RR * A
        ei.Nflux[3] ~ RR * A
        ei.Nflux[4] ~ 0.0
        
        #ei.Dflux ~ A / h * (-eps_r * eps_0 * dϕ)
        ei.Dflux ~ A / h * (-eps_r * eps_0 * dϕ / ϵ)
        ei.sflux ~ 0.0
    end
end

@mtkmodel System begin
    @parameters begin
        ϕ_we        ,[description = "electric potential at working electrode", unit = u"V"]
        ϕ_bulk      ,[description = "electric potential in the bulk", unit = u"V"]
        p_bulk      ,[description = "pressure in the bulk", unit = u"Pa"]
        C_bulk[1:4] ,[description = "bulk concentrations", unit = u"mol / m^3"]
        A           ,[description = "crossectional area", unit = u"m^2"]
        κ[1:4]      ,[description = "solvation number", unit=u"1"]
        eps_r       ,[unit = u"1"]
        v[1:4]      ,[unit = u"m^3 / mol"]
        Gf[1:4]     ,[unit = u"J / mol"]
        M[1:4]      ,[unit = u"kg / mol"]
        RT          ,[unit = u"m^2 * kg / s^2 / mol"]
        DC[1:4]     ,[unit = u"m^2 / s"]
        z[1:4]      ,[description = "charge numbers", unit = u"1"]
        sc[1:4]     ,[description = "stoichiometric coefficients of the boundary reaction", unit=u"1"]
    end
    @components begin
        # e = Electrolyte(;
        #     eps_r   = 1.0                               ,#[unit = u"1"]
        #     v       = [1.0, 1.0, 1.0, 1.0] ./ 55.4      ,#[unit = u"dm^3/mol"]
        #     κ       = [0, 0, 0, 0]                      ,#[unit = u"1"]
        #     Gf      = [0.0, 0.0, 0.0, 0.0]              ,#[unit = u"J / mol"]
        #     M       = [1.0, 1.0, 1.0, 1.0] .* 18.0153   ,#[unit = u"g / mol"]
        #     DC      = fill(2.0e-9, 4)                   ,#[unit = u"m^2 / s"]
        #     RT      = 8.31446261815324 * 298.0   
        # )
        eeb = ElectrodeElectrolyteBoundary(; A, h=0.5e-3, ϕ_we=ϕ_we, κ=scalarize(κ), eps_r=eps_r, v=v, Gf=Gf, M=M, RT=RT, DC=DC, z=z, sc=sc)
        cv1 = ElectrolyteControlVolume(; V=1.0e-3, κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
        b12 = ElectrolyteControlVolumeBoundary(; A=A, h=1.0e-3, κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
        cv2 = ElectrolyteControlVolume(; V=1.0e-3, κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
        b23 = ElectrolyteControlVolumeBoundary(; A=A, h=1.0e-3, κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
        cv3 = ElectrolyteControlVolume(; V=1.0e-3, κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
        ebb = ElectrolyteBulkBoundary(; A=A, h=1.0e-3, C_bulk, κ=κ, eps_r=eps_r, v=v, Gf=Gf, M=M, DC=DC, RT=RT, z=z)
    end
    @equations begin
        connect(eeb.ei, cv1.ei)
        connect(cv1.ei, b12.n)
        connect(b12.p, cv2.ei)
        connect(cv2.ei, b23.n)
        connect(b23.p, cv3.ei)
        connect(cv3.ei, ebb.ei)
    end
end

defaults = Dict(
    "ϕ_we" => -1.0,
    "ϕ_bulk" => 0.0,
    "p_bulk" => 0.0,
    "C_bulk" => [1.0, 0.1, 0.1, 0.75] .* 1.0e3,
    "κ" => [0.0, 0, 0, 0],
    "A" => 1.0,
    "eps_r" => 1.0,
    "v" => [1.0, 1.0, 1.0, 1.0] ./ 55.4 * 1.0e-3,
    "Gf" => [0.0, 0.0, 0.0, 0.0],
    "M" => [1.0, 1.0, 1.0, 1.0] .* 18.0153 * 1.0e-3,
    "RT" => 8.31446261815324 * 298.0,
    "DC" => [1.0, 1.0, 1.0, 1.0] * 1.0e-9,
    "z" => [1, 2, 3, -2],
    "sc" => [0, -1, 1, 0]
)

@mtkbuild test = System(; 
       ϕ_we=defaults["ϕ_we"], 
       ϕ_bulk=defaults["ϕ_bulk"], 
       p_bulk=defaults["p_bulk"],
       C_bulk=defaults["C_bulk"],
       κ=defaults["κ"],
       A=defaults["A"],
       eps_r=defaults["eps_r"],
       v=defaults["v"],
       Gf=defaults["Gf"],
       M=defaults["M"],
       RT=defaults["RT"],
       DC=defaults["DC"],
       z=defaults["z"]
)

# export ElectrolyteControlVolume
# export ElectrolyteControlVolumeBoundary
# export ElectrolyteBulkBoundary
# export ElectrodeElectrolyteBoundary

# end