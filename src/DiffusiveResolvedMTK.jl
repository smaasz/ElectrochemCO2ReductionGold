module DiffusiveResolvedMTK begin
using ModelingToolkit, DynamicQuantities
using IfElse

rlog(x; eps=1.0e-20) = IfElse.ifelse(x < eps, log(eps) + (x - eps) / eps, log(x))

dμex_func(γk, γl, RT) = IfElse.ifelse(γk > γl, rlog(γk / γl) * RT, -rlog(γl / γk) * RT)

function γfunc(p, p°, c0, cbar, M, M0, v, v0, RT) 
    Mrel = M / M0
    vbar = v + κ * v0
    vrel = vbar - Mrel * v0
    return exp(vrel * (p - p°) / (RT)) * (cbar / c0)^Mrel * (1/cbar)
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

# Bernoulli thresholds optimized for Float64
const bernoulli_small_threshold = 0.25
const bernoulli_large_threshold = 40.0

function fbernoulli_pm(x)
    return IfElse.ifelse(
        x < -bernoulli_large_threshold,
        (-x, 0.0),
        IfElse.ifelse(
            x > bernoulli_large_threshold,
            (x, 0.0),
            IfElse.ifelse(
                abs(exp(x) - 1.0) > bernoulli_small_threshold,
                (x / (exp(x) - 1.0), x / (1.0 - 1.0 / (exp(x) - 1.0))),
                (bernoulli_horner(x), x + bernoulli_horner(x))
            )
        )
    )
end

function rexp(x; trunc = 20.0)
    IfElse.ifelse(
        x < -trunc,
        1.0 / rexp(-x; trunc),
        IfElse.ifelse(
            x <= trunc,
            exp(x),
            exp(trunc) * (x - trunc + 1)
        )
    )
end

rrate(R0, β, A) = R0 * (rexp(-β * A) - rexp((1 - β) * A))

@connector ElectrolyteInterface begin
    @constants begin
        R       = 8.31446261815324,     [unit = u"m^2 * kg / s^2 / K / mol"]
        p_bulk  = 0.0,                  [unit = u"Pa"]
        F       = 96485.33212331001,    [unit = u"s * A / mol"]
    end
    @parameters begin
        z[1:4]  = [1, 2, 3, -2]
        v0      = 1.0 / 55.4,                       [unit = u"dm^3 / mol"]
        v[1:4]  = [1.0, 1.0, 1.0, 1.0] ./ 55.4,     [unit = u"dm^3/mol"]
        κ[1:4]  = [0, 0, 0, 0]  
        C°      = 1.0,                              [unit = u"mol / dm^3"]
        Gf[1:4] = [0.0, 0.0, 0.0, 0.0],             [unit = u"J / mol"]
        M0      = 18.0153,                          [unit = u"g / mol"]
        M[1:4]  = [1.0, 1.0, 1.0, 1.0] .* 18.0153,  [unit = u"g / mol"]
        T       = 298.0                             [unit = u"K"]
        RT      = R * T,                            [unit = u"m^2 * kg / s^2 / mol"]
    end
    end
    @variables begin
        C(t)[1:4],      [description = "concentration", unit = u"mol / dm^3"]
        p(t),           [description = "pressure", unit = u"bar"]
        ϕ(t),           [description = "electrical potential", unit = u"V"]
        
        C0(t),          [description = "concentration of the solvent", unit = u"mol / dm^3"]
        Cbar(t),        [description = "total concentration", unit = u"mol / dm^3"]
        qf(t),          [description = "free charge density", unit = u"C / m^3"]
        γ(t)[1:4],      [description = "activity coefficient"]
        μ(t)[1:4],      [description = "chemical potential", unit = u"J / mol"]

        Nflux(t)[1:4],  [description = "molar mass flux", unit = u"mol / s", connect = flow]
        Dflux(t),       [description = "electric flux", unit = u"V * m", connect = flow]
        sflux(t),       [description = "stress gradient flux", unit = "kg / s^2", connect = flow]
    end
    @equations begin
        C0 ~ 1.0 / v0 - sum((v ./ v0 .+ κ) .* C)
        Cbar ~ C0 + sum(C)
        qf ~ sum(F * z .* C)
        γ[1] ~ γfunc(p, p°, C0, Cbar, M[1], M0, v[1], v0, RT)
        γ[2] ~ γfunc(p, p°, C0, Cbar, M[2], M0, v[2], v0, RT)
        γ[3] ~ γfunc(p, p°, C0, Cbar, M[3], M0, v[3], v0, RT)
        γ[4] ~ γfunc(p, p°, C0, Cbar, M[4], M0, v[4], v0, RT)
        μ[1] ~ Gf[1] + RT * rlog(γ[1] * C[1]/C°, RT)
        μ[2] ~ Gf[2] + RT * rlog(γ[2] * C[2]/C°, RT)
        μ[3] ~ Gf[3] + RT * rlog(γ[3] * C[3]/C°, RT)
        μ[4] ~ Gf[4] + RT * rlog(γ[4] * C[4]/C°, RT)
    end
end

@mtkmodel ElectrolyteControlVolume begin
    @parameters begin
        V, [description = "Volume", unit = u"m^3"]
    end
    @components begin
        ei = ElectrolyteInterface()
    end
    @variables begin
        C(t)[1:4],      [description = "concentration", unit = u"mol / dm^3"]
        p(t),           [description = "pressure", unit = u"bar"]
        ϕ(t),           [description = "electrical potential", unit = u"V"]
        qf(t),          [description = "free charge density", unit = u"C / m^3"]
        der_C(t)[1:4]   [unit = u"mol / dm^3 / s"]
    end
    @equations begin
        C[1] ~ ei.C[1]
        C[2] ~ ei.C[2]
        C[3] ~ ei.C[3]
        C[4] ~ ei.C[4]
        p ~ ei.p
        ϕ ~ ei.ϕ
        qf ~ ei.qf

        der_C ~ ei.Nflux ./ V
        
        D(C[1]) ~ der_C[1]
        D(C[2]) ~ der_C[2]
        D(C[3]) ~ der_C[3]
        D(C[4]) ~ der_C[4]

        qf ~ ei.Dflux / V
    end

    
end

@mtkmodel ElectrolyteControlVolumeBoundary begin
    @constants begin
        eps_0   = 8.8541878128e-12,     [unit = u"A^2 * s^4 / kg / m^3"]
        R       = 8.31446261815324,     [unit = u"m^2 * kg / s^2 / K / mol"]
        F       = 96485.33212331001,    [unit = u"s * A / mol"]
    end
    @parameters begin
        A,              [description = "Area of the control volume boundary part", unit = u"m^2"]
        h,              [description = "distance between collocation points", unit = u"m"]
        RT = 298 * R,   [unit = u"m^2 * kg / s^2 / mol"]
        eps_r = 1.0
    end
    @components begin
        n = ElectrolyteInterface()
        p = ElectrolyteInterface()
    end
    @variables begin
        dϕ(t),          [unit = u"V"]
        dp(t),          [unit = u"bar"]   
        dμex(t)[1:4]    [unit = u"J / mol"]

        Nflux(t)[1:4],  [description = "molar mass flux", unit = u"mol / s", connect = flow]
        Dflux(t),       [description = "electric flux", unit = u"V * m", connect = flow]
        sflux(t),       [description = "stress gradient flux", unit = "kg / s^2", connect = flow]
    end
    @equations begin
        dϕ ~ p.ϕ - n.ϕ
        dp ~ p.p - n.p
        dμex[1] ~ dμex_func(p.γ[1], n.γ[1], RT)
        dμex[2] ~ dμex_func(p.γ[2], n.γ[2], RT)
        dμex[3] ~ dμex_func(p.γ[3], n.γ[3], RT)
        dμex[4] ~ dμex_func(p.γ[4], n.γ[4], RT)
        
        Nflux[1] ~ p.Nflux[1]
        Nflux[2] ~ p.Nflux[2]
        Nflux[3] ~ p.Nflux[3]
        Nflux[4] ~ p.Nflux[4]

        Nflux[1] ~ -n.Nflux[1]
        Nflux[2] ~ -n.Nflux[2]
        Nflux[3] ~ -n.Nflux[3]
        Nflux[4] ~ -n.Nflux[4]

        Nflux[1] ~ A / h * fbernoulli_pm(z[1] * F * dϕ / RT + dμex[1] / RT)
        Nflux[2] ~ A / h * fbernoulli_pm(z[2] * F * dϕ / RT + dμex[2] / RT)
        Nflux[3] ~ A / h * fbernoulli_pm(z[3] * F * dϕ / RT + dμex[3] / RT)
        Nflux[4] ~ A / h * fbernoulli_pm(z[4] * F * dϕ / RT + dμex[4] / RT)

        Dflux ~ p.Dflux
        Dflux ~ -n.Dflux
        Dflux ~ A / h * (-eps_r * eps_0 * dϕ)

        sflux ~ p.sflux
        sflux ~ -n.sflux
        sflux ~ A / h * (dp + (p.qf + n.qf) / 2 * dϕ)
    end
    
end

@mtkmodel ElectrolyteBulkBoundary begin
    @constants begin
        ϕ_bulk = 0.0   [description = "electric potential in the bulk", unit = u"V"]
        p_bulk = 0.0   [description = "pressure in the bulk", unit = u"bar"]
    end
    @parameters begin
        A,                                  [description = "area", unit = u"m^2"]
        h,                                  [description = "distance between collocation points", unit = u"m"]
        C_bulk[1:4] = [0.001, 0.01, 0.1, 1] [description = "bulk concentrations", unit = u"mol / dm^3"]
        ϵ = 1.0e-10
    end
    @components begin
        ei = ElectrolyteInterface()
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

        ei.Nflux[1] ~ A / h * (dC[1] / ϵ)
        ei.Nflux[2] ~ A / h * (dC[2] / ϵ)
        ei.Nflux[3] ~ A / h * (dC[3] / ϵ)
        ei.Nflux[4] ~ A / h * (dC[4] / ϵ)
        ei.Dflux ~ A / h * (dϕ / ϵ)
        ei.sflux ~ A / h * (dp + ei.qf * dϕ) / ϵ
    end
end

@mtkmodel ElectrodeElectrolyteBoundary begin
    @constants begin
        sc[1:4]     = [0, -1, 1, 0]         [description = "stoichiometric coefficients of the boundary reaction"]
        R0          = 1.0e-10               [description = "exchange rate constant", unit = u"mol / cm^2 / s"]
        β           = 0.5                   [description = "transfer coefficient"]
        F           = 96485.33212331001,    [unit = u"s * A / mol"]
        R           = 8.31446261815324,     [unit = u"m^2 * kg / s^2 / K / mol"]
        eps_0       = 8.8541878128e-12,     [unit = u"A^2 * s^4 / kg / m^3"]
    end
    @parameters begin
        A,              [description = "area", unit = u"m^2"]
        h,              [description = "distance between collocation points", unit = u"m"]
        ϕ_we,           [description = "electric potential at working electrode", unit = u"V"]
        T = 298.0       [description = "temperature", unit = u"K"]
        RT = R * T,     [unit = u"m^2 * kg / s^2 / mol"]
        eps_r = 1.0
        ϵ = 1.0e-10
    end
    @components begin
        ei = ElectrolyteInterface()
    end
    @variables begin
        RR(t),  [description = "reaction rate", unit = u"mol / s"]
        dG(t),  [description = "change in Gibbs free energy per reaction unit", unit = u"J / mol"]
        dϕ(t),  [unit = u"V"]
    end
    @equations begin
        dϕ ~ ϕ_we - ei.ϕ
        dG ~ (sum(sc .* ei.μ) - F * dϕ) / RT
        
        RR ~ rrate(R0, β, -dμ)
        
        ei.Nflux[1] ~ 0.0
        ei.Nflux[2] ~ -R * A
        ei.Nflux[3] ~ R * A
        ei.Nflux[4] ~ 0.0
        
        #ei.Dflux ~ A / h * (-eps_r * eps_0 * dϕ)
        ei.Dflux ~ A / h * (dϕ / ϵ)
        ei.sflux ~ 0.0
    end
end

export ElectrolyteControlVolume
export ElectrolyteControlVolumeBoundary
export ElectrolyteBulkBoundary
export ElectrodeElectrolyteBoundary

end