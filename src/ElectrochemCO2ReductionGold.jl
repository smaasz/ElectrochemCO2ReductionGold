module ElectrochemCO2ReductionGold

using LiquidElectrolytes, ExtendableGrids, PreallocationTools, LessUnitful, VoronoiFVM, ForwardDiff 

include("fully_resolved.jl")
export nc, na, iK⁺, iH⁺, iHCO₃⁻, iCO₃²⁻, iCO₂, iOH⁻, iCO, iCOOH, S
export simulate, ActivityType, pressureconstrained, latticeconstrained

end