using DataFrames, CSV

include("./simulation/functions.jl")

function run_rft_rate_sim(mean_intervals::Array{Float64, 1})
    data_IRT = DataFrame(cond = Float64[], IRT = Float64[])
    data_qf = DataFrame(cond = Float64[], q_pref = Float64[], q_cost = Float64[])
    data_choice = DataFrame(cond = Float64[], state = Int64[])
    data_isop = DataFrame(cond = Float64[], state = Int64[])
    data_isot = DataFrame(cond = Float64[], state = Int64[])

    for mi in mean_intervals
        Ir, Ie, qp, qc, cs, isop, isot = simulation(mean_interval = mi)
        data_IRT = vcat(data_IRT, DataFrame(cond = [mi for i in 1:length(Ir)], IRT = Ir))
        data_qf = vcat(data_qf, DataFrame(cond = [mi for i in 1:length(qp)], q_pref = qp, q_cost = qc))
        data_choice = vcat(data_choice, DataFrame(cond = [mi for i in 1:length(cs)], state = cs))
        data_isop = vcat(data_isop, DataFrame(cond = [mi for i in 1:length(isop)], state = isop))
        data_isot = vcat(data_isot, DataFrame(cond = [mi for i in 1:length(isot)], state = isot))
    end
    [data_IRT, data_qf, data_choice, data_isop, data_isot]
end

function run_deprivation_sim(deprivation_level::Array{Float64, 1})
    data_IRT = DataFrame(cond = Float64[], IRT = Float64[])
    data_qf = DataFrame(cond = Float64[], q_pref = Float64[], q_cost = Float64[])
    data_choice = DataFrame(cond = Float64[], state = Int64[])
    data_isop = DataFrame(cond = Float64[], state = Int64[])
    data_isot = DataFrame(cond = Float64[], state = Int64[])

    for dep in deprivation_level
        Ir, Ie, qp, qc, cs, isop, isot = simulation(reward_operant = dep)
        data_IRT = vcat(data_IRT, DataFrame(cond = [dep for i in 1:length(Ir)], IRT = Ir))
        data_qf = vcat(data_qf, DataFrame(cond = [dep for i in 1:length(qp)], q_pref = qp, q_cost = qc))
        data_choice = vcat(data_choice, DataFrame(cond = [dep for i in 1:length(cs)], state = cs))
        data_isop = vcat(data_isop, DataFrame(cond = [dep for i in 1:length(isop)], state = isop))
        data_isot = vcat(data_isot, DataFrame(cond = [dep for i in 1:length(isot)], state = isot))
    end
    [data_IRT, data_qf, data_choice, data_isop, data_isot]
end

function run_VR_sim(VR_values::Array{Int64, 1})
    data_IRT = DataFrame(cond = Int64[], IRT = Float64[])
    data_qf = DataFrame(cond = Int64[], q_pref = Float64[], q_cost = Float64[])
    data_choice = DataFrame(cond = Int64[], state = Int64[])
    data_isop = DataFrame(cond = Int64[], state = Int64[])
    data_isot = DataFrame(cond = Int64[], state = Int64[])

    for VR in VR_values
        Ir, Ie, qp, qc, cs, isop, isot = simulation(tand_VR = VR)
        data_IRT = vcat(data_IRT, DataFrame(cond = [VR for i in 1:length(Ir)], IRT = Ir))
        data_qf = vcat(data_qf, DataFrame(cond = [VR for i in 1:length(qp)], q_pref = qp, q_cost = qc))
        data_choice = vcat(data_choice, DataFrame(cond = [VR for i in 1:length(cs)], state = cs))
        data_isop = vcat(data_isop, DataFrame(cond = [VR for i in 1:length(isop)], state = isop))
        data_isot = vcat(data_isot, DataFrame(cond = [VR for i in 1:length(isot)], state = isot))
    end
    [data_IRT, data_qf, data_choice, data_isop, data_isot]
end


function run_extinction_sim(ext::Bool)
    Ir, Ie, qp, qc, cs, isop, isot = simulation(ext = ext)
    data_rft = DataFrame(cond = ["rft" for i in 1:length(Ir)], IRT = Ir)
    data_ext = DataFrame(cond = ["ext" for i in 1:length(Ie)], IRT = Ie)
    data_qf = DataFrame(cond = ["rft" for i in 1:length(qp)], q_pref = qp, q_cost = qc)
    data_choice = DataFrame(cond = ["rft" for i in 1:length(cs)], state = cs)
    data_isop = DataFrame(cond = ["rft" for i in 1:length(isop)], state = isop)
    data_isot = DataFrame(cond = ["rft" for i in 1:length(isot)], state = isot)
    [data_rft, data_ext, data_qf, data_choice, data_isop, data_isot]
end

mean_intervals = Float64[30., 120., 480.]
deprivation_levels = Float64[.5, 1., 2.]
VRs = Int64[0, 4, 8]
ext = true

res_rft_rate = run_rft_rate_sim(mean_intervals)
res_dep_lev = run_deprivation_sim(deprivation_levels)
res_tand_VR = run_VR_sim(VRs)
res_ext = run_extinction_sim(ext)

name_res = ["IRT", "qf", "choice", "isop", "isot"]
name_res_ext = ["rft_IRT", "ext_IRT", "qf", "choice", "isop", "isot"]

function wrtite_csv(sim::String, names::Array{String, 1}, res::Array{DataFrame, 1})
    for i in 1:length(names)
        CSV.write("../data/"*sim*"_"*names[i]*".csv", res[i])
    end
end

wrtite_csv("rft_rate", name_res, res_rft_rate)
wrtite_csv("dep_lev", name_res, res_dep_lev)
wrtite_csv("tand_VR", name_res, res_tand_VR)
wrtite_csv("ext", name_res_ext, res_ext)
