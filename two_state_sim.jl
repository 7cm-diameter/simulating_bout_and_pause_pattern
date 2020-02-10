using StatsBase, Distributions
using CSV
using DataFrames


@enum ScheduleType FI VI FR VR

mutable struct Schedule
    schedule::ScheduleType
    value::Float64
    requirement::Float64
end

function init_schedule(comp::ScheduleType, val::Float64)
    Schedule(comp, val, 0.0)
end

mutable struct Combination
    component::Array{Schedule, 1}
    current_component::Int64
end


mutable struct Experiment
    schedules::Array{Combination, 1}
    session_timer::Float64
    response_counter::Float64
    reward_counter::Array{Int64, 1}
end

function init_experiment(schedule_type::Array{Combination, 1})
    function closure(num_state::Int64)
        Experiment(schedule_type, 0.0, Float64(1), repeat(Float64[1], num_state))
    end
end


mutable struct Environment
    num_state::Int64
    state_set::Array{Int64, 1}
    rewards::Array{Float64, 1}
    experiment::Experiment
    reward_availablity::Array{Bool, 1}
end

function init_env(rewards::Array{Float64, 1}, exp_gen::Function)
    ns = length(rewards)
    experiment = exp_gen(Int64(ns))
    Environment(ns, [1:1:ns; ], rewards, experiment, repeat([false], ns))
end


mutable struct Parameters
    alpha_reward::Float64
    alpha_extinction::Float64
    weigth_preferences::Float64
    weigth_costs::Float64
    response_prob::Float64
end

function init_params(ar::Float64, ae::Float64, wp::Float64, wc::Float64, rp::Float64)
    Parameters(ar, ae, wp, wc, rp)
end

mutable struct Agent
    params::Parameters
    preferences::Array{Float64, 1}
    costs::Array{Float64, 2}
    state::Int64
end

function init_agent(params, num_state::Int64)
    Agent(params, zeros(num_state), zeros((num_state, num_state)), 1)
end

function update_preference(agent::Agent, rewards::Array{Float64, 1})
    agent.preferences[agent.state] += agent.params.alpha_reward *
        (rewards[agent.state] - agent.preferences[agent.state])
end

function extinction(agent::Agent)
    agent.preferences[agent.state] += agent.params.alpha_extinction *
        (0.0 - agent.preferences[agent.state])
end

function update_cost(agent::Agent, response::Float64)
    agent.costs[agent.state, agent.state] += agent.params.alpha_reward *
        (log(response) - agent.costs[agent.state, agent.state])
end

function diff_max(qs)
    diffs = Float64[]
    qmax = maximum(qs)
    for q in qs
        push!(diffs, q - qmax)
    end
    return diffs
end

function exp_q(qs, weigth)
    exp_invs = Float64[]
    for q in qs
        push!(exp_invs, exp(weigth * q))
    end
    return exp_invs
end

function calc_probs(prefs, costs)
    probs = Float64[]
    sum_qs = sum(prefs) + sum(costs)
    for i in 1:length(prefs)
        push!(probs, (prefs[i] + costs[i]) / sum_qs)
    end
    return probs
end

function softmax(agent::Agent)
    diff_prefs = agent.preferences
    diff_costs = agent.costs[agent.state, :]
    exp_inv_prefs = exp_q(diff_prefs, agent.params.weigth_preferences)
    exp_inv_costs = exp_q(diff_costs, agent.params.weigth_costs)
    probs = calc_probs(exp_inv_prefs, exp_inv_costs)
    return weights(probs)
end

function variable_interval(mean_interval::Float64)
    interval = Exponential(mean_interval)
    return rand(interval)
end

function fixed_interval(interval::Float64)
    return interval
end

function variable_ratio(mean_ratio::Float64)
    rate = Geometric(1 / mean_ratio)
    rand(rate) + 1
end

function fixed_ratio(ratio::Float64)
    return ratio
end

function get_requirement(schedule::Schedule)
    schedule_type = schedule.schedule
    value = schedule.value
    if schedule_type == FI
        fixed_interval(value)
    elseif schedule_type == VI
        variable_interval(value)
    elseif schedule_type == FR
        fixed_ratio(value)
    elseif schedule_type == VR
        variable_ratio(value)
    end
end

function update_requirements(agent::Agent, env::Environment)
    state = agent.state
    components = env.experiment.schedules[state].component
    env.experiment.schedules[state].current_component = 1
    for i in 1:length(components)
        requirement = get_requirement(components[i])
        env.experiment.schedules[state].component[i].requirement = requirement
        env.reward_availablity[state] = false
    end
end

function elapse(env::Environment, time_step::Float64)
    env.experiment.session_timer += time_step
    schedules = env.experiment.schedules
    for i in 1:length(schedules)
        comp = schedules[i].current_component
        st = schedules[i].component[comp].schedule
        if  st == VI || st == FI
            env.experiment.schedules[i].component[comp].requirement -= time_step
        end
    end
end

function update_availability(env::Environment)
    schedules = env.experiment.schedules
    for i in 1:length(schedules)
        num_comp = length(schedules[i].component)
        cur_comp = schedules[i].current_component
        if num_comp == cur_comp
            if schedules[i].component[cur_comp].requirement <= 0.0
                env.reward_availablity[i] = true
            else
                env.reward_availablity[i] = false
            end
        end
        if num_comp > cur_comp
            if schedules[i].component[cur_comp].requirement <= 0.0
                env.experiment.schedules[i].current_component += 1
            end
        end
    end
end

function is_available(agent::Agent, env::Environment)
    env.reward_availablity[agent.state]
end

function update_reward_count(agent::Agent, env::Environment)
    env.experiment.reward_counter[agent.state] += 1
end

# agent
function emit_response(agent::Agent, env::Environment)
    b = Bool(rand(Bernoulli(agent.params.response_prob)))
    state = agent.state
    if b
        env.experiment.response_counter += 1.0
        schedule = env.experiment.schedules[state]
        comp = schedule.current_component
        st = schedule.component[comp].schedule
        if st == FR || st == VR
            env.experiment.schedules[state].component[comp].requirement -= 1.0
        end
    end
    return b
end

function choose_state(agent::Agent, env::Environment)
    prev = agent.state
    probs = softmax(agent)
    agent.state = sample(env.state_set, probs)
    if prev != agent.state
        env.experiment.response_counter = 1.0
    end
end

function run_simulation(agent::Agent, env::Environment, terminate::Int64)
    IRTs = Float64[]
    while env.experiment.reward_counter[1] <= terminate
        elapse(env, 0.1)
        if emit_response(agent, env)
            if env.experiment.reward_counter[1] >= 500 && agent.state == 1
                push!(IRTs, env.experiment.session_timer)
            end
            update_availability(env)
            if is_available(agent, env)
                update_preference(agent, env.rewards)
                update_cost(agent, env.experiment.response_counter)
                update_reward_count(agent, env)
                update_requirements(agent, env)
                choose_state(agent, env)
                env.experiment.response_counter = 1.0
            else
                prev = agent.state
                choose_state(agent, env)
                next = agent.state
                if env.experiment.response_counter == 1.0
                    agent.state = prev
                    extinction(agent)
                    agent.state = next
                end
            end
        end
        update_availability(env)
    end
    IRTs
end

function count_IRTs(response_time::Array{Float64, 1})
    freq = Float64[]
    IRTs = round.(diff(response_time), digits = 1)
    sort!(IRTs)
    unique_IRTs = unique(IRTs)
    counts = countmap(IRTs)
    for i in unique_IRTs
        push!(freq, counts[i])
    end
    (unique_IRTs, freq)
end

vi = init_schedule(VI, 480.0)
fr = init_schedule(FR, 1.0)
target = Combination([vi, fr], 1)
alt = Combination([fr], 1)

gen_exp = init_experiment(Combination[target, alt])
env = init_env(Float64[1, .5], gen_exp)

params = init_params(0.01, 0.01, 4.0, 3.5, 1 / 3)
agent = init_agent(params, Int64(2))
update_requirements(agent, env)

response_time = run_simulation(agent, env, Int64(1000))

IRTs, freq = count_IRTs(response_time)
svr = log10.(1 .- (cumsum(freq) / sum(freq)))
pop!(IRTs); pop!(svr)

condition = repeat(["480"], length(freq) - 1)

res = DataFrame(cond = condition, IRT = IRTs, svr = svr)
res |> CSV.write("./two_state_VI480.csv", delim = ",")
