using StatsBase
using Distributions

function update_q_pref(q_pref, alpha,reward_val)
    return q_pref + alpha * (reward_val - q_pref)
end


function update_q_cost(q_cost, alpha, resp)
    return q_cost + alpha * (log(resp + 1) - q_cost)
end

function softmax(beta, q_pref_1, q_pref_2)
    q_max = ifelse(q_pref_1 >= q_pref_2, q_pref_1, q_pref_2)
    q_pref_1_beta = exp(beta * (q_pref_1 - q_max))
    q_pref_2_beta = exp(beta * (q_pref_2 - q_max))
    return q_pref_1_beta / (q_pref_1_beta + q_pref_2_beta)
end

function calc_p_stay(w_pref, w_cost, q_pref, q_cost)
    return exp(-1 / (w_pref * q_pref + w_cost * q_cost))
end

function create_VI_table(mean_interval, table_length)
    rate = 1 / mean_interval
    N = table_length
    VI_table = Float64[]
    for n in 1:N-1
        push!(VI_table, ((-log(1 - rate))^ -1)* (1 + log(N) + (N - n)* log(N - n) - (N - n + 1)* log(N - n + 1)))
    end
    push!(VI_table, ((-log(1 - rate))^ -1)* (1 + log(N) - (N - N + 1)* log(N - N + 1)))
    return VI_table
end

function rbern(p)
    return rand(Bernoulli(p))
end

function simulation(;mean_interval= 120.0, table_length = 20, tand_VR = 0, reward_operant = 1.0,
    reward_other = .5, within_bout = 1/3, max_reward = 1000, ext = false, ext_duration = 3600.0,
    alpha_rft = .05, alpha_ext = .01, beta = 12.5, w_pref = 1.0, w_cost = 3.5)

    q_pref_1 = .0; q_pref_2 = .0
    q_cost_1 = .0; q_cost_2 = .0
    state = 0
    resp_operant = 0; resp_other = 1
    IRT = .0
    t_in_IRI = .0
    t_in_session = .0
    obtained_reward = 0

    IRT_rft_phase = Float64[]
    IRT_ext_pahse = Float64[]
    q_pref_1_rft_phase = Float64[]
    q_pref_2_rft_pahse = Float64[]
    q_cost_1_rft_pahse = Float64[]
    q_cost_2_rft_phase = Float64[]
    chosed_state = Int64[]
    is_stayed_operant = Int64[]
    is_stayed_other = Int64[]

    VI_table = create_VI_table(mean_interval, table_length)
    half_max_reward = max_reward / 2

    while true

        IRI = sample(VI_table)
        t_in_IRI = 0.0
        IRT = 0.
        VR = 1

        if tand_VR > 0
            VR += rand(Geometric(1 / tand_VR))
        end

        while true

            t_in_IRI += .1
            IRT += .1

            if state == 0
                state = ifelse(rbern(softmax(beta, q_pref_1, q_pref_2)) == 1, 1, 2)
                if obtained_reward >= half_max_reward
                    push!(chosed_state, state)
                end
            elseif state == 1
                if rbern(within_bout) == 1
                    resp_operant += 1
                    if obtained_reward >= half_max_reward
                        push!(IRT_rft_phase, IRT)
                        push!(q_pref_1_rft_phase, q_pref_1)
                        push!(q_pref_2_rft_pahse, q_pref_2)
                        push!(q_cost_1_rft_pahse, q_cost_1)
                        push!(q_cost_2_rft_phase, q_cost_2)
                    end
                    IRT = 0.0
                    state = rbern(calc_p_stay(w_pref, w_cost, q_pref_1, q_cost_1))
                    if obtained_reward >= half_max_reward
                        push!(is_stayed_operant, state)
                    end
                    if state == 0
                        q_pref_1 = update_q_pref(q_pref_1, alpha_ext, .0)
                    end
                end
            elseif state == 2
                if rbern(within_bout) == 1
                    resp_operant = 0
                    if obtained_reward >= half_max_reward
                        push!(q_pref_1_rft_phase, q_pref_1)
                        push!(q_pref_2_rft_pahse, q_pref_2)
                        push!(q_cost_1_rft_pahse, q_cost_1)
                        push!(q_cost_2_rft_phase, q_cost_2)
                    end
                    q_pref_2 = update_q_pref(q_pref_2, alpha_rft, reward_other)
                    q_cost_2 = update_q_cost(q_cost_2, alpha_rft, resp_other)
                    state = ifelse(rbern(calc_p_stay(w_pref, w_cost, q_pref_2, q_cost_2)) == 1, 2, 0)
                    if obtained_reward >= half_max_reward
                        push!(is_stayed_other, state)
                    end
                end
            end
            if t_in_IRI >= IRI
                break
            end
        end

        while true

            t_in_IRI += .1
            IRT += .1

            if state == 0
                state = ifelse(rbern(softmax(beta, q_pref_1, q_pref_2)) == 1, 1, 2)
                if obtained_reward >= half_max_reward
                    push!(chosed_state, state)
                end
            elseif state == 1
                if rbern(within_bout) == 1
                    resp_operant += 1
                    VR -= 1
                    if obtained_reward >= half_max_reward
                        push!(IRT_rft_phase, IRT)
                        push!(q_pref_1_rft_phase, q_pref_1)
                        push!(q_pref_2_rft_pahse, q_pref_2)
                        push!(q_cost_1_rft_pahse, q_cost_1)
                        push!(q_cost_2_rft_phase, q_cost_2)
                    end
                    IRT = 0.0
                    if VR == 0
                        obtained_reward += 1
                        q_pref_1 = update_q_pref(q_pref_1, alpha_rft, reward_operant)
                        q_cost_1 = update_q_cost(q_cost_1, alpha_rft, resp_operant)
                        resp_operant = 1
                        state = 0
                        break
                    end
                    if VR > 0
                        state = rbern(calc_p_stay(w_pref, w_cost, q_pref_1, q_cost_1))
                        if state == 0
                            q_pref_1 = update_q_pref(q_pref_1, alpha_ext, .0)
                        end
                        if obtained_reward >= half_max_reward
                            push!(is_stayed_operant, state)
                        end
                    end
                end
            elseif state == 2
                if rbern(within_bout) == 1
                    resp_operant = 0
                    if obtained_reward >= half_max_reward
                        push!(q_pref_1_rft_phase, q_pref_1)
                        push!(q_pref_2_rft_pahse, q_pref_2)
                        push!(q_cost_1_rft_pahse, q_cost_1)
                        push!(q_cost_2_rft_phase, q_cost_2)
                    end
                    q_pref_2 = update_q_pref(q_pref_2, alpha_rft, reward_other)
                    q_cost_2 = update_q_cost(q_cost_2, alpha_rft, resp_other)
                    state = ifelse(rbern(calc_p_stay(w_pref, w_cost, q_pref_2, q_cost_2)) == 1, 2, 0)
                    if obtained_reward >= half_max_reward
                        push!(is_stayed_other, state)
                    end
                end
            end
        end

        if obtained_reward >= max_reward 
            break
        end
    end

    if ext == true
        t_in_ext = .0
        while true
            IRT += .1
            t_in_ext += .1
            if state == 0
                state = ifelse(rbern(softmax(beta, q_pref_1, q_pref_2)) == 1, 1, 2)
                push!(chosed_state, state)
            elseif state == 1
                if rbern(within_bout) == 1
                    resp_operant += 1
                    push!(IRT_ext_pahse, IRT)
                    IRT = 0.0
                    state = rbern(calc_p_stay(w_pref, w_cost, q_pref_1, q_cost_1))
                    if state == 0
                        q_pref_1 = update_q_pref(q_pref_1, alpha_ext, .0)
                    end
                end
            elseif state == 2
                if rbern(within_bout) == 1
                    resp_operant = 0
                    q_pref_2 = update_q_pref(q_pref_2, alpha_rft, reward_other)
                    q_cost_2 = update_q_cost(q_cost_2, alpha_rft, resp_other)
                    state = ifelse(rbern(calc_p_stay(w_pref, w_cost, q_pref_2, q_cost_2)) == 1, 2, 0)
                end
            end
            if t_in_ext >=ext 
                break
            end
        end
    end
    IRT_rft_phase, IRT_ext_pahse, q_pref_1_rft_phase, q_cost_1_rft_pahse, chosed_state, is_stayed_operant, is_stayed_other
end
