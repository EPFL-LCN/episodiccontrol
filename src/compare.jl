include("episodiccontrol.jl")
include(joinpath(Pkg.dir("TabularReinforcementLearning"),
                 "examples/mdpexamples.jl"))
loadcomparisontools()

const DATADIR = joinpath(@__DIR__, "data")
if !isdir(DATADIR); mkdir(DATADIR); end

function getbaselineperformances(env, γ, policy)
    mdpl = MDPLearner(env, γ)
    policy_iteration!(mdpl)
    xmax = RLSetup(Agent(mdpl, policy = policy), env, MeanReward(), 
                   ConstantNumberSteps(10^5))
    run!(xmax)
    xmin = RLSetup(Agent(mdpl, policy = EpsilonGreedyPolicy(1.)), env, MeanReward(), 
                   ConstantNumberSteps(10^5))
    run!(xmin)
    getvalue(xmin.metric), getvalue(xmax.metric)
end

function getagents(env; γ = 1., ϵ = .1, αql = 1., αnstep = .03 * αql, 
                   αlambda = .01 * αql, explorationinit = 1.,
                   λ = 1., nsteps = 10, explorationinitps = explorationinit,
                   explorationinitql = explorationinit)
    policy = EpsilonGreedyPolicy(ϵ)
    params = ((:na, env.na), (:ns, env.ns), (:γ, γ), (:initvalue, Inf64))
    ql() = Agent(QLearning(; params..., λ = 0, α = αql), 
                 policy = policy)
    qlexplore() = Agent(QLearning(; params..., λ = 0, α = αql, 
                                  initvalue = explorationinitql), 
                 policy = policy)
    qllambda() = Agent(QLearning(; params..., λ = λ, α = αlambda), 
                       policy = policy)
    nstepql() = Agent(NstepLearner(; nsteps = nsteps, learner = Sarsa, 
                                     params..., α = αnstep), 
                      policy = policy)
    ec() = Agent(EpisodicControl(; params...), policy = policy)
    ps() = Agent(SmallBackups(; params...), policy = policy)
    mc() = Agent(MonteCarlo(; params[1:end-1]...), policy = policy)
    psexplore() = Agent(SmallBackups(; params..., initvalue = explorationinitps), 
                        policy = policy)
    psreset() = Agent(SmallBackups(; params..., maxcount = 0), 
                      policy = policy, 
                      callback = ModelReset())
    policy, Dict("ql" => ql, "qlexplore" => qlexplore, "qllambda" => qllambda,
              "nstepql" => nstepql, "ec" => ec, "ps" => ps, "mc" => mc,
              "psexplore" => psexplore, "psreset" => psreset)
end

function scaleres!(res, env, γ, policy)
    rmin, rmax = getbaselineperformances(env, γ, policy)
    res[:value] .-= rmin
    res[:value] .*= 1/(rmax - rmin)
    res
end

function eccompare(getenv, N, T; agents = "all", γ = 1., relperf = true, 
                   offset = 1., kargs...)
    env = getenv()
    env.reward .+= offset
    policy, agentdict = getagents(env; γ = γ, kargs...)
    as = agents == "all" ? collect(values(agentdict)) : [agentdict[a] for a in agents]
    res = compare(N, env, EvaluationPerT(div(T, 100)), ConstantNumberSteps(T), as...)
    relperf ? scaleres!(res, env, γ, policy) : res
end
function paramscompare(getenv, N, T;  offset = 1., params = Dict(:αql => (1., .1)), 
                                      agent = "ql", γ = 1., kargs...)
    env = getenv()
    env.reward .+= offset
    as = []
    policy = Any
    for (key, values) in params
        for value in values
            policy, agentdict = getagents(env; (kargs..., (key, value))...)
            push!(as, agentdict[agent])
        end
    end
    res = compare(N, env, EvaluationPerT(div(T, 100)), ConstantNumberSteps(T),
                  as...)
    scaleres!(res, env, γ, policy)
end

using PyPlot
labels = Dict("QLearning_2" => "QL with expl", "QLearning" => "Qlambda",
              "SmallBackups_1" => "PS with explore", 
              "SmallBackups_2" => "Model reset", 
              "EpisodicLearner" => "MonteCarlo",
              "EpisodicLearner_1" => "EpisodicControl")

@time res1 = vcat([eccompare(getdettreemdp, 5, 100*200, γ = 1., ϵ = .1, 
                             explorationinit = 5., αnstep = 8e-2, 
                             αlambda = 1, λ = .2) 
                   for _ in 1:100]...);
figure(); plotcomparison(res1, labelsdict = labels); plt[:ylim]([0, 1]); plt[:title]("1")

@time res2 = vcat([eccompare(getdettreemdpwithinrew, 5, 200*100, γ = 1., ϵ = .1,
                             explorationinit = 5., αnstep = 8e-2, 
                             αlambda = 1, λ = .2) 
                   for _ in 1:100]...);
figure(); plotcomparison(res2, labelsdict = labels); plt[:ylim]([0, 1]); plt[:title]("2")

@time res3 = vcat([eccompare(getstochtreemdp, 50, 100*100, γ = 1., ϵ = .1, αql = .1,
                             explorationinit = 5., αnstep = .05, 
                             αlambda = .1, λ = .2) 
                   for _ in 1:100]...);
figure(); plotcomparison(res3, labelsdict = labels); plt[:ylim]([0, 1]); plt[:title]("3")

@time res4 = vcat([eccompare(getmazemdp, 2, 100*10000, γ = .99, ϵ = .1,
                             αnstep = 1e-2, nsteps = 50,
                             αlambda = 5e-3, λ = .2,
                             explorationinitql = 1/200,
                             explorationinitps = 5.) 
                   for _ in 1:50]...);
figure(); plotcomparison(res4, labelsdict = labels); plt[:ylim]([0, 1]); plt[:title]("4")

@time res4b = vcat([eccompare(() -> getmazemdp(nx = 500, ny = 500, nwalls =
                                              5000), 2, 100*10^6, γ = .999, ϵ = .1,
                              explorationinitql = 1/200,
                              agents = ["ps", "psreset", "qlexplore", "ec"],
                              relperf = false) 
                   for _ in 1:3]...);
figure(); plotcomparison(res4b); plt[:title]("4b")

@time res5 = vcat([eccompare(getcliffwalkingmdp, 5, 100*750, γ = .99, ϵ = .1,
                             αnstep = 8e-2, αlambda = .05,
                             explorationinit = 1/200) 
                   for _ in 1:50]...);
figure(); plotcomparison(res5, labelsdict = labels); plt[:ylim]([0, 1.3]); plt[:title]("5")

using JLD, DataFrames
@save "$DATADIR/results.jld" res1 res2 res3 res4 res4b res5

function dftocsv(filename, df; T = 1, samples = 5)
    dfo = DataFrame(); dfo[:x] = collect(1:length(df[:value][1]))*T
    for g in groupby(df, :learner)
        dfo[Symbol(g[:learner][1])] = mean(g[:value])
        for i in 1:samples
            dfo[Symbol(g[:learner][i], "_$i")] = g[:value][i]
        end
    end
    writetable(filename, dfo)
end

Ts = [200, 200, 100, 10000, 750]
for i in 1:5
    dftocsv("$DATADIR/res$i.csv", eval(Symbol("res$i")), T = Ts[i])
end
