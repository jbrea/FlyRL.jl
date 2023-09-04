using FiniteDiff

function lr_scan(l0, model, input, target, shock, params, dp)
    for η in 2. .^ (-60:0)
        l1 = logprob(model, input, target, shock, params + η * dp)
        if l1 < l0
#             @show η/2
            return η/2
        end
        η *= 2
    end
    0. # no fine tuning
end
function gradient_descent_fine_tuning(model, track, params0; T = 10^3)
    params = copy(params0)
    input, target, shock = preprocess(model.preprocessor, track)
    l0, dp0 = FlyRL.grad_logprob(model, input, target, shock, params)
    l00 = l0
    η = lr_scan(l0, model, input, target, shock, params, dp0)
    if η > 0
        for _ in 1:T
            params .+= η * dp0
            l1, dp1 = FlyRL.grad_logprob(model, input, target, shock, params)
            if l1 < l0
                params .-= η * dp0
                η = lr_scan(l0, model, input, target, shock, params, dp0)
                η == 0 && break
            else
                dp0 = dp1
                l0 = l1
            end
        end
    end
    (; params, l0, dl = l0 - l00, dparams = params - params0,
       maxdparams = maximum(abs, params - params0),
       dp = dp0, maxdp = maximum(abs, dp0))
end
function laplace_approx(model, track, θ)
    H = FiniteDiff.finite_difference_hessian(θ -> logprob(model, track, θ), θ)
    H[2, 2] ≈ 0 && return Inf
    -1/H[2, 2]
end

