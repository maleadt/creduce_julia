function getEndpoints(distr::UnivariateDistribution, epsilon::Real)
    (left,right) = map(x -> quantile(distr,x), (0,1))
    leftEnd = left!=-Inf ? left : quantile(distr, epsilon)
    rightEnd = right!=-Inf ? right : quantile(distr, 1-epsilon)
    (leftEnd, rightEnd)
end
function expectation(distr::ContinuousUnivariateDistribution, g::Function, epsilon::Real)
    f = x->pdf(distr,x)
    (leftEnd, rightEnd) = getEndpoints(distr, epsilon)
    integrate(x -> f(x)*g(x), leftEnd, rightEnd)
end
function expectation(distr::DiscreteUnivariateDistribution, g::Function, epsilon::Real)
    f = x->pdf(distr,x)
    (leftEnd, rightEnd) = getEndpoints(distr, epsilon)
    sum(x -> f(x)*g(x), leftEnd:rightEnd)
end
function expectation(distr::UnivariateDistribution, g::Function)
    expectation(distr, g, 1e-10)
end
function kldivergence(P::UnivariateDistribution, Q::UnivariateDistribution)
    expectation(P, x -> log(pdf(P,x)/pdf(Q,x)))
end
