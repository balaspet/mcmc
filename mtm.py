from numpy.random import choice
from numpy.random import rand


def draw_sample(t, l, n, k, d, xt):
    candidates = list(set(min(abs(xt + i - 1) + 1, n) for i in range(-d, d + 1)))
    c = len(candidates)

    y = []
    y_w = []
    for i in range(0, k):
        y.append(choice(candidates))
        y_w.append(t.pmf(y[i]) * l(y[i], xt) / c)
    yn = choice(y, p = [w / sum(y_w) for w in y_w])

    candidates = list(set(min(abs(yn + i - 1) + 1, n) for i in range(-d, d + 1)))
    c = len(candidates)

    x = []
    x_w = []
    for i in range(0, k - 1):
        x.append(choice(candidates))
        x_w.append(t.pmf(x[i]) * l(x[i], yn) / c)
    x.append(xt)
    x_w.append(t.pmf(xt) * l(xt, yn) / c)

    a = (sum(x_w) / sum(y_w))
    if (rand() < a):
        return yn
    else:
        return xt

def multiple_try_metropolis(target_dist, l, n, k, d, xt, sample_count, burn_in, step):
    samples = []
    
    # burn in
    for i in range(0, burn_in):
        xt = draw_sample(target_dist, l, n, k, d, xt)
    
    for i in range(0, sample_count * step):
        xt = draw_sample(target_dist, l, n, k, d, xt)
        if (i % step):
            samples.append(xt)
            
    return samples