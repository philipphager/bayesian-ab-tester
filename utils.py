from scipy.special import digamma, polygamma, roots_sh_jacobi


def log_beta_mean(alpha, beta):
    return digamma(alpha) - digamma(alpha + beta)


def var_beta_mean(alpha, beta):
    return polygamma(1, alpha) - polygamma(1, alpha + beta)


def beta_gq(n, a, b):
    x, w, m = roots_sh_jacobi(n, a + b - 1, a, True)
    w /= m
    return x, w
