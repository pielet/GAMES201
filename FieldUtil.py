import taichi as ti

@ti.kernel
def scale(x: ti.template(), a: ti.f32):
    for I in ti.grouped(x):
        x[I] *= a

@ti.kernel
def axpy(a: ti.f32, x: ti.template(), y: ti.template()):
    for I in ti.grouped(y):
        y[I] += a * x[I]

@ti.kernel
def reduce(res: ti.template(), x: ti.template(), y: ti.template()):
    res[None] = 0.0
    for I in ti.grouped(x):
        res[None] += x[I].dot(y[I])

@ti.kernel
def element_wist_mul(x: ti.template(), y: ti.template(), z: ti.template()):
    for I in ti.grouped(z):
        z[I] = x[I] * y[I]

@ti.kernel
def print_field(x: ti.template()):
    for I in ti.grouped(x):
        print(I, x[I], end="\n")

@ti.func
def safe_normalized(vec):
    return vec / max(vec.norm(), 1e-12)

@ti.func
def lerp(x0, x1, a):
    return (1 - a) * x0 + a * x1

@ti.func
def bilerp(x0, x1, x2, x3, a, b):
    return (1 - a) * (1 - b) * x0 + a * (1 - b) * x1 + (1 - a) * b * x2 * a * b * x3

@ti.func
def lerp(dim, var, I, a):
    """
    N-d interpolation
    :param dim: dimension
    :param var: variable field
    :param I: origin index [ti.Vector]
    :param a: alpha [ti.Vector]
    :return: interpolated value
    """
    res = 0.0
    for II in ti.static(ti.ndrange(*(2,) * dim)):
        weight = 1.0
        for d in ti.static(dim):
            weight *= II[d] * a[d] + (1 - II[d]) * (1 - a[d])
        res += weight * var[I + II]
    return res
