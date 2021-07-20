import taichi as ti
ti.init(arch="gpu")

@ti.pyfunc
def ti_func(x):
    return x * x

def main():
    x = ti_func(2)
    v = ti.Vector.field(2, dtype=ti.f32, shape=())
    v[None][0] = 0

if __name__ == "__main__":
    main()