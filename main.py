import taichi as ti
from SPH import SPHSolver

ti.init(arch=ti.gpu)

# scene parameters
screen_res = (800, 800)
n_dim = 2
domain_size = (1, 1)

# draw info
radius = 3
domain_2_screen = [1 / x for x in domain_size]
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2

# simulation parameters
dt = 0.001
smooth_length = 0.02
interval = 0.01
viscosity_alpha = 0.2


def render(gui, sim):
    gui.clear(bg_color)
    pos_np = sim.pos.to_numpy()
    #for i in range(n_dim):
    #    pos_np[:, i] *= domain_2_screen[i]
    gui.circles(pos_np[:sim.n_fluid], radius=radius, color=particle_color)
    gui.circles(pos_np[sim.n_fluid + 1:], radius=radius, color=boundary_color)
    gui.show()


def test_SPH():
    sim = SPHSolver(domain_size, n_dim, dt, smooth_length, alpha=viscosity_alpha, damping=0.999)
    sim.init_particles([0.25, 0.3], 50, 50, interval)
    gui = ti.GUI("WCSPH demo", screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        render(gui, sim)
        sim.step()


if __name__ == "__main__":
    test_SPH()
