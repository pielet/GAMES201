import taichi as ti
from FieldUtil import *
from LinearSolver import LinearSolver

ti.init(debug=True)

@ti.data_oriented
class Euler:
    def __init__(self, dim, N, dt):
        self.dt = dt
        self.dim = dim
        self.N = N
        self.grid_size = 1 / N

        # simulation variables
        self.u = ti.field(ti.f32)
        self.v = ti.field(ti.f32)
        self.u_tmp = ti.field(ti.f32)  # for advection
        self.v_tmp = ti.field(ti.f32)

        self.p = ti.field(ti.f32)
        self.rho = ti.field(ti.f32)
        self.rho_tmp = ti.field(ti.f32)

        self.fluid_mask = ti.field(ti.int32)

        self.rho_boundary_mask = ti.field(ti.i32)  # for padding
        self.rho_boundary_mask_tmp = ti.field(ti.i32)

        self.output_field = ti.field(ti.f32)

        self.padding = 5
        if dim == 2:
            ti.root.dense(ti.ij, (N + 1, N)).place(self.u, self.u_tmp)
            ti.root.dense(ti.ij, (N, N + 1)).place(self.v, self.v_tmp)

            # padding rho to advection
            ti.root.dense(ti.ij, N + 2 * self.padding).place(
                self.rho, self.rho_tmp, self.rho_boundary_mask, self.rho_boundary_mask_tmp,
                offset=(-self.padding,) * self.dim)

            # padding p for MGPCG solver
            ti.root.dense(ti.ij, N + 2).place(self.p, offset=(-1,) * self.dim)

            # fluid mask defines the Neumann boundary
            ti.root.dense(ti.ij, N).place(self.fluid_mask)

            # used for visualization
            ti.root.dense(ti.ij, N).place(self.output_field)

        # solver
        self.solver = LinearSolver(dim, N)

        # scene
        self.external_func = []

        self.source_dim = []
        self.source_velocity = []
        self.source_position = []
        self.source_radius = []
        self.source_density = 1.0

        self.unit = []
        for d in range(self.dim):
            unit = [0] * self.dim
            unit[d] = 1
            self.unit.append(ti.Vector(unit, dt=ti.i32))

    def init(self):
        """ init() must be called after adding sphere and source """
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.p.fill(0.0)
        self.rho.fill(0.0)

        self.__update_fluid_mask()
        self.__enforce_boundary_velocity()

    def add_source(self, dim, pos, radius, vel):
        """ pos is a list[dim] at bottom, radius is scalar """
        self.source_dim.append(dim)
        pos.insert(dim, 0)
        self.source_position.append(pos)
        self.source_radius.append(radius)
        self.source_velocity.append(vel)

    def add_sphere(self, origin, radius):
        """ origin is list[dim], radius is scalar """
        @ti.kernel
        def update_mask():
            r = int(radius)
            for I in ti.grouped(ti.ndrange(*((-r, r + 1),) * self.dim)):
                if I.norm() < radius:
                    self.fluid_mask[origin + I] = 0
        self.external_func.append(update_mask)

    @ti.kernel
    def __get_field(self, f: ti.template()):
        for I in ti.grouped(self.output_field):
            self.output_field[I] = f[I]

    def get_density_field(self):
        self.__get_field(self.rho)
        return self.output_field

    def __update_fluid_mask(self):
        @ti.kernel
        def update_boundary():
            for I in ti.grouped(self.fluid_mask):
                if self.is_fluid(I):
                    for i in ti.static(range(self.dim)):
                        if not self.is_fluid(I + self.unit[i]):
                            self.fluid_mask[I] -= 1
                        if not self.is_fluid(I - self.unit[i]):
                            self.fluid_mask[I] -= 1

        self.fluid_mask.fill(2 ** self.dim)
        for func in self.external_func:
            func()
        update_boundary()

    def __enforce_boundary_velocity(self):
        @ti.kernel
        def enforce_boundary_velocity_1d(vel: ti.template(), dim: ti.i32):
            for I in ti.grouped(vel):
                if not self.is_interior(dim, I):
                    vel[I] = self.get_boundary_velocity(dim, I)

        enforce_boundary_velocity_1d(self.u, 0)
        enforce_boundary_velocity_1d(self.v, 1)

    @ti.func
    def is_fluid(self, I):
        valid_idx = I.min() >= 0 and I.max() < self.N
        return valid_idx and self.fluid_mask[I]
        # return I.min() >= 0 and I.max() < self.N and self.fluid_mask[I] > 0  # TODO: why this is wrong ???

    @ti.func
    def is_interior(self, dim, I):
        dim = ti.static(dim)
        print(dim)
        return self.is_fluid(I) and self.is_fluid(I - self.unit[dim]) and self.is_fluid(I + self.unit[dim])

    @ti.func
    def get_boundary_velocity(self, dim, I):
        vel = 0.0
        if I[dim] <= 0:
            proj_p = I.copy()
            proj_p[dim] = 0
            for i in ti.static(range(len(self.source_position))):
                if dim == self.source_dim[i] and (proj_p - self.source_position[i]).norm() < self.source_radius[i]:
                    vel = self.source_velocity[i]
                    break
        return vel

    def extrapolate_density(self, rho, buffer):
        @ti.kernel
        def copy_density_mask(mask: ti.template()):
            for I in ti.grouped(self.fluid_mask):
                mask[I] = self.fluid_mask[I]

        @ti.kernel
        def extrapolate(src: ti.template(), dst: ti.template(), mask_src: ti.template(), mask_dst: ti.template()):
            for I in ti.grouped(dst):
                if mask_src[I] == 0:
                    count = 0
                    for II in ti.static(ti.grouped(ti.ndrange(*((-1, 2),) * self.dim))):
                        if mask_src[I + II] > 0:
                            count += 1
                            dst[I] += src[I + II]
                    if count > 0:
                        dst[I] /= count
                        mask_dst[I] = 1

        self.rho_boundary_mask.fill(0)
        copy_density_mask(self.rho_boundary_mask)
        for i in range(self.padding):
            buffer.copy_from(rho)
            self.rho_boundary_mask_tmp.copy_from(self.rho_boundary_mask)
            extrapolate(buffer, rho, self.rho_boundary_mask_tmp, self.rho_boundary_mask)

    def semi_Lagragian_velocity(self):
        @ti.kernel
        def semi_Larangian_1d(src: ti.template(), dst: ti.template(), dim: ti.i32):
            for I in ti.grouped(dst):
                if self.is_interior(dim, I):  # only advect interior velocity
                    prev_p = I[dim] - src[I] * self.dt  # FIXME: RK1
                    I0 = I.copy()
                    I1 = I.copy()
                    I0[dim] = int(prev_p // 1)  # downward for negative number
                    I1[dim] = int(prev_p // 1) + 1
                    x0 = src[I0] if self.is_interior(dim, I0) else self.get_boundary_velocity(dim, I0)
                    x1 = src[I1] if self.is_interior(dim, I0) else self.get_boundary_velocity(dim, I1)
                    dst[I] = lerp(x0, x1, prev_p - I0[dim])

        # velocity
        semi_Larangian_1d(self.u, self.u_tmp, 0)
        semi_Larangian_1d(self.v, self.v_tmp, 1)

        self.u.copy_from(self.u_tmp)
        self.v.copy_from(self.v_tmp)

    def semi_Lagrangian_density(self, src, dst):
        @ti.func
        def compute_offset(I, v, origin: ti.template(), alpha: ti.template(), dim):
            prev_p = I[dim] - (v[I] + v[I + self.unit[dim]]) / 2 * self.dt  # FIXME: RK1
            origin[dim] = int(prev_p // 1)
            alpha[dim] = prev_p - origin[dim]

        @ti.kernel
        def semi_Lagragian(src: ti.template(), dst: ti.template()):
            for I in ti.grouped(self.fluid_mask):  # only advect interior density
                orig = ti.Vector([0] * self.dim)
                alpha = ti.Vector([0.0] * self.dim)

                compute_offset(I, self.u, orig, alpha, 0)  # FIXME: assume 2d
                compute_offset(I, self.v, orig, alpha, 1)

                dst[I] = lerp(2, src, orig, alpha)

        semi_Lagragian(src, dst)

    def compute_b(self):
        @ti.func
        def finite_diff(var, I, dim):
            return var[I + self.unit[dim]] - var[I]

        @ti.kernel
        def compute_b(b: ti.template()):
            for I in ti.grouped(self.fluid_mask):
                b[I] = - self.rho_tmp[I] / self.dt * (finite_diff(self.u, I, 0) + finite_diff(self.v, I, 1))

        self.solver.b.fill(0.0)
        compute_b(self.solver.b)

    def update_velocity(self):
        @ti.kernel
        def update_velocity_1d(vel: ti.template(), p: ti.template(), dim: ti.i32):
            for I in ti.grouped(vel):
                if self.is_interior(dim, I):  # only update interior
                    unit = ti.static(self.unit[dim])
                    vel[I] += -2 * self.dt / (self.rho_tmp[I - unit] + self.rho_tmp[I]) * (p[I] - p[I - unit])  # FIXME: use which rho?

        update_velocity_1d(self.u, self.p, 0)
        update_velocity_1d(self.v, self.p, 1)

    def step(self):
        # update scene (reset fluid mask)
        # self.update_fluid_mask()

        # advection
        self.semi_Lagragian_velocity()
        self.extrapolate_density(self.rho, self.rho_tmp)
        self.semi_Lagrangian_density(self.rho, self.rho_tmp)

        # projection
        self.compute_b()
        self.solver.conjugate_gradient(self.fluid_mask, self.p)

        # update velocity and density
        self.update_velocity()
        self.rho_tmp.copy_from(self.rho)
        self.semi_Lagrangian_density(self.rho_tmp, self.rho)


def test():
    bg_color = 0x112f41
    N = 512

    sim = Euler(2, N, 0.001)

    sim.add_source(1, [200], 2, 10)
    sim.add_sphere([200, 200], 10)

    sim.init()

    gui = ti.GUI("Euler gas", (N, N))
    while gui.running and not gui.get_event(gui.ESCAPE):
        gui.clear(bg_color)
        gui.set_image(sim.get_density_field())
        sim.step()


if __name__ == "__main__":
    test()
