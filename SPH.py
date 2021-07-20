import taichi as ti
import numpy as np
from math import pi

ti.init(arch=ti.gpu)

eps = 1e-6

@ti.data_oriented
class SPHSolver:
    def __init__(self, domain_size, n_dim, dt, smooth_length, gravity=-9.8, rho_0=1000, alpha=0.1, dampping=0.9, gamma=7, cs=20):
        self.domain_size = domain_size
        self.n_dim = n_dim
        self.dt = dt
        self.gravity = gravity
        self.h = smooth_length
        self.rho_0 = rho_0
        self.alpha = alpha  # viscosity
        self.dampping = dampping
        self.kernel_norm_factor = 40 / (7 * pi * self.h ** 2) if n_dim == 2 else 8 / (pi * self.h ** 3)
        self.gamma = gamma
        self.stiffness = rho_0 * cs * cs / gamma

    def init_particles(self, offset, n_row, n_col, interval):
        """
        Allocate space
        Place regular fluid particles
        Place one-layer boundary particles in the domain boundary
        Compute boundary multiplier and mass
        """
        # count particle numbers FIXME: assuming 2d
        self.n_fluid = n_row * n_col
        self.n_side_bound = int(self.domain_size[0] / interval) + 1
        self.n_bottom_bound = int(self.domain_size[1] / interval) + 1
        self.n_total_bound = 2 * (self.n_side_bound + self.n_bottom_bound)

        # allocate particle space
        self.pos = ti.Vector.field(self.n_dim, dtype=ti.f32, shape=self.n_fluid + self.n_total_bound)
        self.vel = ti.Vector.field(self.n_dim, dtype=ti.f32, shape=self.n_fluid)
        self.density = ti.field(dtype=ti.f32, shape=self.n_fluid)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n_fluid)

        # allocate spatial grid space (for neighborhood search)
        self.max_num_neighbors = 100
        self.max_num_particles_per_cell = 500
        self.cell_size = 2 * 1.05 * self.h
        self.grid_pad = [1, 1]
        self.grid_size = [int(x / self.cell_size) + 1 + 2 * y for x, y in zip(self.domain_size, self.grid_pad)]

        grid_offs = [-self.grid_pad[0], -self.grid_pad[1]]
        self.grid_2_particles = ti.field(dtype=ti.i32, shape=(*self.grid_size, self.max_num_particles_per_cell), offset=(*grid_offs, 0))
        self.grid_num_particle = ti.field(dtype=ti.i32, shape=self.grid_size, offset=grid_offs)

        self.particle_neighbors = ti.field(dtype=ti.i32, shape=(self.n_fluid, self.max_num_neighbors))
        self.particle_num_neighbors = ti.field(dtype=ti.i32, shape=self.n_fluid)

        # fill particles
        self.init_fluid_particle(offset[0], offset[1], n_col, interval)
        self.init_boundary_particle(interval)

        # estimate mass and boundary multiplier
        n_surr = int(self.h / interval)
        w_f = 0
        w_b1 = 0
        w_b2 = 0
        d_f = np.array([0.0, 0.0])
        d_b1 = np.array([0.0, 0.0])
        for i in range(-n_surr, n_surr + 1):
            # fluid contribution
            for j in range(0, n_surr + 1):
                pos = ti.Vector([i, j]) * interval
                if pos.norm() < self.h:
                    w_f += self.cubic_kernel(-pos)
                    d_f += self.cubic_kernel_derivative(-pos).to_numpy()
            # one-layer boundary
            pos = ti.Vector([i, -1]) * interval
            if pos.norm() < self.h:
                w_b1 += self.cubic_kernel(-pos)
                d_b1 += self.cubic_kernel_derivative(-pos).to_numpy()
            # rest boundary layers
            for j in range(-n_surr, -1):
                pos = ti.Vector([i, j]) * interval
                if pos.norm() < self.h:
                    w_b2 += self.cubic_kernel(-pos)

        self.mass = self.rho_0 / (w_f + w_b1 + w_b2)
        self.gamma_1 = (w_b2 / w_b1 + 1) * 2
        self.gamma_2 = - np.dot(d_f, d_b1) / np.dot(d_b1, d_b1) * 2


    @ti.kernel
    def init_fluid_particle(self, ox: ti.f32, oy: ti.f32, n_col: ti.i32, interval: ti.f32):
        for p_i in range(self.n_fluid):
            self.pos[p_i] = ti.Vector([ox + p_i // n_col * interval, oy + p_i % n_col * interval])  # FIXME: assuming 2d
            self.vel[p_i] = ti.Vector([0.0] * self.n_dim)

    @ti.kernel
    def init_boundary_particle(self, interval: ti.f32):  # FIXME: assuming 2d
        # side boundary
        for p_i in range(self.n_side_bound):
            self.pos[self.n_fluid + p_i] = ti.Vector([eps, interval * p_i])
            self.pos[self.n_fluid + self.n_side_bound + 2 * self.n_bottom_bound + p_i] = ti.Vector([self.domain_size[0] - eps, interval * p_i])
        # bottom boundary
        for p_i in range(self.n_bottom_bound):
            self.pos[self.n_fluid + self.n_side_bound + p_i] = ti.Vector([p_i * interval, eps])
            self.pos[self.n_fluid + self.n_side_bound + self.n_bottom_bound + p_i] = ti.Vector([p_i * interval, self.domain_size[1] - eps])

    @ti.func
    def is_fluid(self, p_i):
        return p_i < self.n_fluid

    @ti.func
    def get_cell(self, pos):
        return int(pos / self.cell_size)

    @ti.func
    def is_in_grid(self, cell):
        res = True
        for i in ti.static(range(self.n_dim)):
            res = res and -self.grid_pad[i] <= cell[i] < self.grid_size[i] - self.grid_pad[i]
        return res

    @ti.pyfunc
    def cubic_kernel(self, r):
        q = r.norm() / self.h
        res = 0.0
        if q < 0.5:
            res = self.kernel_norm_factor * (6 * (q**3 - q**2) + 1)
        elif q < 1:
            res = self.kernel_norm_factor * 2 * (1 - q)**2
        return res

    @ti.pyfunc
    def cubic_kernel_derivative(self, r):
        q = r.norm() / self.h
        res = ti.Vector([0.0] * self.n_dim)
        if 1e-12 < q < 0.5:  # itself doesn't contribute derivative
            res = self.kernel_norm_factor * 6 * (3 * q**2 - 2 * q) * r.normalized()
        elif 0.5 < q < 1:
            res = -self.kernel_norm_factor * 6 * (1 - q)**2 * r.normalized()
        return res

    @ti.kernel
    def find_neighbors(self):
        """ Find neighbors by spatial grid """
        for I in ti.grouped(self.grid_num_particle):
            self.grid_num_particle[I] = 0
        for i in range(self.n_fluid):
            self.particle_num_neighbors[i] = 0

        # allocate particles to grid
        for p_i in self.pos:
            cell = self.get_cell(self.pos[p_i])
            if self.is_in_grid(cell):
                offs = ti.atomic_add(self.grid_num_particle[cell], 1)
                self.grid_2_particles[cell, offs] = p_i

        # find neighbors
        total_num_neighbor = 0
        for p_i in range(self.n_fluid):
            cell = self.get_cell(self.pos[p_i])
            nb_i = 0
            if self.is_in_grid(cell):
                for offs in ti.static(ti.grouped(ti.ndrange(*((-1, 2),) * self.n_dim))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particle[cell_to_check]):
                            p_j = self.grid_2_particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and (self.pos[p_j] - self.pos[p_i]).norm() < self.h:
                                self.particle_neighbors[p_i, nb_i] = p_j
                                nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i
            total_num_neighbor += nb_i

    @ti.kernel
    def wc_comptue(self):
        """ Compute density and pressure """
        for p_i in range(self.n_fluid):
            rho_i = 0.0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                r = self.pos[p_i] - self.pos[p_j]
                if self.is_fluid(p_j):
                    rho_i += self.cubic_kernel(r)
                else:
                    rho_i += self.gamma_1 * self.cubic_kernel(r)
            rho_i *= self.mass
            self.density[p_i] = rho_i
            self.pressure[p_i] = ti.max(self.stiffness * ((rho_i / self.rho_0) ** self.gamma - 1), 0)

    @ti.kernel
    def wc_update(self):
        """ Update positions and velocities based on pressure acceleration """
        for p_i in range(self.n_fluid):
            # compute acceleration (a = g - \grad p / \rho)
            acc = ti.Vector([0.0] * self.n_dim)
            visc = ti.Vector([0.0] * self.n_dim)
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                r = self.pos[p_i] - self.pos[p_j]
                if self.is_fluid(p_j):
                    acc -= (self.pressure[p_i] / self.density[p_i] ** 2 + self.pressure[p_j] / self.density[p_j] ** 2) \
                           * self.cubic_kernel_derivative(r)
                    visc += (self.vel[p_j] - self.vel[p_i]) * self.cubic_kernel(r) / self.density[p_j]
                else:
                    acc -= self.gamma_2 * 2 * self.pressure[p_i] / self.density[p_i] ** 2 * self.cubic_kernel_derivative(r)

            acc *= self.mass
            acc[1] += self.gravity
            visc *= self.mass

            self.vel[p_i] += self.dt * acc + self.alpha * visc
            self.pos[p_i] += self.dt * self.vel[p_i]


    def step(self):
        self.find_neighbors()
        self.wc_comptue()
        self.wc_update()
