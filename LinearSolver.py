import taichi as ti
from FieldUtil import *
from math import log2


@ti.data_oriented
class LinearSolver:
    CG_PRECOND_METHED = {
        "None": 0,
        "Jacobi": 1,
        "multigrid": 2
    }

    SMOOTHING_METHOD = {
        "weighted Jacobi": 0,
        "red-black GS": 1
    }

    def __init__(self, dim, N, cg_precond="multigrid", smooth="red-black GS",
                 cg_iter=100, cg_err=1e-6,
                 n_mg_level=6, prev_smooth=3, post_smooth=3):
        self.dim = dim
        self.N = N

        self.cg_precond = self.CG_PRECOND_METHED[cg_precond]
        self.cg_iter = cg_iter
        self.cg_err = cg_err

        self.smooth_method = self.SMOOTHING_METHOD[smooth]
        self.prev_smooth = prev_smooth
        self.post_smooth = post_smooth

        # CG iteration variables
        self.alpha = ti.field(ti.f32, shape=())
        self.beta = ti.field(ti.f32, shape=())
        self.rTz = ti.field(ti.f32, shape=())
        self.sum = ti.field(ti.f32, shape=())
        self.residual = ti.field(ti.f32, shape=())

        self.b = ti.field(ti.f32)
        self.p = ti.field(ti.f32)
        self.Ap = ti.field(ti.f32)
        
        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.r = [ti.field(ti.f32)]
            self.z = [ti.field(ti.f32)]
            self.A_mask = [ti.field(ti.f32)]
            self.inv_A_diag = ti.field(ti.f32)

            ti.root.dense(ti.indices(self.dim), self.N).place(self.A_mask[0])
            ti.root.dense(ti.indices(self.dim), self.N + 2).place(
                self.b, self.r[0], self.z[0], self.Ap, self.p, self.inv_A_diag, offsert=(-1,) * self.dim)
        elif self.cg_precond == self.CG_PRECOND_METHED["multigrid"]:
            self.mg_level = min(n_mg_level, int(log2(self.N)) - 2)
            assert self.mg_level > 0  # minimum grid size = 8

            self.A_mask = [ti.field(ti.f32) for _ in range(self.mg_level)]
            self.r = [ti.field(ti.f32) for _ in range(self.mg_level)]
            self.z = [ti.field(ti.f32) for _ in range(self.mg_level)]
            self.res = [ti.field(ti.f32) for _ in range(self.mg_level)]
            
            indices = ti.ij if self.dim == 2 else ti.ijk
            ti.root.dense(indices, self.N + 2).place(self.b, self.Ap, self.p, offset=(-1,) * self.dim)  # padding p for computing Ap
            for i in range(self.mg_level):
                # padding z for computing Az
                # padding res for restriction
                ti.root.dense(indices, self.N // 2 ** i + 2).place(self.z[i], self.r[i], self.res[i], offset=(-1,) * self.dim)
                ti.root.dense(indices, self.N // 2 ** i).place(self.A_mask[i])

    def reset(self):
        self.b.fill(0.0)
        self.p.fill(0.0)
        self.Ap.fill(0.0)
        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.inv_A_diag.fill(0.0)
        for i in range(len(self.A_mask)):
            self.r[i].fill(0.0)
            self.z[i].fill(0.0)
            self.A_mask[i].fill(0.0)
            if self.cg_precond == self.CG_PRECOND_METHED["multigrid"]:
                self.res[i].fill(0.0)

    @ti.func
    def neighbor_sum(self, x, I):
        res = 0.0
        for i in ti.static(range(self.dim)):
            offs = self.Vector.unit(self.dim, i, dt=ti.i32)
            res += x[I + offs] + x[I - offs]
        return res

    @ti.func
    def is_boundary(self, i, I):
        return I.min() < 0 or I.max() >= self.N or self.A_mask[i][I] == 0
        
    @ti.kernel
    def update_preconditioner(self):
        for I in ti.grouped(self.A_mask[0]):
            if self.A_mask[0][I] > 0:
                self.A_inv_diag[I] = 1.0 / self.A_mask[0][I]

    @ti.kernel
    def construct_level_A_mask(self, i: ti.i32):
        # mark fluid grid
        for I in ti.grouped(self.A_mask[i]):
            for II in ti.static(ti.grouped(ti.ndrange(*(2,) * self.dim))):
                if self.A_mask[i - 1][2 * I + II] > 0:
                    self.A_mask[i][I] = 2 ** self.dim

        # count Neumann boundary
        for I in ti.grouped(self.A_mask[i]):
            if self.A_mask[i][I] > 0:
                for i in ti.static(range(self.dim)):
                    offs = ti.Vector.unit(self.dim, i, dt=ti.i32)
                    if self.is_boundary(i, I + offs):
                        self.A_mask[i][I] -= 1
                    if self.is_boundary(i, I - offs):
                        self.A_mask[i][I] -= 1

    @ti.kernel
    def compute_Ap(self, x: ti.template()):
        for I in ti.grouped(self.A_mask[0]):
            if self.A_mask[0][I] > 0:
                self.Ap[I] = self.A_mask[0][I] * x[I] - self.neighbor_sum(x, I)

    @ti.kernel
    def Cholesky(self):
        pass

    def multigrid_solver(self):
        """
        Solve Az = r \n
        smoothing: weighted Jacobi (\omega = 2/3) or red-black GS \n
        finest level solver: Cholesky \n
        **NOTE:** assuming level A_mask has been constructed \n
        """
        omega = 2 / 3

        @ti.kernel
        def compute_residual(A_mask: ti.template(), z: ti.template(), r: ti.template(), res: ti.template()):
            for I in ti.grouped(A_mask):
                if A_mask[I] > 0:
                    res[I] = r[I] - (A_mask[I] * z[I]) + self.neighbor_sum(z, I)

        @ti.kernel
        def weighted_Jacobi_smooth(A_mask: ti.template(), z: ti.template(), r: ti.template()):
            for I in ti.grouped(A_mask):
                if A_mask[I] > 0:
                    rhs = r[I] + self.neighbor_sum(z, I)
                    z[I] = (1 - omega) * z[I] + omega * rhs / A_mask[I]

        @ti.kernel
        def red_black_GS_smooth(A_mask: ti.template(), z: ti.template(), r: ti.template(), order: ti.i32):
            for I in ti.grouped(A_mask):
                if A_mask[I] > 0 and I.sum() % 2 == order:
                    rhs = r[I] + self.neighbor_sum(z, I)
                    z[I] = rhs / A_mask[I]
            for I in ti.grouped(A_mask):
                if A_mask[I] > 0 and I.sum() % 2 == 1 - order:
                    rhs = r[I] + self.neighbor_sum(z, I)
                    z[I] = rhs / A_mask[I]

        @ti.func
        def get_weight(offset):
            if offset == 0 or offset == 1:
                return 3 / 8
            elif offset == -1 or offset == 2:
                return 1 / 8
            else:
                return 0.0

        @ti.kernel
        def restriction(A_mask: ti.template(), r_in: ti.template(), r_out: ti.template()):
            for I in ti.grouped(A_mask):
                if A_mask[I] > 0:
                    for offs in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * self.dim))):
                        weight = 1.0
                        for d in ti.static(self.dim):
                            weight *= get_weight(offs[d])
                        r_out[I] += weight * r_in(2 * I + offs)

        @ti.kernel
        def prolongation(A_mask: ti.template(), z_in: ti.template(), z_out: ti.template()):
            for I in ti.grouped(A_mask):
                if A_mask[I] > 0:
                    II = (I - 1) // 2
                    alpha = ti.Vector([0.0] * self.dim)
                    for d in ti.static(self.dim):
                        alpha[d] = 3/4 if I[d] % 2 == 0 else 1/4
                    z_out[I] += lerp(self.dim, z_in, II, alpha)

        for i in range(self.mg_level - 1):
            for j in range(self.prev_smooth):
                if self.smooth_method == self.SMOOTHING_METHOD["weighted Jacobi"]:
                    weighted_Jacobi_smooth(self.A_mask[i], self.z[i], self.r[i])
                elif self.smooth_method == self.SMOOTHING_METHOD["red-black GS"]:
                    red_black_GS_smooth(self.A_mask[i], self.z[i], self.r[i], 0)

            compute_residual(self.A_mask[i], self.z[i], self.r[i], self.res[i])
            restriction(self.A_mask[i + 1], self.r[i], self.r[i + 1])

        # direct solver

        for i in reversed(range(self.mg_level - 1)):
            prolongation(self.A_mask[i], self.z[i + 1], self.z[i])

            for j in range(self.post_smooth):
                if self.smooth_method == self.SMOOTHING_METHOD["weighted Jacobi"]:
                    weighted_Jacobi_smooth(self.A_mask[i], self.z[i], self.r[i])
                elif self.smooth_method == self.SMOOTHING_METHOD["red-black GS"]:
                    red_black_GS_smooth(self.A_mask[i], self.z[i], self.r[i], 1)  # reversed order

    def conjugate_gradient(self, A_mask, x):
        """
        Solve Ax = b \n
        [A_mask: ti.field(int)] stores Neumann boundary count \n
        [x: ti.field(float)] is both input (initial guess) and output \n
        **NOTE:** and self.b must be set before calling this method
        """
        self.reset()

        # r = b - Ax (x's initial value is lambda from last epoch)
        self.A_mask[0].copy_from(A_mask)
        self.r[0].copy_from(self.b)
        self.Ap.fill(0.0)
        self.compute_Ap(x)
        axpy(-1.0, self.Ap, self.r[0])

        reduce(self.sum, self.b, self.b)
        threshold = min(self.sum[None] * self.cg_err, self.cg_err)  # |b| scaled threshold

        # z and p
        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.update_preconditioner()
            element_wist_mul(self.inv_A_diag, self.r[0], self.z[0])
        elif self.cg_precond == self.CG_PRECOND_METHED["multigrid"]:
            for i in range(1, self.mg_level):
                self.construct_level_A_mask(i)
            self.multigrid_solver()
        elif self.cg_precond == self.CG_PRECOND_METHED["None"]:
            self.z[0].copy_from(self.r[0])
        self.p.copy_from(self.z[0])

        # rTz
        reduce(self.rTz, self.r[0], self.z[0])
        # print("CG iter -1: %.1ef" % self.rTz[None])

        reduce(self.residual, self.r[0], self.r[0])
        if self.residual[None] < threshold:
            return 0, self.residual[None]

        n_iter = 0
        for i in range(self.cg_iter):
            n_iter += 1
            self.Ap.fill(0.0)
            self.compute_Ap(self.p)

            # alpha
            reduce(self.sum, self.p, self.Ap)
            self.alpha[None] = self.rTz[None] / self.sum[None]

            # update x and r(z)
            axpy(self.alpha[None], self.p, x)
            axpy(-self.alpha[None], self.Ap, self.r[0])

            reduce(self.residual, self.r[0], self.r[0])
            if self.residual[None] < threshold:
                break

            if self.cg_precond == self.CG_PRECOND_METHED["None"]:
                self.z[0].copy_from(self.r[0])
            elif self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
                element_wist_mul(self.inv_A_diag, self.r[0], self.z[0])
            elif self.cg_precond == self.CG_PRECOND_METHED["multigrid"]:
                self.multigrid_solver()

            # beta
            reduce(self.sum, self.r[0], self.z[0])
            self.beta[None] = self.sum[None] / self.rTz[None]
            self.rTz[None] = self.sum[None]

            scale(self.p, self.beta[None])
            axpy(1.0, self.z[0], self.p)

        return n_iter, self.residual[None]
