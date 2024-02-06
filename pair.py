from analytics import eq0_V_lambdified, eq0_Fu_lambdified, eq0_Fv_lambdified, eq0_C_lambdified, Eu_lambdified, Ev_lambdified, B_lambdified
from derivatives import d
from coords import BP2cart, cart2BP
import numpy as np
from matplotlib import pyplot as plt
from time import time
from scipy.optimize import fsolve
from tqdm import tqdm

class Pair: 
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'filename':
                data = np.load(value)
                self.n = int(data[0])
                self.k = float(data[1])
                self.a = float(data[2])
                self.nu = int(data[3])
                self.nv = int(data[4])
                self.max_v = float(data[5])
                self.ier = int(data[6])
                self.solution = data[7:]
            else:
                self.__dict__[key] = value
        self.j = -4/self.k**4
        self.us = np.linspace(0, 2*np.pi, self.nu + 1)[:-1]
        self.vs = np.linspace(0, self.max_v, self.nv + 1)[1:]
        self.du = self.us[1] - self.us[0]
        self.dv = self.vs[1] - self.vs[0]
    
    def save(self, filename):
        nunv = self.nu*self.nv
        data = np.zeros(7 + 4*nunv)
        data[0:7] = [self.n, self.k, self.a, self.nu, self.nv, self.max_v, self.ier]
        data[7:] = self.solution
        np.save(filename, data)

    def dfunc_du(self, func, N = 1):
        result = np.zeros(func.shape)
        for j in range(func.shape[1]):
            boundary_left = func[-1,j]
            boundary_right = func[0,j]
            result[:,j] = d(func[:,j], self.du, N, boundaries = [boundary_left, boundary_right])
        return result
    
    def dV_du(self, V, N = 1):
        return self.dfunc_du(V, N)

    def dFu_du(self, Fu, N = 1):
        return self.dfunc_du(Fu, N)

    def dFv_du(self, Fv, N = 1):
        return self.dfunc_du(Fv, N)

    def dF_du(self, Fu, Fv, N = 1):
        return self.dfunc_du(Fu, N), self.dfunc_du(Fv, N)
        
    def dC_du(self, C, N = 1):
        return self.dfunc_du(C, N)

    def dV_dv(self, V, N = 1):
        result = np.zeros(V.shape)
        for i in range(V.shape[0]):
            if i == 0:
                boundary_left = -1
            else:
                boundary_left = V[i,0]
            boundary_right = np.mean(V[:,-1])
            result[i,:] = d(V[i,:], self.dv, N, boundaries = [boundary_left, boundary_right])
        return result

    def dF_dv(self, Fu, Fv, N = 1):
        Fu_result = np.zeros(Fu.shape)
        Fv_result = np.zeros(Fv.shape)
        xs, ys = [], []
        for i, u in enumerate(self.us):
            x, y = BP2cart(Fu[i, -1], Fv[i, -1], u, self.vs[-1])
            xs.append(x)
            ys.append(y)
        avg_x = np.mean(xs)
        avg_y = np.mean(ys)
        for i, u in enumerate(self.us):
            if i == 0:
                boundary_left_u = self.n / self.a
            else:
                boundary_left_u = Fu[i,0]
            boundary_left_v = 0
            boundary_right_u, boundary_right_v = cart2BP(avg_x, avg_y, u, self.vs[-1] + self.dv) # self.vs[-1] + self.dv ~ infinity
            Fu_result[i,:] = d(Fu[i,:], self.dv, N, boundaries = [boundary_left_u,boundary_right_u])
            Fv_result[i,:] = d(Fv[i,:], self.dv, N, boundaries = [boundary_left_v, boundary_right_v])
        return Fu_result, Fv_result

    def dC_dv(self, C, N = 1):
        result = np.zeros(C.shape)
        for i in range(C.shape[0]):
            if i == 0:
                boundary_left = np.sqrt(-self.j)
            else:
                boundary_left = C[i,0]
            boundary_right = 0
            result[i,:] = d(C[i,:], self.dv, N, boundaries = [boundary_left, boundary_right])
        return result

    def split(self, sol):
        nunv = self.nu*self.nv
        V = np.reshape(sol[0:nunv], (self.nu, self.nv))
        Fu = np.reshape(sol[nunv:2*nunv], (self.nu, self.nv))
        Fv = np.reshape(sol[2*nunv:3*nunv], (self.nu, self.nv))
        C = np.reshape(sol[3*nunv:], (self.nu, self.nv))
        return V, Fu, Fv, C
    
    def split_derivatives(self, sol):
        V, Fu, Fv, C = self.split(sol)
        V_u = self.dV_du(V, N = 1)
        V_v = self.dV_dv(V, N = 1)
        V_uu = self.dV_du(V, N = 2)
        V_uv = self.dV_du(V_v, N = 1)
        V_vv = self.dV_dv(V, N = 2)
        Fu_u, Fv_u = self.dF_du(Fu, Fv, N = 1)
        Fu_v, Fv_v = self.dF_dv(Fu, Fv, N = 1)
        Fu_uu, Fv_uu = self.dF_du(Fu, Fv, N = 2)
        Fu_uv, Fv_uv = self.dF_du(Fu_v, Fv_v, N = 1)
        Fu_vv, Fv_vv = self.dF_dv(Fu, Fv, N = 2)
        C_u = self.dC_du(C, N = 1)
        C_v = self.dC_dv(C, N = 1)
        C_uu = self.dC_du(C, N = 2)
        C_uv = self.dC_du(C_v, N = 1)
        C_vv = self.dC_dv(C, N = 2)
        V_derivatives = [V, V_u, V_v, V_uu, V_uv, V_vv]
        Fu_derivatives = [Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv]
        Fv_derivatives = [Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv]
        C_derivatives = [C, C_u, C_v, C_uu, C_uv, C_vv]
        return V_derivatives, Fu_derivatives, Fv_derivatives, C_derivatives
    
    def solve(self):
        def f(V_Fu_Fv_C):
            V_derivatives, Fu_derivatives, Fv_derivatives, C_derivatives = self.split_derivatives(V_Fu_Fv_C)
            V, V_u, V_v, V_uu, V_uv, V_vv = V_derivatives
            Fu, Fu_u, Fu_v, Fu_uu, Fu_uv, Fu_vv = Fu_derivatives
            Fv, Fv_u, Fv_v, Fv_uu, Fv_uv, Fv_vv = Fv_derivatives
            C, C_u, C_v, C_uu, C_uv, C_vv = C_derivatives
            eq0_V = np.zeros((self.nu, self.nv))
            eq0_Fu = np.zeros((self.nu, self.nv))
            eq0_Fv = np.zeros((self.nu, self.nv))
            eq0_C = np.zeros((self.nu, self.nv))
            # define arguments for the lambdified expression from sympy and fill the result array
            args = np.zeros(29)
            for i, u in enumerate(self.us):
                for j, v in enumerate(self.vs):
                    args = [V[i,j], V_u[i,j], V_v[i,j], V_uu[i,j], V_uv[i,j], V_vv[i,j]]
                    args.extend([Fu[i,j], Fu_u[i,j], Fu_v[i,j], Fu_uu[i,j], Fu_uv[i,j], Fu_vv[i,j]])
                    args.extend([Fv[i,j], Fv_u[i,j], Fv_v[i,j], Fv_uu[i,j], Fv_uv[i,j], Fv_vv[i,j]])
                    args.extend([C[i,j], C_u[i,j], C_v[i,j], C_uu[i,j], C_uv[i,j], C_vv[i,j]])
                    args.extend([u, v, self.a, self.j, self.n])
                    eq0_V[i,j] = eq0_V_lambdified(*args)
                    eq0_Fu[i,j] = eq0_Fu_lambdified(*args)
                    eq0_Fv[i,j] = eq0_Fv_lambdified(*args)
                    eq0_C[i,j] = eq0_C_lambdified(*args)
            return np.concatenate([eq0_V.flatten(), eq0_Fu.flatten(), eq0_Fv.flatten(), eq0_C.flatten()])

        # solve nonlinear problem
        nunv = self.nu*self.nv
        x0 = np.zeros(4*nunv)
        x0[0:nunv] = -1
        x0[3*nunv:] = np.sqrt(-self.j)
        start = time()
        self.solution, self.infodict, self.ier, self.mesg = fsolve(f, x0, full_output = True)
        end = time()
        self.time = end-start

    def observables(self):
        V_derivatives, Fu_derivatives, Fv_derivatives, C_derivatives = self.split_derivatives(self.solution)
        self.V, self.V_u, self.V_v, self.V_uu, self.V_uv, self.V_vv = V_derivatives
        self.Fu, self.Fu_u, self.Fu_v, self.Fu_uu, self.Fu_uv, self.Fu_vv = Fu_derivatives
        self.Fv, self.Fv_u, self.Fv_v, self.Fv_uu, self.Fv_uv, self.Fv_vv = Fv_derivatives
        self.C, self.C_u, self.C_v, self.C_uu, self.C_uv, self.C_vv = C_derivatives
        self.Ju = np.zeros((self.nu, self.nv))
        self.Jv = np.zeros((self.nu, self.nv))
        self.rho = np.zeros((self.nu, self.nv))
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                self.Ju[i,j] = -(self.Fu[i,j] - self.n * np.cosh(v)/self.a) * self.C[i,j]**2
                self.Jv[i,j] = -(self.Fv[i,j]) * self.C[i,j]**2
                self.rho[i,j] = - self.C[i,j]**2 * self.V[i,j]
        self.Eu = np.zeros((self.nu, self.nv))
        self.Ev = np.zeros((self.nu, self.nv))
        self.B = np.zeros((self.nu, self.nv))
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                E_args = [self.V[i,j], self.V_u[i,j], self.V_v[i,j], self.V_uu[i,j], self.V_uv[i,j], self.V_vv[i,j], u, v, self.a, self.n]
                self.Eu[i,j], self.Ev[i,j] = Eu_lambdified(*E_args), Ev_lambdified(*E_args)
                B_args = [self.Fu[i,j], self.Fu_u[i,j], self.Fu_v[i,j], self.Fu_uu[i,j], self.Fu_uv[i,j], self.Fu_vv[i,j]]
                B_args.extend([self.Fv[i,j], self.Fv_u[i,j], self.Fv_v[i,j], self.Fv_uu[i,j], self.Fv_uv[i,j], self.Fv_vv[i,j]])
                B_args.extend([u, v, self.a, self.n])
                self.B[i,j] = B_lambdified(*B_args)
        self.electric_energy_density = (self.Eu**2 + self.Ev**2)/2
        self.magnetic_energy_density = (self.B**2)/2
        self.hydraulic_energy_density = self.C**2 * self.V**2
        self.bulk_energy_density = - self.j
    
    def plot_solution(self):
        V, Fu, Fv, C = self.split(self.solution)
        plt.imshow(V, cmap='hot', interpolation='nearest')
        plt.show()
        plt.imshow(Fu, cmap='hot', interpolation='nearest')
        plt.show()
        plt.imshow(Fv, cmap='hot', interpolation='nearest')
        plt.show()
        plt.imshow(C, cmap='hot', interpolation='nearest')
        plt.show()

    def cart(self, u, v):
        h = self.a / (np.cosh(v) - np.cos(u))
        return h * np.sinh(v), h * np.sin(u)

    def plottable(self, f, nx, ny, max_x, max_y, sym = True):
        f_points = []
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                x, y, = self.cart(u, v)
                f_points.append((x, y, f[i,j]))
        min_x, min_y = -max_x, -max_y
        X = np.linspace(max_x/nx, max_x, nx)
        Y = np.linspace(min_y, max_y, ny)
        Z = np.zeros((nx, ny))
        def dist(x1, y1, x2, y2):
            return np.sqrt((x1-x2)**2 + (y1-y2)**2)
        for i, x in tqdm(enumerate(X)):
            for j, y in enumerate(Y):
                closest_dist = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
                for p in f_points:
                    px, py, pz = p
                    d = dist(x, y, px, py)
                    if d < closest_dist:
                        Z[i, j] = pz
                        closest_dist = d
        X_full = np.zeros(2*nx+1)
        X_full[nx+1:] = X
        for i in range(nx):
            X_full[nx-i-1] = -X[i]
        Y_full = Y
        Z_full = np.zeros((2*nx+1,ny))
        for i in range(nx):
            for j in range(ny):
                Z_full[nx+i+1,j] = Z[i,j]
                if sym == True:
                    Z_full[nx-i-1,j] = Z[i,j]
                else:
                    Z_full[nx-i-1,j] = -Z[i,j]
        if sym == True:
            Z_full[nx,:] = (Z_full[nx-1,:] + Z_full[nx+1,:])/2
        return X_full, Y_full, Z_full

    # sym = True for symmetric data, sym = False for antisymmetric data
    # diverging_cmap = True for diverging data (zero-centered), diverging_cmap = False for sequential data
    def plot_scalar(self, f, nx, ny, max_x, max_y, sym = True, diverging_cmap = True):
        X_full, Y_full, Z_full = self.plottable(f, nx, ny, max_x, max_y, sym)
        Z_full_flipped = np.zeros((Z_full.shape[1], Z_full.shape[0]))
        for i in range(len(X_full)):
            for j in range(len(Y_full)):
                Z_full_flipped[j,i] = Z_full[i,j]
        
        if diverging_cmap == False:
            plt.imshow(Z_full_flipped, cmap='hot', interpolation='nearest')
        else:
            plt.imshow(Z_full_flipped, cmap='RdBu', interpolation='nearest')
        plt.show()

    def plot_vector(self, Fu, Fv, nx, ny, max_x, max_y):
        Fx = np.zeros((self.nu, self.nv))
        Fy = np.zeros((self.nu, self.nv))
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                Fx[i,j], Fy[i,j] = BP2cart(Fu[i,j], Fv[i,j], u, v)
        norms = np.sqrt(Fx**2 + Fy**2)
        max_norm = max(norms.flatten())
        dx = max_x/nx
        dy = max_y/ny

        Fx = min(dx, dy) * Fx / max_norm
        Fy = min(dx, dy) * Fy / max_norm

        X_full, Y_full, Fx_full = self.plottable(Fx, nx, ny, max_x, max_y, sym = False)
        X_full, Y_full, Fy_full = self.plottable(Fy, nx, ny, max_x, max_y, sym = True)
        XX = []
        YY = []
        UU = []
        VV = []
        for i, x in enumerate(X_full):
            for j, y in enumerate(Y_full):
                XX.append(x)
                YY.append(y)
                UU.append(Fx_full[i,j])
                VV.append(Fy_full[i,j])
        fig1, ax1 = plt.subplots()
        Q = ax1.quiver(XX, YY, UU, VV, units='width', angles='xy', scale_units='xy', scale=2/3, pivot = 'middle')
        plt.xlim(-max_x, max_x)
        plt.ylim(-max_y, max_y)
        plt.savefig('pair_test.pdf')
        plt.show()
