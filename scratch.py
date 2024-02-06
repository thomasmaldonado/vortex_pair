from pair import Pair

n = 1
k = 1
a = 1
nu = 30
nv = 30
max_v = 2

p = Pair(n=n, k=k, a=a, nu=nu, nv=nv, max_v=max_v)
p.solve()
p.save('test.npy')
#p = Pair(filename = 'test.npy')
print(p.ier)
p.observables()
p.plot_vector(p.Ju, p.Jv, 20,20,4*a,4*a)
p.plot_vector(p.Eu, p.Ev, 20, 20, 4*a, 4*a)
p.plot_scalar(p.V, 200,200,4*a,4*a,sym=True, diverging_cmap=False)
p.plot_scalar(p.C, 200,200,4*a,4*a,sym=True, diverging_cmap=False)
p.plot_scalar(p.magnetic_energy_density, 200,200,4*a,4*a,sym=True, diverging_cmap=False)
p.plot_scalar(p.electric_energy_density, 200,200,4*a,4*a,sym=True, diverging_cmap=False)
p.plot_scalar(p.hydraulic_energy_density, 200,200,4*a,4*a,sym=True, diverging_cmap=False)
#p2 = Pair(filename = 'test.npy')
#print(p2.ier)
#p2.observables()
#p2.plot_scalar(p.magnetic_energy_density, 50,50,4*a,4*a,sym=True, diverging_cmap=False)
#p2.plot_scalar(p.electric_energy_density, 50,50,4*a,4*a,sym=True, diverging_cmap=False)
#p2.plot_scalar(p.hydraulic_energy_density-p.bulk_energy_density, 50,50,4*a,4*a,sym=True, diverging_cmap=True)

"""
p.plot_scalar(p.B, 50, 50, 4*a,4*a, sym = False)
p.plot_vector(p.Ju, p.Jv, 20, 20, 4*a, 4*a)
#p.plot_solution()
"""