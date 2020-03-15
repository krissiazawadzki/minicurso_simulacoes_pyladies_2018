import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp



def oscilador(t, quantidades_t, omega_0):
	'''
		quantidades_t = x(t), v(t)
	'''
	x = quantidades_t[0]
	v = quantidades_t[1]
	return (v, - omega_0**2 * x)



'''

	main
	
'''


omega_0 = 2.0

t_0 = 0.0
t_f = 10.0

ts = np.linspace(t_0, t_f, 1001)

x_0 = 0
v_0 = 1.0

resultado = solve_ivp(lambda t, q: oscilador(t, q, omega_0), [t_0, t_f], (x_0, v_0), t_eval=ts) 


xts = resultado.y[0]
vts = resultado.y[1]




fig = plt.figure()
ax = fig.add_subplot(211)

ax.plot(ts, xts, color='#de2440')


ax.set_xlabel(r"$t$", fontsize=20)
ax.set_ylabel(r"$x(t)$", fontsize=20)


plt.show()
