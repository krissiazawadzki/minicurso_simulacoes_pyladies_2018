import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp




def oscilador_acoplado(t, quantidades_t, params_oh):
	'''
		quantidades_t = x1(t), x2(t), v1(t), v2(t)
	'''
	k1 = params_oh['k1']
	k2 = params_oh['k2']
	k3 = params_oh['k1']
	m1 = params_oh['m1']
	m1 = params_oh['m2']
	x1 = quantidades_t[0]
	x2 = quantidades_t[1]
	v1 = quantidades_t[2]
	v2 = quantidades_t[3]
	return (v1, v2, - k2 / m1 *(x1-x2) - k1 / m1 * x1, - k2 / m2 *(x2-x1) - k3 / m2 * x2 )





import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('-k1','--k1', type=float, default=1, help='k1 na mola 1')
parser.add_argument('-k2','--k2', type=float, default=1, help='k2 na mola 2')
parser.add_argument('-k3','--k3', type=float, default=1, help='k3 na mola 3')


parser.add_argument('-m1','--m1', type=float, default=1, help='massa 1')
parser.add_argument('-m2','--m2', type=float, default=1, help='massa 2')


# condicoes iniciais

parser.add_argument('-x10','--x10', type=float, default=0, help='posicao 1')
parser.add_argument('-x20','--x20', type=float, default=0, help='posicao 2')


parser.add_argument('-v10','--v10', type=float, default=1.0, help='velocidade 1')
parser.add_argument('-v20','--v20', type=float, default=1.0, help='velocidade 2')


# tempo de simulacao

parser.add_argument('-t0','--t0', type=float, default=0.0, help='tempo inicial')
parser.add_argument('-tf','--tf', type=float, default=10.0, help='tempo final')


opts = parser.parse_args()



k1 = opts.k1
k2 = opts.k2
k3 = opts.k3

m1 = opts.m1
m2 = opts.m2

params_oh = dict()
params_oh['k1'] = k1
params_oh['k2'] = k2
params_oh['k3'] = k3
params_oh['m1'] = m1
params_oh['m2'] = m2



t_0 = opts.t0
t_f = opts.tf

ts = np.linspace(t_0, t_f, 201)


'''
	modo paralelo
'''
x1_0 = opts.x10
v1_0 = opts.v10

x2_0 = opts.x20
v2_0 = opts.v20

resultado = solve_ivp(lambda t, q: oscilador_acoplado(t, q, params_oh), [t_0, t_f], (x1_0, x2_0, v1_0, v2_0), t_eval=ts) 


x1ts = resultado.y[0]
x2ts = resultado.y[1]


v1ts = resultado.y[2]
v2ts = resultado.y[3]



fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(111)

fig.subplots_adjust(bottom=0.15, left=0.15)

plt.ion()

ax.plot(ts[0], x1ts[0], color='blue', lw=2, label=r"$x_1(t)$")
ax.plot(ts[0], x2ts[0], color='red', lw=1., linestyle='--', label=r"$x_2(t)$")

ax.set_xlabel(r"$t$", fontsize=20)
ax.set_ylabel(r"$x_i(t)$", fontsize=20)

leg = ax.legend(ncol=2, loc='center', bbox_to_anchor=(0.5,1.1))
plt.draw()


ax.set_xlim(t_0, t_f)
ylim = max(x1ts.max(),x2ts.max())
ax.set_ylim(-ylim, ylim)

for i, t in enumerate(ts):
	ax.plot(ts[0:i], x1ts[0:i], color='blue', lw=2)
	ax.plot(ts[0:i], x2ts[0:i], color='red', lw=2., linestyle='--')
	plt.draw()
		
	plt.pause(0.05)






plt.show()
