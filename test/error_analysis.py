import pylab as pl
import numpy as np
import erroranalysis as ea
from scipy.optimize import curve_fit

def jack_estimate_1obs(func, obs, block_size, index):
    resample = np.concatenate((obs[:index*block_size], obs[block_size*index+block_size:]))
    return func(resample)

def jack_error_1obs(func, obs, block_size):
    est = np.zeros(len(obs)//block_size)
    for x in range(0, len(obs)//block_size):
        est[x] = jack_estimate_1obs(func, obs, block_size, x)
    sum1 = np.sum(est**2)
    sum2 = np.sum(est)
    sum1 = sum1/(len(obs)/block_size)
    sum2 = (sum2/(len(obs)/block_size))**2
    return np.sqrt(len(obs)/block_size - 1) * np.sqrt(sum1-sum2)

def jack_estimate_2obs(func, obs1, obs2, block_size, index):
    resample1 = np.concatenate((obs1[:index*block_size], obs1[block_size*index+block_size:]))
    resample2 = np.concatenate((obs2[:index*block_size], obs2[block_size*index+block_size:]))
    return func(resample1, resample2)

def jack_error_2obs(func, obs1, obs2, block_size):
    est = np.zeros(len(obs1)//block_size)
    for x in range(0, len(obs1)//block_size):
        est[x] = jack_estimate_2obs(func, obs1, obs2, block_size, x)
    sum1 = np.sum(est**2)
    sum2 = np.sum(est)
    sum1 = sum1/(len(obs1)/block_size)
    sum2 = (sum2/(len(obs1)/block_size))**2
    return np.sqrt(len(obs1)/block_size - 1) * np.sqrt(sum1-sum2)

def media(obs):
    return np.mean(obs)

def specific_heat(obs):
    return V*(np.mean(obs**2) - (np.mean(obs))**2)

def binder(obs):
    return (np.mean(obs**2))/((np.mean(obs))**2)

def corr_lenght(obs1, obs2):
    return np.sqrt((np.mean(obs1)/np.mean(obs2)) -1) / (2*np.sin(np.pi/L))

fpath = '/home/n-francini/Scrivania/TESI/SIMULAZIONI NUMERICHE/u1_nc/data/L_6/J_0.30000_k_0.00000.dat'

V = 125
L = 5
skip = 1000
L, V, D, J, K = np.genfromtxt(fpath, dtype = "double", delimiter = "\t", unpack = True, max_rows = 1)
ene_sp, ene_g, ene_dens, susc, G_pm, mu2 = np.genfromtxt(fpath, dtype = "double", delimiter = "\t", unpack = True, skip_header = skip+3)

max_len = 8000
ene_dens = ene_dens[0:max_len]
ene_sp = ene_sp[0:max_len]
ene_g = ene_g[0:max_len]
susc = susc[0:max_len]
G_pm = G_pm[0:max_len]
mu2 = mu2[0:max_len]

err_ene_sp = np.zeros(0)
err_ene_g = np.zeros(0)
err_ene = np.zeros(0)
err_susc = np.zeros(0)
err_G_pm = np.zeros(0)
err_spec_heat = np.zeros(0)
err_binder = np.zeros(0)
err_corr_len = np.zeros(0)

# taglie = np.array([100, 200, 300, 400, 500, 900, 1000])
taglie = np.array([25, 50, 100])

for size in taglie:
    err_ene_sp = np.append(err_ene_sp, jack_error_1obs(media, ene_sp, size))
    err_ene_g = np.append(err_ene_g, jack_error_1obs(media, ene_g, size))
    err_ene = np.append(err_ene, jack_error_1obs(media, ene_dens, size))
    err_susc = np.append(err_susc, jack_error_1obs(media, susc, size))
    err_G_pm = np.append(err_G_pm, jack_error_1obs(media, G_pm, size))
    err_spec_heat = np.append(err_spec_heat, jack_error_1obs(specific_heat, ene_dens, size))
    err_binder = np.append(err_binder, jack_error_1obs(binder, mu2, size))
    err_corr_len = np.append(err_corr_len, jack_error_2obs(corr_lenght, susc, G_pm, size))

print("DENSITÀ DI ENERGIA = ", np.mean(ene_dens), "+-", np.mean(err_ene))
print("DENSITÀ DI ENERGIA SPIN= ", np.mean(ene_sp), "+-", np.mean(err_ene_sp))
print("DENSITÀ DI ENERGIA GAUGE= ", np.mean(ene_g), "+-", np.mean(err_ene_g))
print("SUSCETTIVITÀ = ", np.mean(susc), "+-", np.mean(err_susc))
print("G_PM = ", np.mean(G_pm), "+-", np.mean(err_G_pm))
print("CALORE SPECIFICO = ", specific_heat(ene_dens), "+-", np.mean(err_spec_heat))
print("BINDER = ", binder(mu2), "+-", np.mean(err_binder))
print("LUNGHEZZA DI CORRELAZIONE = ", corr_lenght(susc, G_pm), "+-", np.mean(err_corr_len))

pl.xscale('log')
pl.yscale('log')
pl.scatter(taglie, err_ene)
pl.scatter(taglie, err_susc)
pl.scatter(taglie, err_G_pm)
pl.scatter(taglie, err_spec_heat)
pl.scatter(taglie, err_binder)
pl.scatter(taglie, err_corr_len)
pl.scatter(taglie, err_ene_g)
pl.show()
