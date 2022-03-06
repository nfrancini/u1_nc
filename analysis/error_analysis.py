import pylab as pl
import numpy as np
import sys
import os
from scipy.optimize import curve_fit

# FUNZIONI PER L'ANALISI DEGLI ERRORI
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

# FUNZIONI PER ESTRARRE LA GIUSTA OSSERVABILE
def media(obs):
    return np.mean(obs)

def specific_heat(obs):
    return V*(np.mean(obs**2) - (np.mean(obs))**2)

def binder(obs):
    return (np.mean(obs**2))/((np.mean(obs))**2)

def corr_lenght(obs1, obs2):
    return np.sqrt((np.mean(obs1)/np.mean(obs2)) -1) / (2*np.sin(np.pi/L))

# LETTURA DEL FILE DA TERMINALE
input_file = open(sys.argv[1], "r")

L, V, D, J, K = np.genfromtxt(input_file, dtype = "double", delimiter = "\t", unpack = True, max_rows = 1)
ene_sp, ene_g, ene_dens, susc, G_pm, mu2 = np.genfromtxt(input_file, dtype = "double", delimiter = "\t", unpack = True, skip_header = 2)

input_file.close()

# SCARTO LA TESTA DEL FILE SCARTO IL PRIMO DECIMO
skip = len(ene_dens)//10
ene_dens = ene_dens[skip:]
ene_sp = ene_sp[skip:]
ene_g = ene_g[skip:]
susc = susc[skip:]
G_pm = G_pm[skip:]
mu2 = mu2[skip:]

# CREO GLI ARRAY PER SALVARE L'ERRRORE
err_ene_sp = np.zeros(0)
err_ene_g = np.zeros(0)
err_ene = np.zeros(0)
err_susc = np.zeros(0)
err_G_pm = np.zeros(0)
err_spec_heat = np.zeros(0)
err_binder = np.zeros(0)
err_corr_len = np.zeros(0)

# FACCIO DUE VOLTE LA PROCEDURA DI JACKKNIFE E POI MEDIO SU DUE TAGLIE DIVERSE
taglie = np.array([len(ene_dens)//100, len(ene_dens)//200, len(ene_dens)//500])

for size in taglie:
    err_ene_sp = np.append(err_ene_sp, jack_error_1obs(media, ene_sp, size))
    err_ene_g = np.append(err_ene_g, jack_error_1obs(media, ene_g, size))
    err_ene = np.append(err_ene, jack_error_1obs(media, ene_dens, size))
    err_susc = np.append(err_susc, jack_error_1obs(media, susc, size))
    err_G_pm = np.append(err_G_pm, jack_error_1obs(media, G_pm, size))
    err_spec_heat = np.append(err_spec_heat, jack_error_1obs(specific_heat, ene_dens, size))
    err_binder = np.append(err_binder, jack_error_1obs(binder, mu2, size))
    err_corr_len = np.append(err_corr_len, jack_error_2obs(corr_lenght, susc, G_pm, size))

# STAMPO SU FILE I RISULTATI NEL FORMATO ENE_SP, ENE_G, ENE_DENS, SUSC, GPM, C, BINDER, CSI
if(os.path.isdir('./data_w_errors/L_%d' % L) == True):
    f_name = "L_%d_K_%f" % (L, K)
    output_file = open('./data_w_errors/L_%d/%s.dat' % (L, f_name) , 'a')
else:
    dir_name = "./data_w_errors/L_%d" % (L)
    os.makedirs(dir_name, exist_ok = False)
    f_name = "L_%d_K_%f" % (L, K)
    output_file = open('./data_w_errors/L_%d/%s.dat' % (L, f_name) , 'a')

output_file.write(str(J) + '\t')                        # COLONNA 1
output_file.write(str(np.mean(ene_sp)) + '\t')          # COLONNA 2
output_file.write(str(np.mean(err_ene_sp)) + '\t')      # COLONNA 3
output_file.write(str(np.mean(ene_g)) + '\t')           # COLONNA 4
output_file.write(str(np.mean(err_ene_g)) + '\t')       # COLONNA 5
output_file.write(str(np.mean(ene_dens)) +'\t')         # COLONNA 6
output_file.write(str(np.mean(err_ene)) + '\t')         # COLONNA 7
output_file.write(str(np.mean(susc)) + '\t')            # COLONNA 8
output_file.write(str(np.mean(err_susc)) + '\t')        # COLONNA 9
output_file.write(str(np.mean(G_pm)) + '\t')            # COLONNA 10
output_file.write(str(np.mean(err_G_pm)) + '\t')        # COLONNA 11
output_file.write(str(specific_heat(ene_dens)) + '\t')  # COLONNA 12
output_file.write(str(np.mean(err_spec_heat)) +'\t')    # COLONNA 13
output_file.write(str(binder(mu2)) + '\t')              # COLONNA 14
output_file.write(str(np.mean(err_binder)) +'\t')       # COLONNA 15
output_file.write(str(corr_lenght(susc, G_pm)) + '\t')  # COLONNA 16
output_file.write(str(np.mean(err_corr_len)) + '\n')    # COLONNA 17

output_file.close()

# CONTROLLO CON PLOT
# pl.xscale('log')
# pl.yscale('log')
# pl.scatter(taglie, err_ene)
# pl.scatter(taglie, err_susc)
# pl.scatter(taglie, err_G_pm)
# pl.scatter(taglie, err_spec_heat)
# pl.scatter(taglie, err_binder)
# pl.scatter(taglie, err_corr_len)
# pl.scatter(taglie, err_ene_g)
# pl.show()
