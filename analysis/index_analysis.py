import numpy as np
import pylab as pl
import sys
import os
from scipy.optimize import curve_fit
from matplotlib.pyplot import cm
from scipy.odr import odrpack

def poly(X, a, b, *c):
    j,l = X
    poly = 0
    for i in range(n_term):
        poly = poly + c[i] * ((j-a)*l**(1/b))**i
    return poly

def poly_odr(pars, x):
    poly = 0
    for i in range(n_term):
        poly = poly + pars[1+i] * (x[0])**i
    return (x[1]**(2-pars[0]))*poly

# MEMORIZZO TUTTI I NOMI DEI FILE IN CARTELLA
# SUCCESSIVAMENTE CREO UN FILE TEMPORANEO DOVE METTO INSIEME TUTTI I DATI
basepath = sys.argv[1]
filenames = os.listdir(basepath)
path_filenames=[]
for entry in filenames:
    path_filenames.append(os.path.join(basepath, entry))

with open("./temp.dat", "w") as tempfile:
    for fname in path_filenames:
        with open(fname) as infile:
            for line in infile:
                tempfile.write(line)

# APRO I VARI FILE DA ANALIZZARE E CREO UN UNICO BLOCCO
L, J, ene_sp, err_ene_sp, ene_g, err_ene_g, ene_dens, err_ene_dens, susc, err_susc, G_pm, err_G_pm, C, err_C, U, err_U, corr_len, err_corr_len = np.genfromtxt("./temp.dat", delimiter ="\t", unpack = True)

# PROVO IL FIT PER LA LUNGHEZZA DI CORRELAZIONE
n_term_min = 3
n_term_max = 10

Jc = np.array([])
err_Jc = np.array([])
nu = np.array([])
err_nu = np.array([])
red_chisq = np.array([])
term = np.array([])
coeff = []
for n_term in range(n_term_min, n_term_max):
    c = np.ones(n_term)
    initParam = [0.70, 0.7117, *c]
    popt, pcov = curve_fit(poly, (J,L), corr_len/L, p0 = initParam, sigma =err_corr_len/L, absolute_sigma = True)
    chiq = np.sum(((corr_len/L - poly((J,L), *popt)) / (err_corr_len/L))**2)
    ndof = len(corr_len) - len(initParam)
    red_chiq = chiq/ndof
    perr = np.sqrt(np.diag(pcov))
    Jc = np.append(Jc, popt[0])
    err_Jc = np.append(err_Jc, perr[0])
    nu = np.append(nu, popt[1])
    err_nu = np.append(err_nu, perr[1])
    term = np.append(term, n_term)
    red_chisq = np.append(red_chisq, red_chiq)
    coeff.append(popt[2:])

c = np.array(coeff[np.argmin(red_chisq)])
popt = [Jc[np.argmin(red_chisq)], nu[np.argmin(red_chisq)], *c]
n_term = int(term[np.argmin(red_chisq)])

print("J_c = %f +- %f " % (Jc[np.argmin(red_chisq)], err_Jc[np.argmin(red_chisq)]))
print("nu  = %f +- %f " % (nu[np.argmin(red_chisq)], err_nu[np.argmin(red_chisq)]))
print("red_chisq %f" % (red_chisq[np.argmin(red_chisq)]))

# PROVO IL FIT PER LA SUSCETTIVITÀ, USO ODR PER TENERE IN CONTO
# DEGLI ERRORI SULLA LUNGHEZZA DI CORRELAZIONE CHE FA DA VARIABILE INDIPENDENTE
coeff = np.ones(n_term)
model = odrpack.Model(poly_odr)
x = np.row_stack( (corr_len/L, L) )
sigma_x = np.row_stack( (err_corr_len, np.array(([0.000000001])*len(err_corr_len))))
data = odrpack.RealData(x, susc, sx = sigma_x, sy = err_susc)
init_odr = [0.0378, *coeff]
odr = odrpack.ODR(data, model, beta0 = init_odr)
out = odr.run()

eta = out.beta[0]
err_eta = np.sqrt(out.cov_beta.diagonal())[0]

print("eta = %f +- %f" % (eta, err_eta))
print("red_chisq %f" % (red_chisq[np.argmin(red_chisq)]))

pl.scatter(corr_len/L, susc*L**(eta-2))

print(out.sum_square)


# APRO I VARI FILE E PROVO A FARE I GRAFICI INSIEME
# color = iter(cm.rainbow(np.linspace(0, 1, len(path_filenames))))
#
# for fname in path_filenames:
#     with open(fname) as infile:
#         col = next(color)
#         aux_L, aux_J, aux_ene_sp, aux_err_ene_sp, aux_ene_g, aux_err_ene_g, aux_ene_dens, aux_err_ene_dens, aux_susc, aux_err_susc, aux_G_pm, aux_err_G_pm, aux_C, aux_err_C, aux_U, aux_err_U, aux_corr_len, aux_err_corr_len = np.genfromtxt(infile, delimiter ="\t", unpack = True)
#         pl.figure(1)
#         pl.errorbar(aux_J, aux_corr_len/aux_L, aux_err_corr_len/aux_L, ls='', marker='o', color=col, fillstyle = 'none', label = 'L=' + str(int(aux_L[0])))
#         x = np.linspace(np.min(aux_J), np.max(aux_J), 500)
#         y = np.array([aux_L[0]]*len(x))
#         pl.plot(x, poly((x,y), *popt), color = col)
#         pl.figure(2)
#         pl.errorbar((aux_J-popt[0])*(aux_L**popt[1]), aux_corr_len/aux_L, aux_err_corr_len/aux_L, ls='', marker='o', color=col, fillstyle = 'none', label = 'L=' + str(int(aux_L[0])))
#
# pl.legend()

pl.show()

# ELIMINO IL FILE TEMPORANEO
os.remove("./temp.dat")
