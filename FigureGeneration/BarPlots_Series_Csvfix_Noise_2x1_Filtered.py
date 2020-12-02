from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

mean_cost_0 = np.zeros([8])
std_cost_0 = np.zeros([8])

threshold_pct = 0.25#0.75
threshold_mat = np.zeros([8])
folderpaths = ["./"]
parameter_order = ["V_tot", "E_max", "R_sys", "C_ao", "t_peak", "E_min", "Z_ao", "R_mv"]

filename = "SeriesResults.csv"
folderpath = "./"
for ii, ee in enumerate(parameter_order):
    # Enter appropriate directory
    folder = folderpath+str(ii+1)+"T"
    path = os.path.join("./",folder,filename)
    
    data = pd.read_csv(path, index_col=None, header=None)
    
    data = np.asarray(data[2])
    
    print(data.shape)
    print(folder)
    tempmean = np.mean(data)  
    tempstd = np.sqrt(np.var(data))
    threshold = np.min(data)*(1. + threshold_pct)
    
    print(tempmean.shape)
    mean_cost_0[ii] = tempmean
    std_cost_0[ii] = tempstd
    threshold_mat[ii] = threshold


filename = "SeriesErrors.csv"

mean_0 = np.zeros([8,8])
mean_abs_0 = mean_0.copy()
std_devs_0 = np.zeros([8,8])
std_devs_abs_0 = np.zeros([8,8])
top = 7
for ii, ee in enumerate(parameter_order):
    # Enter appropriate directory
    folder = folderpath+str(ii+1)+"T"
    path = os.path.join("./",folder,filename)
    data = np.genfromtxt(path, delimiter = ",")
    data = 100.*data
    path = os.path.join("./",folder,"SeriesResults.csv")
    dataCost = pd.read_csv(path, index_col=None, header=None)
    dataCost = np.asarray(dataCost[2])

    if ii == 0:
        datatemp = data[dataCost <= threshold_mat[ii]]
    else:
        datatemp = data[dataCost <= threshold_mat[ii],:]
    data = datatemp
    
    absdata = np.abs(data)
    if ii == 0:
        tempmean = np.mean(data)
        tempmeanabs = np.mean(absdata)
        tempstd = np.sqrt(np.var(data))
        tempstdabs = np.sqrt(np.var(absdata))    
    else:
        tempmean = np.mean(data,axis=0)
        tempmeanabs = np.mean(absdata,axis=0)
        tempstd = np.sqrt(np.var(data,axis=0))
        tempstdabs = np.sqrt(np.var(absdata,axis=0))
    print(tempmean.shape, tempmeanabs.shape)
    mean_0[top-ii,0:ii+1] = tempmean
    mean_abs_0[top-ii,0:ii+1] = tempmeanabs
    std_devs_0[top-ii,0:ii+1] = tempstd
    std_devs_abs_0[top-ii,0:ii+1] = tempstdabs
    
print(mean_abs_0)
print(std_devs_abs_0)

##################
# ABSOLUTE ERROR #
##################
fig_ratio = (6,6) #Inches
fig, axs = plt.subplots(2,1,num="NoiseAbsTimeseries95", figsize=fig_ratio, dpi=300, facecolor='w', edgecolor='k')
ax = axs[1]
ax.set_xlabel(r'Model parameters, $\theta$')
ax.set_ylabel('Relative error [%]')
vars = ['8 Parameters', '7 Parameters', '6 Parameters', '5 Parameters']
var_latex = [r'8 Parameters', r'7 Parameters', r'6 Parameters', r'5 Parameters']

mean = mean_abs_0[0:5,:]
std_devs = std_devs_abs_0[0:5,:]

colors = ["#FF0000", "#FFACAC", "#0000FF", "#9D9DFF", "#FFFF00"]
patterns = ["", "//", "",  "//", ""]

nvars = 5
pad = 1
xvals = [ [i,nvars+i+pad,2*(nvars + pad) + i,3*(nvars + pad) + i,4*(nvars + pad) + i,5*(nvars + pad) + i,6*(nvars + pad) + i,7*(nvars + pad) + i,8*(nvars + pad) + i]  for i in range(nvars)]

offs = 2.0
xtick_loc = [offs, nvars + pad + offs, 2*(nvars+pad) + offs, 3*(nvars+pad) + offs, 4*(nvars+pad) + offs, 5*(nvars+pad) + offs, 6*(nvars+pad) + offs, 7*(nvars+pad) + offs]
print(xtick_loc)
labeled_bars =[]
for ll in range(len(xtick_loc)):
    for kk, var_name  in enumerate(vars):
        bar_loc = xvals[kk]
        if (mean[kk][ll] != 0.0):
            bars = ax.bar(bar_loc[ll], mean[kk][ll], yerr=std_devs[kk][ll], color=colors[kk],
                       error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1), label=var_latex[kk][ll], hatch = patterns[kk])



ax.set_xticks(xtick_loc)
ax.set_yscale('log')
ax.set_xlim([-1.0, 48.0])
ax.set_ylim([0.1, 200.0])
ax.set_xticklabels([r'$V_{tot}$',r'$E_{max}$',r'$C_{ao}$',r'$R_{sys}$', r'$t_{peak}$',r'$E_{min}$',r'$Z_{ao}$',r'$R_{mv}$'])

ax.legend(var_latex)

ax = axs[0]
ax.set_title('MAPE for parameter estimates using\n'+'noisy waveform data with fixed $C_{sv}$',pad=15)
ax.set_xlabel(r'Model parameters, $\theta$')
ax.set_ylabel('Relative error [%]')
vars = ['4 Parameters', '3 Parameters', '2 Parameters', '1 Parameter']
var_latex = [r'4 Parameters', r'3 Parameters', r'2 Parameters', r'1 Parameters']
mean = mean_abs_0[4:,:]
std_devs = std_devs_abs_0[4:,:]

nvars = 4
pad = 1
xvals = [ [i,nvars+i+pad,2*(nvars + pad) + i,3*(nvars + pad) + i,4*(nvars + pad) + i]  for i in range(nvars)]
print(xvals)
offs = 1.5
xtick_loc = [offs, nvars + pad + offs, 2*(nvars+pad) + offs, 3*(nvars+pad) + offs]
print(xtick_loc)
labeled_bars =[]
for ll in range(len(xtick_loc)):
    for kk, var_name  in enumerate(vars):
        bar_loc = xvals[kk]
        if (mean[kk][ll] != 0.0):
            bars = ax.bar(bar_loc[ll], mean[kk][ll], yerr=std_devs[kk][ll], color=colors[kk],
                       error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1), label=var_latex[kk][ll], hatch = patterns[kk])


ax.set_xticks(xtick_loc)
ax.set_yscale('log')
ax.set_xlim([-1.0, 18.5])
ax.set_ylim([0.001, 2.5])

ax.set_xticklabels([r'$V_{tot}$',r'$E_{max}$',r'$C_{ao}$',r'$R_{sys}$', r'$t_{peak}$',r'$E_{min}$',r'$Z_{ao}$',r'$R_{mv}$'])

ax.legend(var_latex)
fig.tight_layout()
plt.savefig('CsvfixNoiseSeriesSingle_Filtered.svg')
plt.savefig('CsvfixNoiseSeriesSingle_Filtered.pdf')
plt.show()
