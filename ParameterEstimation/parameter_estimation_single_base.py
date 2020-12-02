import numpy as np
import scipy.optimize as opt
import pandas as pd
import csv
import matplotlib.pyplot as plt
from timeout import timeout

import scipy_vewk3 as models

# Taken from Segers et al., Hypertension (2000)
closed_loop_base_pars = {'C_ao': 1.13, 
                         'E_max': 1.5, 
                         'E_min': 0.03,
                         'R_mv': 0.006,
                         'R_sys': 1.11,
                         'T': 0.85,
                         'Z_ao': 0.033,
                         't_peak': 0.3,
                         'C_sv': 11,
                         'V_tot': 300
                        }


@timeout(240)
def cost_function(pars, measurements=dict(P_sys=120, P_dia=80), active_pars=None, ret_all=False,
                base_pars=dict(closed_loop_base_pars)):
    """ general purpose wrapper for optimization with both lmfit and scipy"""
    measurement_scales = dict(P_sys=120, 
                              P_dia=80,
                              P_meas=100,
                              P_ao=100,
                              Q_lvao=500,
                              Q_aosv=500,
                              Q_svlv=300,
                              SV=100,
                              Q_max=500,
                              PP=40)
    
    it_base_pars = dict(base_pars)
    for idx, name in enumerate(active_pars):
        try: #LMFIT
            it_base_pars[name] = pars[name] 
        except IndexError: #SCIPY
            it_base_pars[name] = pars[idx]
    ve_closed = models.VaryingElastance()
    ve_closed.set_pars(**it_base_pars)
    var_dict, t_eval, scipy_sol = models.solve_to_steady_state(ve_closed, n_cycles=10)
    ret_dict = models.calc_summary(var_dict)
    residual = []
    for name, val in measurements.items():
        try:
            residual.append((val - ret_dict[name])/measurement_scales[name])
        except KeyError:
            residual.append((val - var_dict[name])/measurement_scales[name])
    
    residual = np.array(residual).flatten()
    if ret_all:
        return residual, var_dict, t_eval, ret_dict
    else:
        return residual

@timeout(240)
def fit(x0, active_params, measurements,bounds, bounds_low, base_pars):
    residual, var_dict, t_eval, ret_dict = cost_function(x0, 
                                                         active_pars=active_params,
                                                         measurements=measurements, ret_all=True)
    
    results = opt.least_squares(cost_function, x0,
                                xtol=2.3e-16,ftol=2.3e-16,gtol=2.3e-16,diff_step=1e-3,
                                bounds=(bounds_low,bounds),
                                kwargs=dict(active_pars=active_params, measurements=measurements, base_pars=base_pars))
    
    residual, var_dict, t_eval, ret_dict = cost_function(results.x, 
                                                         active_pars=active_params,
                                                         measurements=measurements, ret_all=True)
    
    return results, residual, var_dict, t_eval, ret_dict


def run_multi_singlepoints():
    SingleFileParameters = open('ClinicalParameters.csv','w')
    SingleFileErrors = open('ClinicalErrors.csv','w')
    SingleFileResults = open('ClinicalResults.csv','w')
    writerSFP = csv.writer(SingleFileParameters)
    writerSFE = csv.writer(SingleFileErrors)
    writerSFR = csv.writer(SingleFileResults)
    #Outfiles
    SeriesPress = open('SeriesPressure.csv','a')
    SeriesVol = open('SeriesVolume.csv','a')
    SeriesFlow = open('SeriesFlow.csv','a')
    SeriesVals = open('SeriesValues.csv','a')
    SeriesVeinPress = open('SeriesVeinPress.csv', 'a')
    SeriesTime = open('SeriesTime.csv', 'a')
    writerSPress = csv.writer(SeriesPress)
    writerSVol = csv.writer(SeriesVol)
    writerSFlow = csv.writer(SeriesFlow)
    writerSVals = csv.writer(SeriesVals)
    writerSVeinP = csv.writer(SeriesVeinPress)
    writerSTime = csv.writer(SeriesTime)
    
    ve_closed = models.VaryingElastance()
    ve_closed.set_pars(**closed_loop_base_pars)
    var_dict_0, t_eval_0, scipy_sol = models.solve_to_steady_state(ve_closed, n_cycles=10)
    ret_dict_0 = models.calc_summary(var_dict_0)
    print(pd.DataFrame(ret_dict_0, index=[0]))
    
    np.random.seed(876554321)
    
    signalnoise = 0.05
    measurements = dict()

    for meas in ["P_sys", "P_dia", "PP", "SV", "Q_max"]:
        measurements[meas] = ret_dict_0[meas]
    
    np.random.seed(112233)
    
    ll = 50
    ps = 9 # The number of estimated parameters in the subset
    
    active_params = ["V_tot", "E_max", "C_ao", "R_sys", "t_peak", "C_sv", "E_min",  "Z_ao", "R_mv"]
    noise = 0.3
    x0 = np.array([250.0, 2.0, 1.0, 1.0, 0.32, 10.0, 0.06, 0.1, 0.003])
    x_scale = [500.0, 2.0, 2.0, 2.0, 0.32, 10.0, 0.05, 0.1, 0.005]
    bounds = [2000.0, 5.0, 10.0, 3.0, 0.75, 30.0, 1.0, 1.0, 0.1]
    bounds_low = [50.0, 0.9, 0.5, 0.5, 0.05, 0.5, 0.0, 0.0, 0.0]
    
    seeds = 112233
    
    x_scale = x_scale[:ps]
    bounds = bounds[:ps]
    bounds_low = bounds_low[:ps]
    active_params = active_params[:ps]    
    
    x0array = np.zeros([ll,ps])

    for ii in range(0,ps):
        temp_x = x0[ii]
        for jj in range(0,ll):
            x0array[jj,ii] = temp_x*(1. + noise*np.random.randn())


    results_header = ["Iteration",] + active_params + ["cost", "status"]
    print(results_header)
    results_list = []
    
    total_list = np.zeros([len(active_params),ll])
    total_error = np.zeros([len(active_params),ll])
    for kk in range(ll):
        print('Iteration #'+str(kk)+' initiated')
        x0 = x0array[kk,:]
        
        x0 = np.min([x0, bounds],axis=0)
        x0 = np.max([x0, bounds_low],axis=0)
        base_pars = dict(closed_loop_base_pars)
        try:
            results, residual, var_dict, t_eval, ret_dict = fit(x0, active_params, measurements, bounds, bounds_low, base_pars)
            
            print(active_params)
            estimated_pars = results.x
            print(estimated_pars)
            print([closed_loop_base_pars[par] for par in active_params])
            print("Percent ERROR in Parameter estimate")
            errorvec = [(par_est- closed_loop_base_pars[par])/closed_loop_base_pars[par] for par_est,par in zip(estimated_pars,active_params)]
            print(errorvec)
            fig = plt.figure(kk)
            plt.plot(t_eval_0, var_dict_0["P_ao"], label="data") 
            plt.plot(t_eval, var_dict["P_ao"], label="opt")
            plt.legend()
            fig.savefig('Fit'+str(kk), pad_inches = 0.2)
            plt.close()
        
            total_list[:,kk] = estimated_pars
            total_error[:,kk] = errorvec
            results_list.append([kk, results.x, results.cost, results.status])
            print([kk, results.x, results.cost, results.status])
            writerSFP.writerow(estimated_pars)
            writerSFE.writerow(errorvec)
            writerSFR.writerow([kk, results.x, results.cost, results.status])
            writerSPress.writerow(var_dict["P_ao"])
            writerSVol.writerow(var_dict["V_lv"])
            writerSFlow.writerow(var_dict["Q_lvao"])
            writerSVals.writerow([ret_dict["SV"], ret_dict["P_map"], ret_dict["PP"], ret_dict["V_sys"], ret_dict["V_dia"], ret_dict["stroke_work_1"], ret_dict["Q_max"]])
            writerSVeinP.writerow(var_dict["P_sv"] )
            writerSTime.writerow(var_dict["t"])
        except Exception as exc:
            print(exc)
    
    print(np.sum(total_list,1)/ll)    
    print(np.sum(total_error,1)/ll)
    SingleFileParameters.close()
    SingleFileErrors.close()
    SingleFileResults.close()
    return results_list




if __name__ == "__main__":
    run_multi_singlepoints()
    
