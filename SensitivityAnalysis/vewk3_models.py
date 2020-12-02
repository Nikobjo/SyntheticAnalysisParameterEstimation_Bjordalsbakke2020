import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt


ml_per_sec_to_L_per_min = 60/1000

def stergiopolous_elastance(self, t):
    """
    Computes the normalized elastance at time t, according to the shape parameters given by
    Stergiopolus et al. (1996)
    """
    # Note changing these may result in a non-nomralized elastance curve
    a1 = 0.708 * self.t_peak/self.T
    a2 = 1.677 * a1
    n1 = 1.32
    n2 = 21.9
    alpha = 1.672
    shapeFunction1 = (t/(a1*self.T))**n1 / (1.0 + (t / (a1*self.T)) ** n1)
    shapeFunction2 = (1.0 + (t/(a2*self.T))**n2)
    e = alpha * shapeFunction1/shapeFunction2
    return e


class VaryingElastanceOpenLoop():
    def __init__(self):
        self.E_max = 3.0
        self.E_min = 0.3
        self.t_peak = 0.4
        self.T = 60/73
        self.e_std = 1/np.sqrt(80)
        self.Z_ao = 0.1
        self.C_ao = 1
        self.R_sys = 1
        self.P_sv = 7.5
        self.R_mv = 0.006
        self.V_tot = 100 + 80 + 100
        self.elastance_fcn = stergiopolous_elastance


    def elastance(self, tau):
        return self.elastance_fcn(self, tau)


    def set_pars(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
            else:
                print("Warning: object has no attribute %s" % key)


    def calc_consistent_initial_values(self, V_lv_0=100, P_ao_0=100):
        u0 = (V_lv_0, P_ao_0)
        return u0


    def rhs(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        P_sv = self.P_sv
        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv
        Q_lvao = (P_lv > P_ao)*(P_lv - P_ao)/self.Z_ao
        Q_aosv = (P_ao - P_sv)/self.R_sys
        Q_svlv = (P_sv > P_lv)*(P_sv - P_lv)/self.R_mv

        der_V_lv = Q_svlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_u = [der_V_lv, der_P_ao]
        return der_u
    
    def calc_all(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        P_sv = self.P_sv*np.ones_like(t)
        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv
        Q_lvao = (P_lv - P_ao)/self.Z_ao * (P_lv > P_ao)
        Q_aosv = (P_ao-P_sv)/self.R_sys
        Q_svlv = (P_sv - P_lv)/self.R_mv * (P_sv > P_lv)

        all_vars = locals()
        del all_vars["self"]
        del all_vars["u"]

        der_V_lv = Q_svlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_u = [der_V_lv, der_P_ao,]
        return der_u, all_vars
 

class VaryingElastanceComplete():
    def __init__(self):
        self.E_max = 3.0
        self.E_min = 0.3
        self.t_peak = 0.3
        self.T = 60/73
        self.e_std = 1/np.sqrt(80)
        self.Z_ao = 0.1
        self.C_ao = 1
        self.R_sys = 1
        self.C_sv = 15 # 1/0.0059
        self.R_tc = 0.006*2 # In Chase et al.'s model it's twice that of the mitral valve
        self.E_rv_ratio = 0.2031391069 # from me
        self.R_pv = 0.0055
        self.C_pa = 1/.369
        self.R_pul = .1552
        self.C_pv = 100 #1/0.0073
        self.R_mv = 0.006
        self.V_tot = 100 + 80 + 100 + 100 + 100 + 2000
        self.elastance_fcn = stergiopolous_elastance


    def elastance(self, tau):
        return self.elastance_fcn(self, tau)


    def set_pars(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
            else:
                print("Warning: object has no attribute %s" % key)


    def calc_consistent_initial_values(self, V_lv_0=100, P_ao_0=100, V_rv_0=100, P_pa_0=30):
        V_ao_0 = self.C_ao*P_ao_0
        V_pa_0 = self.C_pa*P_pa_0
        V_pv_0 = (self.V_tot - V_lv_0 - V_rv_0 - V_ao_0 - V_pa_0)/(1+self.C_sv/self.C_pv)
        V_sv_0 = self.V_tot - V_lv_0  - V_rv_0 - V_ao_0 - V_pa_0 - V_pv_0
        P_sv_0 = V_sv_0/self.C_sv
        P_pv_0 = V_pv_0/self.C_pv
        u0 = (V_lv_0, P_ao_0, P_sv_0, V_rv_0, P_pa_0, P_pv_0)
        #print(V_lv_0, V_ao_0, V_sv_0, V_rv_0, V_pa_0, V_pv_0)
        #print(V_lv_0 + V_ao_0 + V_sv_0 + V_rv_0 + V_pa_0 + V_pv_0)
        #assert (V_lv_0 + V_ao_0 + V_sv_0 + V_rv_0 + V_pa_0 + V_pv_0) nearly equals self.V_tot
        #print(u0)
        return u0


    def rhs(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        V_ao = self.C_ao * P_ao
        P_sv = u[2]
        V_sv = self.C_sv * P_sv
        V_rv = u[3]
        P_pa = u[4]
        V_pa = self.C_pa * P_pa
        P_pv = u[5]
        V_pv = self.C_pv * P_pv

        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv
        P_rv = E * self.E_rv_ratio * V_rv
     
        Q_lvao = (P_lv > P_ao)*(P_lv - P_ao)/self.Z_ao
        Q_aosv = (P_ao - P_sv)/self.R_sys
        Q_svrv = (P_sv > P_rv)*(P_sv - P_rv)/self.R_tc
        Q_rvpa = (P_rv > P_pa)*(P_rv - P_pa)/self.R_pv
        Q_papv = (P_pa - P_pv)/self.R_pul
        Q_pvlv = (P_pv > P_lv)*(P_pv - P_lv)/self.R_mv

        der_V_lv = Q_pvlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_P_sv = (Q_aosv - Q_svrv)/self.C_sv
        der_V_rv = Q_svrv - Q_rvpa
        der_P_pa = (Q_rvpa - Q_papv)/self.C_pa
        der_P_pv = (Q_papv - Q_pvlv)/self.C_pv
        der_u = [der_V_lv, der_P_ao, der_P_sv, der_V_rv, der_P_pa, der_P_pv]
        return der_u


    def calc_all(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        V_ao = self.C_ao * P_ao
        P_sv = u[2]
        V_sv = self.C_sv * P_sv
        V_rv = u[3]
        P_pa = u[4]
        V_pa = self.C_pa * P_pa
        P_pv = u[5]
        V_pv = self.C_pv * P_pv
        V_tot = V_ao + V_sv + V_rv + V_pa + V_pv + V_lv

        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv
        P_rv = E * self.E_rv_ratio * V_lv
               
        Q_lvao = (P_lv > P_ao)*(P_lv - P_ao)/self.Z_ao
        Q_aosv = (P_ao - P_sv)/self.R_sys
        Q_svrv = (P_sv > P_lv)*(P_sv - P_lv)/self.R_tc
        Q_rvpa = (P_rv > P_pa)*(P_rv - P_pa)/self.R_pv
        Q_papv = (P_pa - P_pv)/self.R_pul
        Q_pvlv = (P_pv > P_lv)*(P_pv - P_lv)/self.R_mv

        
        all_vars = locals()
        del all_vars["self"]
        del all_vars["u"]
        
        der_V_lv = Q_pvlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_P_sv = (Q_aosv - Q_svrv)/self.C_sv
        der_V_rv = Q_svrv - Q_rvpa
        der_P_pa = (Q_rvpa - Q_papv)/self.C_pa
        der_P_pv = (Q_papv - Q_pvlv)/self.C_pv
        der_u = [der_V_lv, der_P_ao, der_P_sv, der_V_rv, der_P_pa, der_P_pv]
        return der_u, all_vars



class VaryingElastance():
    def __init__(self):
        self.E_max = 3.0
        self.E_min = 0.3
        self.t_peak = 0.4
        self.T = 60/73
        self.e_std = 1/np.sqrt(80)
        self.Z_ao = 0.1
        self.C_ao = 1
        self.R_sys = 1
        self.C_sv = 15
        self.R_mv = 0.006
        self.V_tot = 100 + 80 + 100
        self.elastance_fcn = stergiopolous_elastance


    def elastance(self, tau):
        return self.elastance_fcn(self, tau)


    def set_pars(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
            else:
                print("Warning: object has no attribute %s" % key)


    def calc_consistent_initial_values(self, V_lv_0=100, P_ao_0=100):
        V_ao_0 = self.C_ao*P_ao_0
        P_sv_0 = (self.V_tot - V_lv_0 - V_ao_0)/self.C_sv
        u0 = (V_lv_0, P_ao_0, P_sv_0)
        return u0


    def rhs(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        P_sv = u[2]
        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv
        Q_lvao = (P_lv > P_ao)*(P_lv - P_ao)/self.Z_ao
        Q_aosv = (P_ao - P_sv)/self.R_sys
        Q_svlv = (P_sv > P_lv)*(P_sv - P_lv)/self.R_mv

        der_V_lv = Q_svlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_P_sv = (Q_aosv - Q_svlv)/self.C_sv
        der_u = [der_V_lv, der_P_ao, der_P_sv]
        return der_u


    def calc_all(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        P_sv = u[2]
        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv
        Q_lvao = (P_lv - P_ao)/self.Z_ao * (P_lv > P_ao)
        Q_aosv = (P_ao-P_sv)/self.R_sys
        Q_svlv = (P_sv - P_lv)/self.R_mv * (P_sv > P_lv)
        P_meas = np.maximum(P_lv, P_ao)

        all_vars = locals()
        del all_vars["self"]
        del all_vars["u"]

        der_V_lv = Q_svlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_P_vc = (Q_aosv - Q_svlv)/self.C_sv
        der_u = [der_V_lv, der_P_ao, der_P_vc]
        return der_u, all_vars
    
    

def plot_flows(var_dict):
    t = var_dict["t"]
    plt.figure()
    plt.plot(t, var_dict["Q_lvao"], label="Q_av")
    plt.plot(t, var_dict["Q_aosv"], label="Q_sys")
    plt.plot(t, var_dict["Q_svlv"], label="Q_mv")
    plt.ylabel("Flow [ml/s]")


def plot_all(var_dict):
    t = var_dict["t"]
    for var, values in var_dict.items():
        plt.figure()
        plt.plot(t, values)
        plt.ylabel(var)


def calc_summary(var_dict):
    P_ao = np.maximum(var_dict["P_ao"], var_dict["P_lv"])
    P_sys = np.max(P_ao)
    P_dia = np.min(P_ao)
    P_map = np.mean(P_ao)
    PP = P_sys - P_dia
    V_sys = np.max(var_dict["V_lv"])
    V_dia = np.min(var_dict["V_lv"])
    SV = V_sys - V_dia
    CO = ml_per_sec_to_L_per_min*SV/(var_dict["t"][-1] - var_dict["t"][0])
    stroke_work_1 = P_map*SV
    #stroke_work_int = np.trapz(var_dict["Q_lvao"]*var_dict["P_lv"], x=var_dict["t"])
    ret_dict = locals()
    del ret_dict["var_dict"]
    del ret_dict["P_ao"]
    return ret_dict


def test_set_pars():
    vewk3 = VaryingElastance()
    cool_pars = dict(E_maxs=0.5, E_max=2.9)
    vewk3.set_pars(E_maxs=0.5, E_max=2.9)


def solve_to_steady_state(model, n_cycles=5, n_eval_pts=100):
    vewk3 = model
    u0 = vewk3.calc_consistent_initial_values()
    t_span = (0, vewk3.T*n_cycles)
    t_eval = vewk3.T*np.linspace(n_cycles-1, n_cycles, n_eval_pts)
    sol = scipy.integrate.solve_ivp(vewk3.rhs, t_span, u0, dense_output=True, method="RK45",
            atol=1e-10, rtol=1e-9)
    u_eval = sol.sol(t_eval)
    _, all_vars = vewk3.calc_all(t_eval, u_eval)
    return all_vars, t_eval, sol

def solve(model, n_cycles=5, n_eval_pts=100):
    vewk3 = model
    u0 = vewk3.calc_consistent_initial_values()
    t_span = (0, vewk3.T*n_cycles)
    t_eval = vewk3.T*np.linspace(0, n_cycles, n_cycles*n_eval_pts)
    sol = scipy.integrate.solve_ivp(vewk3.rhs, t_span, u0, dense_output=True, method="RK45",
            atol=1e-10, rtol=1e-9)
    u_eval = sol.sol(t_eval)
    _, all_vars = vewk3.calc_all(t_eval, u_eval)
    return all_vars, t_eval, sol
