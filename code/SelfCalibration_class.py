# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:19:13 2023

@author: Valdemar
"""

from hmmlearn.hmm import CategoricalHMM, GaussianHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

def VI(xn, K, tol=0, max_iter=10):
    
    dim1, dim2 = xn.shape
    sigma = np.mean(xn)/4
    mu = np.mean(xn)
    
    alpha0 = np.ones(K)
    beta0 = np.ones(K)*10**(-(K-1))
    m0 = np.arange(K)*mu
    W0 = np.ones(K)*(1/sigma**2)
    v0 = np.ones(K)*.1
    
    Nkprev = np.zeros(K)
    Nk = np.zeros(K)
    rnk = np.array([np.random.dirichlet(np.arange(K)+1) for _ in range(dim1)])
    for step in range(max_iter):
        Nkprev = Nk
        Nk = np.sum(rnk,axis = 0)
        xbark = np.zeros(K)
        Sk = np.zeros(K)
        for k in range(K):
            xbark[k] = np.sum(rnk[:,k]*xn.reshape(1,-1))/Nk[k]
            Sk[k] = np.sum((rnk[:,k]*((xn-xbark[k])**2).reshape(1,-1)),axis=1)/Nk[k]
        alphak = alpha0 + Nk
        betak = beta0 + Nk
        mk = (beta0*m0 + Nk*xbark)/betak
        Wk =1/(1/W0 + Nk*Sk + (beta0*Nk*(xbark-m0)**2)/(beta0+Nk))
        vk = v0 + Nk
        Euklambdak = 1/betak + vk*Wk*(xn-mk)**2
        lnlambda = digamma(vk/2) + np.log(2) + np.log(Wk)
        lnpik = digamma(alphak) - digamma(np.sum(alphak))
        lambdak = np.exp(lnlambda)
        pik = np.exp(lnpik)
        rnk = pik*np.sqrt(lambdak)*np.exp(-0.5*Euklambdak)
        rnk = rnk/(np.sum(rnk,axis=1).reshape(-1,1))
        if np.abs(np.min(Nk) - np.min(Nkprev)) < tol:
            break
    sigma2s = (1/((Wk*vk))).T
    means = mk
    pis = pik/np.sum(pik)
    return means, sigma2s, pis

def gmm_pdf(x, means, sigma2s, pis):
    ret = 0
    for m,s,p in zip(means, sigma2s, pis):
        ret += (1/(np.sqrt(2*np.pi*s)))*np.exp(-((x-m)**2)/(2*s))*p
    return ret

class SelfCalibration:
    def __init__(self, training_file_paths, sensor_parameters_file_path, levels,samples_length = 100, hmm_iters = 10, hmm_tol = 1e-3, sample_until = -1):
        self.samples_length = samples_length
        self.sample_until = sample_until #To test the model
        self.levels = levels
        self.train, self.lengths, self.lp8_data, self.lp8_dT, self.lp8_I, self.lp8_p, self.true_sunrise = self.get_train_data(training_file_paths, samples_length)
        self.sensorParams = pd.read_csv(sensor_parameters_file_path, nrows=610, index_col=False,delimiter=";").to_numpy()
        zeros = [i for i in range(0,self.levels)]
        delta_T = [i for i in range(1,4)]

        statepairs = []
        for dT in delta_T:
            for z in zeros:
                statepairs.append(np.array([z, dT]))
        self.statepairs = np.array(statepairs)

        self.A = np.zeros((self.statepairs.shape[0], self.statepairs.shape[0]))
        self.B = np.zeros((self.statepairs.shape[0],self.levels))
        
        self.pi = np.abs(np.ones((self.statepairs.shape[0]))/(self.statepairs.shape[0]) + np.random.normal(loc =0, scale=.1, size=(self.statepairs.shape[0])))
        self.pi = self.pi/np.sum(self.pi)
        self.A = self.get_init_A()
        self.B = self.get_init_B()
        
        self.iters = hmm_iters
        self.tol = hmm_tol
        
    def remove_outlier_rows(self, x, thresh = 40):
        idx_to_remove = []
        
        sunrise = x[:,4]
        lp8 = x[:,7]
        
        Dsunrise = np.abs(sunrise - np.roll(sunrise, -1))
        Dlp8 = np.abs(lp8 - np.roll(lp8, -1))
        for i in range(Dsunrise.shape[0]-1):
            if Dsunrise[i] > thresh:
                x[i,4] = (x[i-1, 4] + x[i+1, 4])/2
            if Dlp8[i] > thresh:
                x[i,7] = (x[i-1,7] + x[i-2,7])/2
                x[i,10] = (x[i-1,10] + x[i+1,10])/2
                x[i,9] = (x[i-1,9] + x[i+1,9])/2
        return x
            
    def calibrate_outside(self, sunrise, lp8, time, ind = -1):
        if ind == 1:
            return sunrise, lp8
                
        outside_ppm = 800
        bias_sunrise = 0
        bias_lp8 = 0
        lp8_diff = []
        sunrise_diff = []
        new_bias = True
        
        for t in range(sunrise.shape[0]):
            curr_time = time[t][11:]
            curr_date = time[t][:11]
            if (int(curr_time[:2]) == 12 and (int(curr_time[3:5]) >= 28 and int(curr_time[3:5]) <= 40)):
                if new_bias:
                    new_bias = False
                    lp8_diff = []
                    sunrise_diff = []
                sunrise_diff.append(outside_ppm - sunrise[t])
                lp8_diff.append(outside_ppm - lp8[t])
                sunrise[t] = outside_ppm
                lp8[t] = outside_ppm
                continue
            
            if not(new_bias):
                bias_sunrise = np.mean(sunrise_diff)
                bias_lp8 = np.mean(lp8_diff)
                new_bias = True
            sunrise[t] += bias_sunrise
            lp8[t] += bias_lp8
                
        return sunrise, lp8
          
    def get_train_data(self, file_path, samples_length):
        recordings_lp8 = []
        recordings_sunrise = []
        true_readings_sunrise = []
        true_readings_lp8 = []
        reading_dT_lp8 = []
        reading_I_lp8 = []
        reading_p_lp8 = []
        lengths = []
        
        for i in range(5):
            reading = pd.read_csv("kth_logger_0" + str(i) +".csv").to_numpy()
            reading = reading[2500:self.sample_until]
            reading = self.remove_outlier_rows(reading)
            reading[:,7][:reading[:,7].shape[0]-14] = moving_average(reading[:,7], n=15)
            reading[:,9][:reading[:,9].shape[0]-14] = moving_average(reading[:,9], n=15)

            maxq = np.mean(reading[:,7]) + 2*np.sqrt(np.var(reading[:,7]))
            minq = np.mean(reading[:,7]) - 2*np.sqrt(np.var(reading[:,7]))
            quant_levels = [minq]
            step = (maxq - minq)/(self.levels - 1)
            for i in range(self.levels-2):
                quant_levels.append(step+quant_levels[-1])
            quant_levels.append(maxq)

            temp_lp8_quant = np.array([])
            temp_lp8 = np.array([])
            temp_true_sunrise = np.array([])
            temp_sunrise = np.array([])
            temp_I = np.array([])
            temp_dT = np.array([])
            temp_p = np.array([])
            temp_lengths = np.array([])
            j = 0
            sl = samples_length
            
            while j+sl < reading.shape[0]:
                temp_lp8 = np.append(temp_lp8, reading[:,7][j:j+sl])
                lp8_reading_q = quantize_timeseries(reading[:,7][j:j+sl],self.levels, quant_levels)
                temp_lp8_quant = np.append(temp_lp8_quant, lp8_reading_q)
                sunrise_reading = reading[:,4][j:j+sl]
                temp_true_sunrise=np.append(temp_true_sunrise,sunrise_reading)
                
                sunrise_reading_q = quantize_timeseries(sunrise_reading, self.levels, quant_levels)
                temp_sunrise=np.append(temp_sunrise,sunrise_reading_q)
                temp_dT=np.append(temp_dT,reading[:,10][j:j+sl])
                temp_I=np.append(temp_I,reading[:,9][j:j+sl])
                temp_p=np.append(temp_p,reading[:,3][j:j+sl])
                temp_lengths = np.append(temp_lengths,sl).astype("int")
                
                j += sl
                sl = int(np.random.choice(np.arange(samples_length/2, samples_length), 1)[0])
                
            true_readings_lp8.append(temp_lp8)    
            recordings_lp8.append(temp_lp8_quant)
            true_readings_sunrise.append(temp_true_sunrise)
            recordings_sunrise.append(temp_sunrise)
            reading_dT_lp8.append(temp_dT)
            reading_I_lp8.append(temp_I)
            reading_p_lp8.append(temp_p)
            lengths.append(temp_lengths)

        return recordings_lp8, lengths, true_readings_lp8, reading_dT_lp8, reading_I_lp8, reading_p_lp8, true_readings_sunrise
    
    def get_init_A(self):
        for curr_state in range(self.A.shape[0]):
            for next_state in range(self.A.shape[1]):
                curr_dT = self.statepairs[curr_state][1]
                curr_z = self.statepairs[curr_state][0]
                next_dT = self.statepairs[next_state][1]
                next_z = self.statepairs[next_state][0]
                
                if curr_dT == 1: #dT < 0
                    if next_z <= curr_z:
                        self.A[curr_state,next_state] = 50*np.exp(-np.abs(next_z-curr_z)**(2))
                        
                    else:
                        self.A[curr_state, next_state] = 1*np.exp(-np.abs(next_z-curr_z)**(2))
                        
                elif curr_dT == 2: #dT = 0
                    self.A[curr_state, next_state] = 50
                else: #dT > 0
                    if next_z >= curr_z:
                        self.A[curr_state,next_state] = 50*np.exp(-np.abs(next_z-curr_z)**(2))
                    else:
                        self.A[curr_state, next_state] = 1*np.exp(-np.abs(next_z-curr_z)**(2))
            self.A[curr_state,:] = self.A[curr_state,:]/np.sum(self.A[curr_state,:])
        return self.A
    
    def get_init_B(self):
        for state in range(self.B.shape[0]):
            for obs in range(self.B.shape[1]):
                z = self.statepairs[state][0]
                dT = self.statepairs[state][1]
                
                if z == 0:
                    if obs == 0:
                        self.B[state,obs] = .9
                    elif obs == z+1:
                        self.B[state,obs] = .1
                
                elif z == np.max(range(self.B.shape[0])):
                    if obs == np.max(range(self.B.shape[0])):
                        self.B[state,obs] = .9
                    elif obs == z-1:
                        self.B[state,obs] = .1
                
                else:
                    if obs == z:
                        self.B[state,obs] = .8
                    elif obs == z+1:
                        self.B[state,obs] = .1
                    elif obs == z-1:
                        self.B[state,obs] = .1
            self.B[state, :] = self.B[state, :]/np.sum(self.B[state, :])
        return self.B
    
    def fit(self):
        self.models = {}
        self.omegas = {}
        
        k=0
        sensor_names = ["S00","S01", "S02", "S03", "S04"]
        temp_measurements = np.array([])
        temp_measurements_no_quant = np.array([self.true_sunrise[0]])
        temp_dT = np.array([self.lp8_dT[0]])
        temp_I = np.array([self.lp8_I[0]])
        temp_p = np.array([self.lp8_p[0]])
        
        st = 0
        ed = self.lengths[0]
        entries = np.arange(17)*4096
        
        for i in range(len(self.lengths)):
            temp_measurements = self.train[i]
            temp_measurements_no_quant = self.true_sunrise[i]
            temp_dT = self.lp8_dT[i]
            temp_I = self.lp8_I[i]
            temp_p = self.lp8_p[i]
        
            m = int(sensor_names[i][2])
            Tz = self.sensorParams[m,1]
            Tz2 = self.sensorParams[m,2]
            Ts = self.sensorParams[m,3]
            Ts2 = self.sensorParams[m,4]
            factoryZero = self.sensorParams[m,5]
            S = self.sensorParams[m,6]
            T0 = 61440
            P = self.sensorParams[m,7:].tolist()
            temp_measurements = (temp_measurements).astype("int")
            
            model = CategoricalHMM(n_components=3*self.levels,n_features=self.levels,
                                  transmat_prior=self.A, emissionprob_prior=self.B, tol=self.tol, n_iter=self.iters, 
                                 init_params = "s", params = "ste", verbose=False, implementation="log")
            #model.startprob_ = self.pi
            model.transmat_ = self.A
            model.emissionprob_ = self.B
            model.fit(temp_measurements.reshape(-1, 1), lengths=self.lengths[i])
            for j in range(self.A.shape[0]):
                if np.sum(model.transmat_[j,:]) != 1: #Model did not converge properly, set values to initial states
                    model.transmat_ = self.A
                    model.emissionprob_ = self.B
                    model.startprob_ = self.pi
            
            self.models[sensor_names[i]] = model
            temp_measurements = np.array([])
            temp_lengths = []
            
            #Find range of zeros for this sensor
            for u in range(temp_measurements_no_quant.shape[0]):
                if temp_p[u] == " nan": 
                    continue
                curr_p = 10*temp_p[u]
                f = -2.7216549*(10**-2) + (6.6504658*(10**-5))*curr_p + (3.4417698*(10**-9))*(curr_p**2)
                temp_measurements_no_quant[u] = temp_measurements_no_quant[u]*f
                
            co2_to_table_entry = np.interp(temp_measurements_no_quant.astype("int"), np.flip(P), np.flip(entries))
            zerovals = Te_to_zero(co2_to_table_entry, Tz, Tz2, Ts, Ts2, S, T0, temp_dT, temp_I)
            
            #Quantize zeros uniformly within a high probable region
            z_high = min([np.mean(zerovals) + 2*np.sqrt(np.var(zerovals)), max(zerovals)])
            z_low = max([np.mean(zerovals) - 2*np.sqrt(np.var(zerovals)), min(zerovals)])
            
            omegas = [z_low]
            step = (z_high -z_low)/(self.levels - 1)
            for _ in range(self.levels - 2):
                omegas.append(omegas[-1] + step)
            omegas.append(z_high)
            self.omegas[sensor_names[i]] = (np.array(omegas))
            
            #Alternative way
            '''
            clusterer = KMeans(n_clusters=self.levels)
            clusterer.fit(zerovals.reshape(-1, 1))
            self.omegas[sensor_names[k]] = np.sort(clusterer.cluster_centers_)
            '''
            
            temp_measurements_no_quant = np.array([])
            temp_dT = np.array([])
            temp_I = np.array([])
            temp_p = np.array([])
            
            k+=1
            
        return self
    
    def supervised_fit(self):
        self.models = {}
        self.omegas = {}
        
        k=0
        sensor_names = ["S00","S01", "S02", "S03", "S04"]
        temp_measurements = np.array([])
        temp_measurements_no_quant = np.array([self.true_sunrise[0]])
        temp_dT = np.array([self.lp8_dT[0]])
        temp_I = np.array([self.lp8_I[0]])
        temp_p = np.array([self.lp8_p[0]])
        
        st = 0
        ed = self.lengths[0]
        entries = np.arange(17)*4096
        
        for i in range(len(self.lengths)):
            temp_measurements = self.train[i]
            temp_measurements_no_quant = self.true_sunrise[i]
            temp_dT = self.lp8_dT[i]
            temp_I = self.lp8_I[i]
            temp_p = self.lp8_p[i]
        
            m = int(sensor_names[i][2])
            Tz = self.sensorParams[m,1]
            Tz2 = self.sensorParams[m,2]
            Ts = self.sensorParams[m,3]
            Ts2 = self.sensorParams[m,4]
            factoryZero = self.sensorParams[m,5]
            S = self.sensorParams[m,6]
            T0 = 61440
            P = self.sensorParams[m,7:].tolist()
            temp_measurements = (temp_measurements).astype("int")
            
            #Find range of zeros for this sensor
            for u in range(temp_measurements_no_quant.shape[0]):
                if temp_p[u] == " nan": 
                    continue
                curr_p = 10*temp_p[u]
                f = -2.7216549*(10**-2) + (6.6504658*(10**-5))*curr_p + (3.4417698*(10**-9))*(curr_p**2)
                temp_measurements_no_quant[u] = temp_measurements_no_quant[u]*f
                
            co2_to_table_entry = np.interp(temp_measurements_no_quant.astype("int"), np.flip(P), np.flip(entries))
            zerovals = Te_to_zero(co2_to_table_entry, Tz, Tz2, Ts, Ts2, S, T0, temp_dT, temp_I)
            print(zerovals.shape)
            #Quantize zeros uniformly within a high probable region
            z_high = min([np.mean(zerovals) + 2*np.sqrt(np.var(zerovals)), max(zerovals)])
            z_low = max([np.mean(zerovals) - 2*np.sqrt(np.var(zerovals)), min(zerovals)])
            
            omegas = [z_low]
            step = (z_high -z_low)/(self.levels - 1)
            for _ in range(self.levels - 2):
                omegas.append(omegas[-1] + step)
            omegas.append(z_high)
            self.omegas[sensor_names[i]] = (np.array(omegas))
            
            dTs = temp_dT - np.roll(temp_dT, 1)
            stateseq = []
            for dT, Z in zip(dTs, zerovals):
                if dT > -.5 and dT < .5:
                    state1 = 2
                elif dT >= .2:
                    state1 = 3
                else:
                    state1 = 1
                
                #quant zero
                state2 = np.argmin(np.abs(self.omegas[sensor_names[i]] - Z))
                        
                for j,pair in enumerate(self.statepairs):
                    if pair[0] == state2 and pair[1] == state1:
                        stateseq.append(j)
            
            stateseq = np.array(stateseq)
            A = np.zeros((self.statepairs.shape[0], self.statepairs.shape[0]))
            for j in range(stateseq.shape[0]):
                if j == stateseq.shape[0] - 1:
                    break
                A[stateseq[j], stateseq[j+1]] += 1
                
            B = np.zeros((self.statepairs.shape[0],self.levels))
            for state, obs in zip(stateseq, temp_measurements):
                B[state, obs] += 1
            
            startprob = np.zeros(self.statepairs.shape[0])
            for state in stateseq:
                startprob[stateseq] += 1
            startprob /= np.sum(startprob)

            for j in range(self.statepairs.shape[0]):
                if np.sum(A[j,:]) > 0:
                    A[j,:] = A[j,:]/np.sum(A[j,:])
                else:
                    A[j] = np.ones(self.statepairs.shape[0])/self.statepairs.shape[0]
                if np.sum(B[j,:]) > 0:
                    B[j,:] = B[j,:]/np.sum(B[j,:])
                else:
                    B[j] = np.ones(self.levels)/self.levels
            
            model = CategoricalHMM(n_components=3*self.levels,n_features=self.levels)
            model.transmat_ = A
            model.emissionprob_ = B
            model.startprob_ = startprob
            self.models[sensor_names[i]] = model
            
            temp_measurements_no_quant = np.array([])
            temp_dT = np.array([])
            temp_I = np.array([])
            temp_p = np.array([])
            temp_measurements = np.array([])
            temp_lengths = []
            
            k+=1
            
        return self
    
    def predict(self, lp8_measurements, lp8_dT, lp8_I, lp8_p, sensorType = "S00"):
        maxq = np.mean(lp8_measurements) + 2*np.sqrt(np.var(lp8_measurements))
        minq = np.mean(lp8_measurements) - 2*np.sqrt(np.var(lp8_measurements))
        quant_levels = [minq]
        step = (maxq - minq)/(self.levels - 1)
        for i in range(self.levels-2):
            quant_levels.append(step+quant_levels[-1])
        quant_levels.append(maxq)
        lp8_measurements = quantize_timeseries(lp8_measurements, self.levels,quant_levels)
        k = int(sensorType[2])
        model = self.models[sensorType]
        omegas = self.omegas[sensorType]
        Tz = self.sensorParams[k,1]
        Tz2 = self.sensorParams[k,2]
        Ts = self.sensorParams[k,3]
        Ts2 = self.sensorParams[k,4]
        factoryZero = self.sensorParams[k,5]
        S = self.sensorParams[k,6]
        T0 = 61440
        P = self.sensorParams[k,7:].tolist()
        entries = np.arange(17)*4096
        p=[]
        Te = []
        zeros = predict_zeros(model,lp8_measurements.reshape(-1,1), omegas, self.statepairs)
        for j,z in enumerate(zeros):
            if lp8_I[j] is None or lp8_dT[j] is None:
                continue
            zero = z
            Te.append(table_entry(zero, T0, lp8_I[j], Tz, Tz2, lp8_dT[j], S, Ts, Ts2))
            p.append(lp8_p[j])
            
        measurements_calibrated = preasure_compensation(np.interp(Te,entries,P),p)
        return measurements_calibrated

def quantize_timeseries(timeseries, n_levels, levels):
    ret = []

    for t,yt in enumerate(timeseries):
        ytquant = 0
        for i,level in enumerate(levels):
            #ytquant = (level[0]+level[1])/2
            ytquant = i
            if i == len(levels) -1:
                break
            elif yt >= levels[i] and yt < levels[i+1]:
                break
        ret.append(ytquant)
    return np.array(ret)

def predict_zeros(model,obs, omegas,statepairs):
    ret = []
    lp, state_seq = model.decode(obs)
    #state_seq = model.predict(obs)
    for state in state_seq:
        ret.append(omegas[statepairs[state][0]])
    return np.array(ret)

def Te_to_zero(Te, Tz, Tz2, Ts, Ts2, S, T0, dT, I):
    I*=(2**-8)
    Tz*=(2**-24)
    Tz2*=(2**-37)
    S*=(2**-12)
    Ts*=(2**-24)
    Ts2*=(2**-37)
    
    num = T0*((dT**2)*S*Ts2 + dT*S*Ts + S - 1) + Te
    den = I*S*(Tz*dT + (dT**2)*Tz2 +1)*((dT**2)*Ts2 + dT*Ts + 1)
    
    return (2**13)*num/den

def table_entry(Z, T0, I, Tz, Tz2, dT, S, Ts, Ts2):
    Z*=(2**-13)
    I*=(2**-8)
    Tz*=(2**-24)
    Tz2*=(2**-37)
    S*=(2**-12)
    Ts*=(2**-24)
    Ts2*=(2**-37)
    ret = T0 - (T0-I*Z*(1+Tz*dT+Tz2*(dT**(2))))*S*(1+Ts*dT+Ts2*(dT**2))
    return ret

def preasure_compensation(c, p):
    co2s = np.array([])
    for i in range(len(p)):
        if p[i] == ' nan':
            f=1
        else:
            curr_p = 10*p[i]
            f = -2.7216549*(10**-2) + (6.6504658*(10**-5))*curr_p + (3.4417698*(10**-9))*(curr_p**2)
        co2s = np.append(co2s, c[i]/f)
    return co2s

def CO2(Te,p, table):
    return preasure_compensation(table[Te],p)

def moving_average(x, n=4):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
'''
### EXAMPLE CODE ###
def main():
    training_paths = ["kth_logger_00.csv", "kth_logger_01.csv", "kth_logger_02.csv", "kth_logger_03.csv", "kth_logger_04.csv"]
    parameters_path = "SensorsParams.csv"
    samples_length = 130
    
    calibrator = SelfCalibration(training_paths, parameters_path,5,
                                samples_length = samples_length,
                                hmm_iters = 50, hmm_tol = 1e-1,
                                sample_until = -1)
    calibrator.supervised_fit()
    import matplotlib.animation as animation
    from IPython import display 
    samples_length = 200
    for i in range(5):
        test_file = pd.read_csv("kth_logger_0"+ str(i) +".csv").to_numpy()
        test_file = calibrator.remove_outlier_rows(test_file)
        
        co2_values = np.array([])
        true_co2_values = np.array([])
        lp8_co2_values = np.array([])
        N = test_file[:,10].shape[0]
        t = np.array([])
        time = np.array([])
        date = np.array([])
        fig, ax = plt.subplots()
        for j in range(0, N-samples_length, samples_length):
            dT = (test_file[:,10][j:(j+samples_length)])
            I = (test_file[:,9][j:(j+samples_length)])
            p = (test_file[:,3][j:(j+samples_length)])
            measurements = test_file[:,7][j:(j+samples_length)]
            true = test_file[:,4][j:(j+samples_length)]
            calibrated = calibrator.predict(measurements, dT, I, p, sensorType = "S0"+ str(i))
                
            co2_values = np.append(co2_values, (calibrated))
            true_co2_values = np.append(true_co2_values, true)
            lp8_co2_values = np.append(lp8_co2_values, measurements)
            date_time = test_file[:,0][j:(j+samples_length)]
            curr_time = np.array([])
            curr_date = np.array([])
            for k in range(date_time.shape[0]):
                curr_time = np.append(curr_time,date_time[k][11:])
                curr_date = np.append(curr_date, date_time[k][:11])
            time = np.append(time, curr_time)
            date = np.append(date, curr_date)
            t = np.append(t, np.arange(j, j+samples_length))
        
        calibrated = ax.plot(t[0:samples_length], co2_values[0:samples_length], label='Calibrated CO2 values', c='r')[0]
        sunrise = ax.plot(t[0:samples_length], true_co2_values[0:samples_length], label='Sunrise CO2 values', c='b')[0]
        lp8 = ax.plot(t[0:samples_length], lp8_co2_values[0:samples_length], label='LP8 CO2 values', c='g')[0]
        ax.legend()
        ax.set_xticklabels(time[0:samples_length])
        ax.set_title("S0"+ str(i) + " " + date[0*samples_length])
        ax.set_xlabel("Time [Hour:Minute:Second]")
        ax.set_ylabel("CO2 concentration (ppm)")
        
        def update(frame):   
            ax.clear()
            calibrated = ax.plot(t[(frame)*samples_length:(frame+1)*samples_length], co2_values[(frame)*samples_length:(frame+1)*samples_length], 
                                 label='Calibrated CO2 values', c='r')[0]
            sunrise = ax.plot(t[(frame)*samples_length:(frame+1)*samples_length], true_co2_values[(frame)*samples_length:(frame+1)*samples_length], 
                              label='Sunrise CO2 values', c='b')[0]
            lp8 = ax.plot(t[(frame)*samples_length:(frame+1)*samples_length], lp8_co2_values[(frame)*samples_length:(frame+1)*samples_length], 
                          label='LP8 CO2 values', c='g')[0]
            tick_vals = []
            tick_labels = []
            step = 1/5
            for k in range(6):
                tick_vals.append(int((frame+step*k)*(samples_length)))
                tick_labels.append(time[tick_vals[-1]])
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(tick_labels)
            ax.set_title("S0"+ str(i) + " " + date[frame*samples_length])
            ax.legend()
            ax.set_xlabel("Time [Hour:Minute:Second]")
            ax.set_ylabel("CO2 concentration (ppm)")
            return (calibrated, sunrise, lp8)
        if i == 1:
            anim = animation.FuncAnimation(fig=fig, func=update, frames=3, interval=10,blit=False, repeat = False, cache_frame_data=False)
        else:
            anim = animation.FuncAnimation(fig=fig, func=update, frames=50, interval=10,blit=False, repeat = False, cache_frame_data=False)
        writervideo = animation.FFMpegWriter(fps=2) 
        anim.save('Sensor S0' + str(i)+'.mp4', writer=writervideo) 
        plt.close()
        plt.plot(t[150:], true_co2_values[150:])
        plt.plot(t[150:], lp8_co2_values[150:])
        plt.show()
'''
        
if __name__ == "__main__":
    main()
    
