# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:33:51 2021

@author: Yongrong Qiu
"""

import numpy as np
from sklearn.decomposition import randomized_svd
import h5py
from scipy import interpolate
from scipy import signal
from scipy import stats
import cmath

import matplotlib.pyplot as plt

def load_h5_data(file_name):
    """
    Helper function to load h5 file.
    """
    with h5py.File(file_name,'r') as f:
        return {key:f[key][:] for key in list(f.keys())}

def filterSignal(x, Fs, high):
    """
    Filter raw signal
    y = filterSignal(x, Fs, high) filters the signal x. Fs is the sampling frequency in Hz.
    The filter delay is compensated in the output y.
    """
    b, a = butter_highpass(high, Fs, order=5)#you can change different order, 4 here as Quian Quiroga 2004
    y = signal.filtfilt(b, a, x)
    return y
def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs #Nyquist frequency
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='high')
    return b, a

class Chirp():
    """
    Preprocess chirp responses: high pass 0.1 Hz (Baden et al., 2016.) ,algin and normalize
    Input:
        #chirp_stim: stimulus waveform of chirp, here stimulus frequency is 100 Hz (the frequency for
        #            the stimulus on the setup is 60 Hz)
        #chirp_stim_time: time base for stimulus waveform, unit: second
        traces_raw: responses, 5 repeats,
        tracetime: time base for responses, raw recording frequency is 7.8125 Hz, unit: second
        triggertime: for each repeat, two triggers, so in total 10 trigger times here. For each repeat, triggers
                    are placed at the start and after 5 seconds.
    Output:
        after preprocessing, all traces with a frequency of 100 Hz, aligned to the chirp
        self.traces: 5 repeats
        self.trace: mean of 5 repeats
        self.qi: quality index based on chirp reponses
        self.ooi: on off index
        self.tsi: transient sustained index
        
    """
    def __init__(self, traces_raw, tracetime, triggertime):
        super().__init__()
        #self.chirp_stim      = chirp_stim
        #self.chirp_stim_time = chirp_stim_time
        self.traces_raw      = traces_raw
        self.tracetime       = tracetime
        self.triggertime     = triggertime
        #after preprocessing
        self.traces          = np.zeros((5,249)) # 249 is the #sample in Baden et al 2016
        self.trace           = np.zeros(249) 
        #self.trace_std       = np.zeros_like(self.chirp_stim) 
        self.qi              = None 
        self.ooi             = None 
        self.tsi             = None 
    
    def _highpass(self,y):
        """
        high pass 0.1 Hz (Baden et al., 2016.) for raw trace (in 7.8125 Hz)
        y: response trace
        """
        #return filterSignal(y, Fs=7.8125, high=0.1)
        return signal.savgol_filter(y, window_length=5, polyorder=3)
        #return y
   
    def _baseline(self,y):
        """
        median of 5-second trace before stimulus onset as the baseline, which will be subtracted
        """
        return np.median(y[np.where((self.tracetime<self.triggertime[0])\
                                   &(self.tracetime>=self.triggertime[0]-5))])
    
    def _norm(self,y,baseline):
        """
        normalize the trace: subtract the baseline, divided by median of stimulus responses
        """
        #y=y[np.where((self.tracetime>=self.triggertime[0])\
        #            &(self.tracetime<self.triggertime[-1]+27))]
        y=y-baseline
        return y/np.abs(np.median(y[np.where((self.tracetime>=self.triggertime[0])\
                                      &(self.tracetime<self.triggertime[-1]+27))])) # np.abs is very important here
    
    def _align(self,y):
        """
        align the trace to upsampling triggertime
        To match Baden et al 2016, for each repeat, we only use 251 data points, and then resample to 249 data points
        """
        temp = np.zeros_like(self.traces)
        triggertime = np.append(self.triggertime, self.triggertime[-1]+27)
        for ii in range(self.traces.shape[0]):
            yy = y[np.where((self.tracetime>=triggertime[ii*2]) & \
                              (self.tracetime<triggertime[ii*2+2]))[0][:251]]
            #yy = signal.savgol_filter(yy, window_length=5, polyorder=3)
            xx = np.linspace(0, 32, yy.shape[0])
            ff = interpolate.interp1d(xx, yy)
            xx_new = np.linspace(0, 32, self.traces.shape[1])   
            yy_new = ff(xx_new)
            # subtract mean of first 8 frames and normalize
            yy_new -= np.min(yy_new)
            yy_new /= np.max(np.abs(yy_new))
            temp[ii] = yy_new
        return temp.flatten()

    def _prepro(self):
        y = self._highpass(self.traces_raw)
        #baseline = self._baseline(y)
        #y = self._norm(y,baseline)
        y = self._align(y)
        # subtract mean of first 8 frames and normalize
        #y -= np.min(y)
        #y /= np.max(np.abs(y))
        return y
    
    def get_trace(self):
        y = self._prepro()
        for ii in range(len(self.traces)):
            self.traces[ii] = y[len(self.trace)*ii:len(self.trace)*(ii+1)]
        self.trace = np.mean(self.traces,axis=0)
        #self.trace_std = np.std(self.traces,axis=0,ddof=1)
        return None
    
class MovingBar():
    """
    Preprocess moving bar responses: high pass 0.1 Hz (Baden et al., 2016.) ,algin and normalize
    Input:
        traces_raw: responses, 3 repeats,
        tracetime: time base for responses, raw recording frequency is 7.8125 Hz, unit: second
        triggertime: One trigger is placed at the start of each trial. 8 directions for 3 repeats, so in total
                     24 trials, each trial lasts 4 seconds.
    Output:
        after preprocessing, all traces with a frequency of 8 Hz
        self.traces: 3 repeats, shape (3 repeats x 8 directions x 32 time samples)
        self.trace: mean of 3 repeats, shape (8 directions x 32 time samples)
        self.qi: quality index based on MB reponses
        self.ooi: on off index based on MB reponses
        self.tsi: transient sustained index
        self.dsi: Direction selectivity index
        self.osi: Orientation selectivity index
        self.dp: Direction tuning p-value
        self.op:  Orientation tuning p-value
        
    """
    def __init__(self, traces_raw, tracetime, triggertime):
        super().__init__()
        self.traces_raw      = traces_raw
        self.tracetime       = tracetime
        self.triggertime     = triggertime
        #
        self.frequency       = 8 # unit: Hz
        #after preprocessing
        self.traces          = np.zeros((3,8,32)) 
        self.trace           = np.zeros((8,32)) 
        self.dsi             = None 
        self.dp              = None 
    
    def _highpass(self,y):
        """
        high pass 0.1 Hz (Baden et al., 2016.) for raw trace (in 7.8125 Hz)
        y: response trace
        """
        #return filterSignal(y, Fs=7.8125, high=0.1)
        return signal.savgol_filter(y, window_length=5, polyorder=3)
   
    def _baseline(self,y):
        """
        median of 5-second trace before stimulus onset as the baseline, which will be subtracted
        """
        return np.median(y[np.where((self.tracetime<self.triggertime[0])\
                                   &(self.tracetime>=self.triggertime[0]-5))])
    
    def _norm(self,y,baseline):
        """
        normalize the trace: subtract the baseline, divided by median of stimulus responses
        """
        #y=y[np.where((self.tracetime>=self.triggertime[0])\
        #            &(self.tracetime<self.triggertime[-1]+27))]
        y=y-baseline
        return y/np.abs(np.median(y[np.where((self.tracetime>=self.triggertime[0])\
                                      &(self.tracetime<self.triggertime[-1]+4))])) # np.abs is very important here
    
    def _align(self,y):
        """
        align the trace to upsampling triggertime
        """
        triggertime = np.append(self.triggertime, self.triggertime[-1]+4)
        nums = [self.traces.shape[2]]*self.traces.shape[1]*self.traces.shape[0]
        for ii in range(len(triggertime)-1):
            temp=np.linspace(triggertime[ii],triggertime[ii+1],nums[ii],endpoint=False) #upsample to self.frequency (100 Hz)
            if ii==0:
                new_triggertime=np.copy(temp)
            else:
                new_triggertime=np.append(new_triggertime,temp)
        #xx = self.tracetime[np.where((self.tracetime>=self.triggertime[0])\
        #                            &(self.tracetime<self.triggertime[-1]+27))]
        xx = np.copy(self.tracetime)
        yy = np.copy(y)
        ff = interpolate.interp1d(xx, yy)
        xx_new = np.copy(new_triggertime)   
        yy_new = ff(xx_new)
        return yy_new
    
    def _prepro(self):
        y = self._highpass(self.traces_raw)
        #baseline = self._baseline(y)
        #y = self._norm(y,baseline)
        y = self._align(y)
        # subtract mean of first 8 frames and normalize
        #y -= np.min(y)
        #y /= np.max(np.abs(y))
        # something changed in the stimulus file such that the onset and offset of bar movement happens
        # at a different time relative to the trigger ... shift 5 points
        y = np.roll(y,-5) 
        y = np.reshape(y,(self.traces.shape[0],self.traces.shape[1],self.traces.shape[2]))
        # subtract mean of first 8 frames and normalize
        y -= np.min(y, axis=(1,2), keepdims=True)
        y /= np.max(np.abs(y), axis=(1,2), keepdims=True)
        return y
    
    def get_trace(self):
        y = self._prepro()
        self.traces = np.reshape(y,(self.traces.shape[0],self.traces.shape[1],self.traces.shape[2]))
        self.trace = np.mean(self.traces,axis=0)
        #self.trace_std = np.std(self.traces,axis=0,ddof=1)
        return None



    
class OsDsIndexes():
    """
    #This class computes the direction and orientation selectivity indexes 
    #as well as a quality index of DS responses as described in Baden et al. (2016)
    -> Stimulus
    -> DetrendSnippets
    
    DetrendSnippets:  3d array, shape: time x directions x repetitions (32,8,3)
    dir_deg:          the directions of the bars in degree, [0,180, 45,225, 90,270, 135,315]
    ---
    ds_index:   float   #direction selectivity index as resulting vector length (absolute of projection on complex exponential)
    ds_pvalue:  float   #p-value indicating the percentile of the vector length in null distribution
    ds_null:    longblob    #null distribution of DSIs
    pref_dir:  float    #preferred direction
    os_index:   float   #orientation selectivity index in analogy to ds_index
    os_pvalue:  float   #analogous to ds_pvalue for orientation tuning
    os_null:    longblob    #null distribution of OSIs
    pref_or:    float   #preferred orientation
    on_off:     float   #on off index based on time kernel
    d_qi:       float   #quality index for moving bar response
    u:     longblob    #time component
    v:     longblob    #direction component
    surrogate_v:    longblob    #computed by projecting on time
    surrogate_dsi:  float   #DSI of surrogate v 
    avg_sorted_resp:    longblob    # response matrix, averaged across reps
    """
    def __init__(self, DetrendSnippets, dir_deg = [0,180, 45,225, 90,270, 135,315]):
        super().__init__()
        self.snippets        = DetrendSnippets
        self.dir_deg         = np.array(dir_deg)
        self.dir_rad         = np.deg2rad(self.dir_deg)   # convert to radians
        #
        self.ds_index = None
        self.ds_pvalue = None
        self.ds_null = None
        self.pref_dir = None
        
        self.os_index = None
        self.os_pvalue = None
        self.os_null = None
        self.pref_or = None
        
        self.sorted_dir_rad = None
        self.d_qi = None
        self.u = None
        self.v = None
        self.surrogate_v = None
        self.surrogate_dsi = None
        self.avg_sorted_resp = None
        self.sorted_resp = None
        self.on_off = None
        self.resp_speed = None
                                          
    def OsDs(self):
        #dir_idx = [list(np.nonzero(dir_order == d)[0]) for d in dir_deg]
        sorted_responses, sorted_directions = self.sort_response_matrix(self.snippets, self.dir_rad)
        avg_sorted_responses = np.mean(sorted_responses, axis=-1)
        u, v, s = self.get_time_dir_kernels(avg_sorted_responses)
        dsi, pref_dir = self.get_si(v, sorted_directions, 1)
        osi, pref_or = self.get_si(v, sorted_directions, 2)
        
        (t, d, r) = sorted_responses.shape
        temp = np.reshape(sorted_responses, (t, d*r))
        projected = np.dot(np.transpose(temp), u)  # we do this whole projection thing to make the result
        projected = np.reshape(projected, (d, r))  #  between the original and the shuffled comparable
        surrogate_v = np.mean(projected, axis = -1)
        #surrogate_v -= np.min(surrogate_v) # updated by yqiu, comment this
        surrogate_v /= np.max(abs(surrogate_v)) # np.max(surrogate_v)
        #surrogate_v = np.copy(v)
        
        dsi_s, pref_dir_s = self.get_si(surrogate_v,
                                                sorted_directions,
                                                1)
        osi_s, pref_or_s = self.get_si(surrogate_v,
                                               sorted_directions,
                                               2)
        null_dist_dsi, p_dsi = self.test_tuning(np.transpose(projected),
                                                 sorted_directions,
                                                 1)
        null_dist_osi, p_osi = self.test_tuning(np.transpose(projected),
                                                 sorted_directions,
                                                 2)
        d_qi = self.quality_index_ds(sorted_responses)
        on_off, response_speed = self.get_on_off_response_speed(u)
        
        self.ds_index = dsi
        self.ds_pvalue = p_dsi
        self.ds_null = null_dist_dsi
        self.pref_dir = pref_dir
        
        self.os_index = osi
        self.os_pvalue = p_osi
        self.os_null = null_dist_osi
        self.pref_or = pref_or
        
        self.sorted_dir_rad = sorted_directions
        self.d_qi = d_qi
        self.u = u
        self.v = v
        self.surrogate_v = surrogate_v # very similar as self.v
        self.surrogate_dsi = dsi_s
        self.avg_sorted_resp = avg_sorted_responses
        self.sorted_resp = sorted_responses
        self.on_off = on_off
        self.resp_speed = response_speed
        return None

    def quality_index_ds(self, raw_sorted_resp_mat):
        """
        This function computes the quality index for responses to moving bar as described in
        Baden et al 2016. QI is computed for each direction separately and the best QI is taken
        Inputs:
        raw_sorted_resp_mat:    3d array (time x directions x reps per direction)
        Output:
        qi: float               quality index
        """

        n_dirs = raw_sorted_resp_mat.shape[1]
        qis = []
        for d in range(n_dirs):
            numerator = np.var(np.mean(raw_sorted_resp_mat[:, d, :], axis=-1), axis=0)
            denom = np.mean(np.var(raw_sorted_resp_mat[:, d, :], axis=0), axis=-1)
            qis.append(numerator / denom)
        return np.max(qis)

    def sort_response_matrix(self, snippets, directions):
        """
        Sorts the snippets according to stimulus condition and repetition into a time x direction x repetition matrix
        Inputs:
        snippets    list or array, time x (directions*repetitions)
        idxs        list of lists giving idxs into last axis of snippets. idxs[0] gives the indexes of rows in snippets
                    which are responses to the direction directions[0]
        Outputs:
        sorted_responses   array, time x direction x repetitions, with directions sorted(!) (0, 45, 90, ..., 315) degrees
        sorted_directions   array, sorted directions
        """
        #structured_responses = snippets[:, idxs]
        sorting = np.argsort(directions)
        sorted_responses = snippets[:, sorting, :]
        sorted_directions = directions[sorting]
        return sorted_responses, sorted_directions

    def get_time_dir_kernels(self, sorted_responses):
        """
        Performs singular value decomposition on the time x direction matrix (averaged across repetitions)
        Uses a heuristic to try to determine whether a sign flip occurred during svd
        For the time course, the mean of the first second is subtracted and then the vector is divided by the maximum
        absolute value.
        For the direction/orientation tuning curve, the vector is normalized to the range (0,1)
        Input:
        sorted_responses:   array, time x direction
        Outputs:
        time_kernel     array, time x 1 (time component, 1st component of U)
        direction_tuning    array, directions x 1 (direction tuning, 1st component of V)
        singular_value  float, 1st singular value
        """

        U, S, V = np.linalg.svd(sorted_responses)
        u = U[:, 0]
        s = S[0]
        v = V[0, :]
        # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
        # correlation is negative, U is likely flipped
        r, _ = stats.spearmanr(u, np.mean(sorted_responses, axis=-1), axis=1)
        su = np.sign(r)
        if su == 0:
            su = 1
        sv = np.sign(np.mean(np.sign(V[0, :])))
        if sv == 1 and su == 1:
            s = 1
        elif sv == -1 and su == -1:
            s = -1
        elif sv == 1 and su == -1:
            s = 1
        elif sv == 0:
            s = su
        else:
            s = 1

        u = s*u
        #determine which entries correspond to the first second, assuming 4 seconds presentation time
        idx = int(len(u)/4)
        u -= np.mean(u[:idx])
        u = u/np.max(abs(u))
        v = s*v
        #v -= np.min(v) # updated by yqiu, comment this
        v /= np.max(abs(v))

        return u, v, s

    def get_si(self, v, dirs, per):
        """
        Computes direction/orientation selectivity index and preferred direction/orientation of a cell by projecting the tuning curve v on a
        complex exponential of the according directions dirs (as in Baden et al. 2016)
        Inputs:
        v:  array, dirs x 1, tuning curve as returned by SVD
        dirs:   array, dirs x 1, directions in radians
        per:    int (1 or 2), indicating whether direction (1) or orientation (2) shall be tested
        Output:
        index:  float, D/O si
        direction:  float, preferred D/O
        """
        bin_spacing = np.diff(per*dirs)[0]
        correction_factor = bin_spacing / (2 * (np.sin(bin_spacing / 2)))  # Zar 1999, Equation 26.16
        complExp = [np.exp(per * 1j * d) for d in dirs]
        vector = np.dot(complExp, v)
        index = correction_factor * np.abs(vector)/np.sum(v)  # get the absolute of the vector, normalize to make it range between 0 and 1
        #index = correction_factor * np.abs(vector) # updated by yqiu, no normalization
        direction = cmath.phase(vector)/per
        #for orientation, the directions are mapped to the right half of a circle. Map instead to upper half
        if per == 2 and direction < 0:
            direction+=np.pi
        return index, direction

    def test_tuning(self, rep_dir_resps, dirs, per, iters=1000):
        """
        """
        (rep_n, dir_n) = rep_dir_resps.shape
        flattened = np.reshape(rep_dir_resps, (rep_n*dir_n))
        rand_idx = np.linspace(0, rep_n*dir_n-1, rep_n*dir_n, dtype=int)
        null_dist = np.zeros(iters)
        complExp = [np.exp(per * 1j * d) for d in dirs] / np.sqrt(len(dirs))
        q = np.abs(np.dot(complExp, rep_dir_resps.mean(axis=0)))
        for i in range(iters):
            np.random.seed(i+1600)
            np.random.shuffle(rand_idx)
            shuffled = flattened[rand_idx]
            shuffled = np.reshape(shuffled, (rep_n, dir_n))
            shuffled_mean = np.mean(shuffled, axis=0)
            null_dist[i] = np.abs(np.dot(complExp, shuffled_mean))

        return null_dist, np.mean(null_dist > q)
    
    def plot_MB(self, figsize, figname, saveflag=False):
        """
        figsize: something like (6,6)
        figname: save figure name, string
        """
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(3, 3, 5, projection='polar', frameon=False) 
        temp = np.max(np.append(self.v, self.ds_index))
        ax.plot((0,np.pi),(temp*1.2,temp*1.2), color ='gray')
        ax.plot((np.pi/2,np.pi/2*3),(temp*1.2,temp*1.2), color ='gray')
        #ax.plot([0, self.pref_dir], [0, self.ds_index], color = 'r')
        ax.plot([0, self.pref_dir], [0, self.ds_index*np.sum(self.v)], color = 'r')
        #ax.plot([0, self.pref_or], [0, self.os_index], color = 'g')
        ax.plot(np.append(self.sorted_dir_rad, self.sorted_dir_rad[0]), np.append(self.v, self.v[0]), color = 'k')
        ax.set_rmin(0)
        #ax.set_rlim([0,5])
        ax.set_thetalim([0, 2*np.pi])
        ax.set_yticks([])  # Less radial ticks
        ax.set_xticks([])
        #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        #ax.grid(True)
        #
        ax_inds = [1,2,3,4,6,7,8,9] # start from 1, not 0
        dir_inds = [3,2,1,4,0,5,6,7]
        vmin, vmax = self.sorted_resp.min(), self.sorted_resp.max()
        for ii in range(len(ax_inds)):
            ax = plt.subplot(3, 3, ax_inds[ii], frameon=False)
            temp = self.sorted_resp[:,dir_inds[ii],:]
            for jj in range(3):
                ax.plot(temp[:,jj],color='gray')
            ax.plot(temp.mean(axis=-1),color='k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([vmin-vmax*0.2,vmax*1.2])
            ax.set_xlim([-len(temp)*0.2, len(temp)*1.2])
        if saveflag == True:
             fig.savefig(figname)
        return None
    
    def get_on_off_response_speed(self, time_kernel):
        """
        Not really working yet.
        Computes a preliminary On-Off Index based on the responses to the On (leading edge of moving bar, first 500 ms
        of ON response) and the OFF (trailing edge of moving bar, first 500 ms of OFF response) 
        Based on On-Off index, calculate response-speed index
        :param avg_response_matrix: average response matrix (average across repetitions per condition), time x directions
        :return:
        """
        normed_kernel = time_kernel - np.min(time_kernel)
        normed_kernel = normed_kernel/np.max(normed_kernel)
        on_response = normed_kernel[8:12]
        off_response = normed_kernel[16:20]
        #calculate the increase in on
        #if np.argmax(on_response)>0:
        #    on = np.max(on_response)-on_response[0]
        #else:
        #    on = np.min(on_response)-on_response[0]
        #calculate the decrease in off
        #if np.argmax(off_response)>0:
        #    off = np.max(off_response)-off_response[0]
        #else:
        #    off = np.min(off_response)-off_response[0]
        #on_off = (on - off)/(np.abs(on)+np.abs(off))
        on_off = (np.max(on_response) - np.max(off_response))/(np.max(on_response) + np.max(off_response))
        on_off = np.round(on_off, 2)
        if np.isnan(on_off):
            on_off = 0
        if on_off>=0: # on cell
            if np.argmax(on_response)>0: # not suppressed
                response_speed = (np.max(on_response) - on_response[0])/(np.argmax(on_response)) * 8 # response frequency, 8 Hz
            else:
                response_speed = (on_response[0] - np.min(on_response))/(np.argmin(on_response)) * 8
        elif on_off<0: # off cell
            if np.argmax(off_response)>0: # not suppressed
                response_speed = (np.max(off_response) - off_response[0])/(np.argmax(off_response)) * 8 # response frequency, 8 Hz
            else:
                response_speed = (off_response[0] - np.min(off_response))/(np.argmin(off_response)) * 8
        return on_off, response_speed
    


def bootstrap(statistics,data,num_exp=10000,seed=66):
    """
    bootstrapping
    apply bootstrapping to estimate standard deviation (error)
    statistics can be offratios, median, mean
    for offratios, be careful with the threshold
    data: for statistics offratios, median, mean: numpy array with shape (sample_size,1)
    num_exp: number of experiments, with replacement
    """
    if   statistics == 'offratios':
        def func(x): return len(x[np.where(x<0)])/len(x[np.where(x>0)]) #threshold is 0, may be different
    elif statistics == 'median':
        def func(x): return np.median(x)
    elif statistics == 'mean':
        def func(x): return np.mean(x)
    #
    sta_boot=np.zeros((num_exp))
    num_data=len(data)
    for ii in range(num_exp):
        np.random.seed(seed+ii)
        tempind=np.random.choice(num_data,num_data,replace=True)
        sta_boot[ii]=func(data[tempind])
    #return np.std(sta_boot,ddof=1)
    return np.percentile(sta_boot,2.5),np.percentile(sta_boot,97.5)