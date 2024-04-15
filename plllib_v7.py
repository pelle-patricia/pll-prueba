import torch
import numpy as np
from math import *
from scipy import signal as sig
from speechbrain.processing.signal_processing import convolve1d
from torchaudio import transforms
import time
from pll import pll_signal


class PLLBank(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        # Parámetros globales
        self.n_channels = params['n_channels'] # Cantidad de canales del banco de plls
        self.Fs = params['Fs'] # Frecuencia de muestreo de la señal de entrada
        self.show_all = params.get('show_all',False) # Si es True devuelve freq y lock además de 
                                                    # freq y lock submuestreados cada ts
                                                    # Si es False solo devuelve freq y lock
                                                    # submuestreadas cada ts
        self.ts = params.get('ts',10) # Muestras de lock y freq cada ts  (en mseg)
        # Parámetros del filtro coclear
        self.lh = params.get('lh',2048)  # Duración de la respuesta impulsiva del filtro FIR coclear
        self.fp_minus_fz = params.get('fp_minus_fz',0.1*np.log(2.)/log(np.sqrt(2.))) # distancia desde la 
                                                                                     # frecuencia central a la frecuencia cero.
        self.gain = params['gain']  # Ganancia aplicada a la señal de entrada
        # Parametros del filtro de loudness
        self.loudness = params.get('loudness',True) # Define la utilización o no del filtro de loudneess
        # Parámetros del banco de filtros cocleares
        self.fpmin = params['fpmin'] # Frecuencia central del filtro más grave
        self.fpmax = params['fpmax'] # Frecuencia central del filtro más agudo
        self.Qmin = params.get('Qmin',0.15) # Q del filtro más grave
        self.Qmax = params.get('Qmax',0.15) # Q del filtro más agudo
        self.Kmel = params.get('Kmel', 200) 
        # Parámetros de los filtros de freq y lock (pasabajos)
        self.fcf = params.get('fcf',25) # Frecuencia de corte del filtro de freq
        self.fcl = params.get('fcl',100) # Frecuencia de corte del filtro de lock 
        # Parámetros del pll
        self.sita = params.get('sita',0.7) # Paŕametro del filtro interno del pll
        self.Fn = params.get('Fn',200) # Parámetro del filtro interno del pll
        self.fc_lock = params.get('fc_lock',4) # Parámetro del filtro interno del pll
        self.agcgain = params.get('agcgain',10.0) # Ganancia del AGC del pll

    def eq_loudness(self,fp):
        #-------------------------------------------------------
        # Sacado de:
        # Perceptual linear predictive (PLP) analysis of speech 
        # Hynek Hermansky
        # J. Acoust. Soc. Am. 87, 1738–1752 (1990)
        #-------------------------------------------------------
        om2 = fp * fp
        au1 = (om2 + 56.8e6)*om2*om2
        au2 = om2 + 0.38e9
        au3 = om2**3 + 9.58e26
        #au3=1
        au4 = (om2 + 6.3e6)**2 * au2 * au3
        au5 = au4**0.33
        return au5*1e-14
    
    def hn_cochlear(self, df=25/log(sqrt(2)), fp=2000):    
        b = 1/(2*pi*df)
        a = self.fp_minus_fz*2*pi*b
        om = np.linspace(0,self.Fs,int(self.lh),endpoint=False)*2*pi
        Hd = np.zeros(int(self.lh))
        Hd[om < fp*2*pi] = (2*pi*fp - om[om < fp*2.*pi])**a*\
            np.exp(-b*(2*pi*fp - om[om < fp*2.*pi]))
        Hd = Hd+ Hd[-1::-1]
        tmp = np.fft.ifft(Hd)
        hd = np.fft.fftshift(tmp)
        au = 1 #<-------------------------------anulamos el loudness
        if self.loudness == True:
            au = self.eq_loudness(fp)
        return  torch.tensor(np.real(hd)*au) 
    
        
    def filter_bank(self,wav):
        #wav[:,1:] = wav[:,1:] - 0.97*wav[:,:-1] # Filtro de pre-enfasis
        #wav = (wav.T - wav.mean(dim=1))/wav.std(dim=1) # Normalización
        wav *= self.gain # Aumento de la energía
        mel_slope = log( (self.Kmel + self.fpmax) / (self.Kmel + self.fpmin) )\
            / (self.n_channels - 1)
        mel_offset = log(1 + self.fpmin/self.Kmel)
        fp = torch.FloatTensor(self.n_channels)
        df = torch.FloatTensor(self.n_channels)
        kernel = torch.FloatTensor(1,self.lh,self.n_channels)
        for k in range(self.n_channels):
            fp[k] = self.Kmel * (exp(mel_offset + k*mel_slope) - 1)
            df[k] = ( self.Qmin +k/(self.n_channels-1)*(self.Qmax -  self.Qmin) ) * fp[k]
            kernel[:,:,k] = self.hn_cochlear(df = df[k].item() ,fp = fp[k].item())
        out = convolve1d(wav.unsqueeze(2).to('cpu'), kernel, padding=(0,self.lh),use_fft=True)
        out = out[:,int(self.lh/2):-int(self.lh/2),:]
        return out.permute(0,2,1).detach().numpy(), \
            kernel.squeeze(0).T.detach().numpy(),fp.detach().numpy(), df.detach().numpy()
    
    #---------------Función pll_signal en python modificada para la versión 6-------------------------
    # def pll_signal(self,ltimes, x, lock, vcos, wo, G0, G1, G2, Fs, a_lock, b_lock):
    #     s = np.zeros(8) 
    #     for n in range(ltimes):
    #         au = s[3] + wo*(n-1)
    #         vco = np.cos(au) # vco(n-1) = cos( wo.(n-1) + theta2(n-1))
    #         vco90 = np.cos(au - pi/2) # vco90(n-1) = sen( wo.(n-1) + theta2(n-1))
    #         #paso2, ud:
    #         au = x[n]*s[4]
    #         tmp0 = vco * au # ud(n) = x(n).agc(n-1).vco(n-1)
    #         tmp1 = vco90 * au # ul(n) = x(n).agc(n-1).vco90(n-1)
    #         tmp2 = np.copy(s[3]) # theta_2(n-1)
    #         #paso4, theta2:
    #         s[3] = G0*s[2] + s[3]  # theta_2(n) = G0.uf(n-1) + theta_2(n-1)
    #         #paso3, uf: 
    #         s[2] = (G1+G2)*tmp0 - G1*s[0]+ s[2]  # uf(n) = (G1+G2).ud(n) - G1.ud(n-1) + uf(n-1) 
    #         #paso5, lockin: lock(n) = b0.ul(n) +b1.ul(n-1) - a.lock(n-1)
    #         s[5] = b_lock[0]*tmp1 + b_lock[1]*s[1] - a_lock[1]*s[5] 
    #         #paso6, agc:
    #         s[4] = self.agcgain/np.exp(np.abs(3.*np.arctan(0.7*np.abs(s[5])))) # <--- antes valía 10.0/...
    #         #paso 7, guardar ud y ul:
    #         s[0] = tmp0 # ud(n) = ud(n-1)
    #         s[1] = tmp1 # ul(n) = ul(n-1)
    #         #5) Cálculo de la frecuencia
    #         s[6] = (s[3] - tmp2 + wo)*(Fs/2/np.pi) # freq(n) = (theta_2(n) - theta_2(n-1) + wo)/(2.pi)
    #         s[7] = vco
    #         # freq[n] = s[6]
    #         lock[n] = s[5]
    #         vcos[n] = s[7]
    #         # states[0,n] = s[0] # ud (theta_e)
    #         # states[1,n] = s[1] # ul
    #         # states[2,n] = s[2] # uf (omega_c)
    #         # states[3,n] = s[3] # theta_2
    #         # states[4,n] = s[4] # agc
    #         # states[5,n] = s[5] # lock
    #         # states[6,n] = s[6] # freq
    #         # states[7,n] = s[7] # vco   
    #     #return states[5,:], states[6,:], states




    def pll_bank_signal(self, out, fp): 
        bs , _, ltimes = out.shape
        # vco = np.zeros((self.n_channels,ltimes))
        lock = np.zeros((bs, self.n_channels, ltimes),dtype=np.float32)
        freq = np.zeros((bs, self.n_channels, ltimes),dtype=np.float32)
        vco = np.zeros((bs, self.n_channels, ltimes),dtype=np.float32)
        # Parámetros del filtro del pll
        wn = 2*np.pi*self.Fn/self.Fs
        G0 = 1
        G1 = (1 - exp(-2*self.sita*wn))/G0
        G2 = (1 + exp(-2*self.sita*wn) - 2*exp(-self.sita*wn)*cos(wn*sqrt(1-self.sita*self.sita)))/G0
        [b,a] = sig.ellip(N=1,rp=1.01,rs=20,Wn=2*pi*self.fc_lock,analog=True)
        [b_lock, a_lock] = sig.bilinear(b,a,self.Fs)
        # pllfilterparams = (G0, G1, G2, self.Fs, a_lock, b_lock)
        for k in range(self.n_channels):         
            wo = 2*np.pi*fp[k]/self.Fs
            for p in range(bs):
                pll_signal(ltimes,np.ascontiguousarray(out[p,k,:]), lock[p,k,:], freq[p,k,:],
                            vco[p,k,:], wo, G0, G1, G2, self.Fs, self.agcgain, a_lock, b_lock)
        return lock, freq, vco


    def forward(self,wav):
        # Recibe un batch de señales de igual longitud
        t0 = time.time()
        out, kernel, fp, df = self.filter_bank(wav)
        t1 = time.time()
        lock, freq, vco = self.pll_bank_signal(out,fp)
        t2 = time.time()
        #print('Tiempo total: {:.2f} segundos'.format(t2-t0))
        l = lock.shape[2]
        d = int(self.ts*1e-3*self.Fs)
        w = int(0.025*self.Fs)
        lr = l - l%d
        bs = out.shape[0]

        #zcross. Nueva feature que cuenta los cruces por cero de pendiente positiva del vco
        zcross = np.zeros(lock.shape,dtype=np.float32)

        # Filtrado y sub-muestreo de freq y lock
        bf, af = sig.butter(N=4, Wn=self.fcf*2/self.Fs, btype='low', analog=False)
        bl, al = sig.butter(N=4, Wn=self.fcl*2/self.Fs, btype='low', analog=False)
        bz, az = np.ones(w), np.ones(1)
        for k in range(self.n_channels):
            freq[:,k,:] = sig.filtfilt(bf, af, freq[:,k])
            lock[:,k,:] = sig.filtfilt(bl, al, lock[:,k])
            for p in range(bs):
                neg = vco[p,k,:] > 0 # <------ver que pasa cuando venga un batch de señales
                au = (neg[:-1] & ~neg[1:]).nonzero()[0]
                aux = np.zeros(vco[p,k,:].shape)
                aux[au] = 1
                zcross[p,k,:] = sig.filtfilt(bz, az, aux)

        if self.show_all == True:
            locks = lock.copy()
            freqs = freq.copy()
        lock = lock[:,:,:lr].reshape(lock.shape[0],lock.shape[1],-1,d)[:,:,:,0]
        freq = freq[:,:,:lr].reshape(freq.shape[0],freq.shape[1],-1,d)[:,:,:,0]
        zcross = zcross[:,:,:lr].reshape(zcross.shape[0],zcross.shape[1],-1,d)[:,:,:,0]
        dzcross = np.zeros(zcross.shape,dtype=np.float32)
        dzcross[:,1:,:] = np.diff(zcross, axis=1)
        dzcross[:,0,:] = dzcross[:,1,:]
        dzcross = dzcross/fp[None,:,None]
        # Lo ponemos en la forma (batch, t, f (filter channel), channel:frec/lock)
        pllout = torch.stack((torch.tensor(lock).permute(0,2,1), torch.tensor(freq).permute(0,2,1), 
                              torch.tensor(zcross).permute(0,2,1),torch.tensor(dzcross).permute(0,2,1))).permute(1,2,3,0) 
        
        # if self.show_all == True:
        #     return pllout, locks
        # else:
        #     return pllout.to('cuda')
        
        return pllout.to('cuda'), fp