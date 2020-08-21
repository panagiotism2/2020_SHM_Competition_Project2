# Import libraries
import numpy as np
import sys 
import pickle
from keras.models import  load_model

from mapminmax import mapminmax_a



def get_predictions(data_raw):
    
    # Load Predictive Model
    custom_Name='NCA9' 
    n_epochs_old=10000# number of epochs of previous training just for the name
    filepath = custom_Name+'_LSTM_e'+str(n_epochs_old)+'_best.hdf5' # Old model name 
    model = load_model(filepath)
    
    #### Get Features
    feat=get_features(data_raw)
    
    #### Normalise Features    
    # Load corresponding training data
    FileName=custom_Name+'_data.pkl'
    with open(FileName, 'rb') as f:  # Python 3: open(..., 'rb')
        paramap,_,_,_,_,_,_ = pickle.load(f)
        
    # Normalise test data
    _,x_test = mapminmax_a(paramap, feat,feat)
    
    # Expand dimension for LSTM
    x_test=np.expand_dims(x_test,axis=-1)
    
    
    #### Get Predictions
    model_preds = model.predict(x = x_test)
    class_prediction = np.argmax(model_preds , axis=-1) # Get prediction
    pred_probabilty = np.zeros(np.size(class_prediction))
    for i in range(len(class_prediction)):
        pred_probabilty[i]=model_preds[i,class_prediction[i]] # get the probability of the prediction

    
    return(class_prediction,pred_probabilty)

def get_features(data_raw):

    from scipy import stats
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    plt.ioff()
    import matplotlib.pyplot 
    from scipy import signal
    from scipy.fft import fft
    import math
    
    
    def nans(data):
        count = 0
        for i in data:
            if not np.isnan(i):
                count += 1
        n_nans=len(data)-count
        return n_nans 
    
    def movingAverage(data,w):
        #k=np.ones([1,w])/w
        y=np.convolve(data, np.ones((w,))/w, mode='same')
        return(y)
    
    def f_FFT(sig,fsamp):
        NFFT=round(len(sig)/10) # take the windowlength of calculation as 10% of the length of the dataset
        OVRLP=round(NFFT/3) # use 30% of the windowlength as overlapping between two windows
        nw = (len(sig)-OVRLP)//(NFFT-OVRLP) # number of windows for calculation
        Wf = signal.hamming(NFFT)                           # window function: hamming
        
        # Define Starting points and end points of data-windows
        startp = np.arange(0,nw,1)*(NFFT-OVRLP)+1     
        endp = np.arange(NFFT,len(sig),NFFT-OVRLP)
        
        # initialize Power-spectrum density matrix
        PSDd = np.zeros([NFFT//2+1,nw])
        
        # Calculate PSD
        for k in range(nw):
            S = sig[startp[k]-1:endp[k]];     #  take a window out of signal
            
            S = np.multiply(S,Wf);                      # apply window function (Hamming) to the signal
            jbjf = fft(S);                  # fourier transform
            
            # adapt fourier transform to power-spectrum density, PSD
            jbjf[1:] = 2*jbjf[1:] 
            jbjf = np.divide(jbjf,NFFT)
            PSDd[:,k] = np.multiply(jbjf[0:NFFT//2+1], np.conj(jbjf[0:NFFT//2+1])  )
        
        # Get average PSD
        psd=np.mean(PSDd, axis=1)
        
        # construct the corresponding frequency values
        freq = np.arange(0,(NFFT-1)/NFFT*fsamp,fsamp/NFFT);
        fVals = freq[1:NFFT//2+1];
        return(psd,fVals)
    
    def f_FFTsimple(sig,fsamp):
        
        jbjf = fft(sig);                  # fourier transform
        
        # adapt values for power-spectrum density, PSD
        jbjf[1:] = 2*jbjf[1:]
        jbjf = np.divide(jbjf,len(sig))
        psd = np.real(np.multiply(jbjf[0:len(sig)//2+1],np.conj(jbjf[0:len(sig)//2+1])))
        
        # construct the corresponding frequency values  
        freq = np.arange(0,(len(sig)-1)/len(sig)*fsamp,fsamp/len(sig))
        fVals = freq[1:len(sig)//2+1];
        
        return(psd,fVals)
       
    def f_getEntr(x):
        x=np.divide(x,np.sum(x))
        log2vect=np.zeros(np.size(x))
        for k in range(len(x)):
            if x[k]==0:
                log2vect[k]=0
            else:
                    log2vect[k]=math.log2(x[k])
                    
        Entropy=-1*np.sum(np.multiply(log2vect,x))
        
        return(Entropy)
    
    def f_findmatch(Array,TargetVal):
        # find closest match of a value in an array
        dummy = abs(Array-TargetVal)     #absolute value of difference between array values and target   
        Index=np.argmin(dummy)
        return Index
    
    ### INPUT
    data=data_raw
    
    # AR Features
    AR_order=4
    
    # Moving Average features
    STAdur = 20;    #number of datasamples for calculation the short-term averagege
    MTAdur = 400;   #number of datasamples for calculation the medium-term averagege
    LTAdur = 6000;  #number of datasamples for calculation the long-term averagege
    n_bins=26
    
    # Frequency Domain features
    fsamp=1
    ### 
    
    #Preallocation
    n_ch=data.shape[1]
    d_mean=np.zeros(n_ch)
    d_median=np.zeros(n_ch)
    d_mode=np.zeros(n_ch)
    d_nans=np.zeros(n_ch)
    d_std=np.zeros(n_ch)
    d_var=np.zeros(n_ch)
    d_skew=np.zeros(n_ch)
    d_kurt=np.zeros(n_ch)
    d_length=np.zeros(n_ch)
    d_ar1=np.zeros(n_ch)
    d_ar2=np.zeros(n_ch)
    d_ar3=np.zeros(n_ch)
    VarSTAdist=np.zeros(n_ch)
    VarMTAdist=np.zeros(n_ch)
    VarLTAdist=np.zeros(n_ch)
    HomogSTAdist=np.zeros(n_ch)
    HomogMTAdist=np.zeros(n_ch)
    HomogLTAdist=np.zeros(n_ch)
    Entr1=np.zeros(n_ch)
    Entr2=np.zeros(n_ch)
    Kurt1=np.zeros(n_ch)
    Kurt2=np.zeros(n_ch)
    Homog1=np.zeros(n_ch)
    Homog2=np.zeros(n_ch)
    PSDrange1=np.zeros(n_ch)
    PSDrange2=np.zeros(n_ch)
    JBJFrange1=np.zeros(n_ch)
    JBJFrange2=np.zeros(n_ch)
    
    for i in range(n_ch):  
        # Exclude NaNs
        nan_array = np.isnan(data[:,i])
        not_nan_array = ~ nan_array # data wo NaNs
        data_ch_noNans = data[not_nan_array,i] 
        
        if len(data_ch_noNans)!=0:
        
            #### Descriptive Statistics
            d_nans[i]=nans(data[:,i])
            d_mean[i]=np.mean(data_ch_noNans)
            d_median[i]=np.median(data_ch_noNans)
            d_mode[i]=stats.mode(data_ch_noNans)[0]
            d_std[i]=np.std(data_ch_noNans)
            d_var[i]=np.var(data_ch_noNans)
            d_skew[i]=stats.skew(data_ch_noNans)
            d_kurt[i]=stats.kurtosis(data_ch_noNans, fisher=False)
            d_length[i]=len(data_ch_noNans)
            
            
            
            #### AR Parameters
            try:
                rho, _ = sm.regression.yule_walker(data_ch_noNans, order=AR_order)
                d_ar1[i]=rho[0]
                d_ar2[i]=rho[1]
                d_ar3[i]=rho[2]
            except:
                pass
            
            
            
            #### Moving Average features
            sta=movingAverage(data_ch_noNans,STAdur)
            mta=movingAverage(data_ch_noNans,MTAdur)
            lta=movingAverage(data_ch_noNans,LTAdur)   
            
            # Get the distribution of the short-, medium-, and long-term moving average values
            fig = matplotlib.figure.Figure()
            ax = matplotlib.axes.Axes(fig, (0,0,0,0))
            Pshort = ax.hist(sta,bins=n_bins)[0]
            del ax, fig
            
            fig = matplotlib.figure.Figure()
            ax = matplotlib.axes.Axes(fig, (0,0,0,0))
            Pmedium = ax.hist(mta,bins=n_bins)[0]
            del ax, fig

            fig = matplotlib.figure.Figure()
            ax = matplotlib.axes.Axes(fig, (0,0,0,0))
            Plong = plt.hist(lta,bins=n_bins)[0]
            del ax, fig
            
            # Variance
            VarSTAdist[i] = np.var(Pshort)
            VarMTAdist[i] = np.var(Pmedium)
            VarLTAdist[i] = np.var(Plong)
            
            # Homogenity of distribution (maximum value compared to mean value)
            HomogSTAdist[i] = Pshort.max()/np.mean(Pshort);
            HomogMTAdist[i] = Pmedium.max()/np.mean(Pmedium);
            HomogLTAdist[i] = Plong.max()/np.mean(Plong);
        
        
        
            #### Frequency Domain Features
            # calculate the frequency spectrum using windows
            psd,fVals=f_FFT(data_ch_noNans-np.mean(data_ch_noNans),fsamp)  
            
            # calculate the frequency spectrum using the entire dataset 
            jbjf,fVals2 = f_FFTsimple(data_ch_noNans-np.mean(data_ch_noNans),fsamp);
            
            # Get features
            Entr1[i]=f_getEntr(np.divide(psd,psd.max()))
            Entr2[i]=f_getEntr(np.divide(jbjf,jbjf.max()))
            
            Kurt1[i] = stats.kurtosis(np.divide(psd,psd.max()))
            Kurt2[i] = stats.kurtosis(np.divide(jbjf,jbjf.max()))
            
            # homogenity
            Homog1[i] = np.divide(np.divide(psd,psd.max()).max(),   np.mean(np.divide(psd,psd.max())))
            Homog2[i] = np.divide(np.divide(jbjf,jbjf.max()).max(),   np.mean(np.divide(jbjf,jbjf.max())))
            
            # Get the cumulated frequency domain spectrum
            PSDcumul = np.cumsum(psd)    
               
            # Get range of frequency containing 95% of energy
            Index_1 = f_findmatch(PSDcumul,0.025*PSDcumul[-1])
            Index_2 = f_findmatch(PSDcumul,0.975*PSDcumul[-1])
            Index_2 = np.minimum(Index_2,len(fVals)-1)
            PSDrange1[i] = fVals[Index_2] - fVals[Index_1]
            
            # Get range of frequency containing 50% of energy
            Index_1 = f_findmatch(PSDcumul,0.25*PSDcumul[-1])
            Index_2 = f_findmatch(PSDcumul,0.75*PSDcumul[-1])
            PSDrange2[i] = fVals[Index_2] - fVals[Index_1]
            
            
            # Get the cumulated frequency domain spectrum (over entire dataset at once)   
            JBJFcumul = np.cumsum(jbjf);
            
            # Get range of frequency containing 95% of energy
            Index_1 = f_findmatch(JBJFcumul,0.025*JBJFcumul[-1])
            Index_2 = f_findmatch(JBJFcumul,0.975*JBJFcumul[-1])
            Index_2 = np.minimum(Index_2,len(fVals2)-1)
            JBJFrange1[i] = fVals2[Index_2] - fVals2[Index_1]
            
            # Get range of frequency containing 50% of energy
            Index_1 = f_findmatch(JBJFcumul,0.25*JBJFcumul[-1])
            Index_2 = f_findmatch(JBJFcumul,0.75*JBJFcumul[-1])
            JBJFrange2[i] = fVals2[Index_2] - fVals2[Index_1]
    
    feat_1=np.concatenate((d_mean[:,None],d_median[:,None],d_mode[:,None],d_skew[:,None],d_kurt[:,None],d_nans[:,None]),axis=1)          
    feat_2=np.concatenate((d_ar1[:,None],d_ar2[:,None],d_ar3[:,None]),axis=1)
    feat_3=np.concatenate((VarSTAdist[:,None],VarMTAdist[:,None],VarLTAdist[:,None],HomogSTAdist[:,None],HomogMTAdist[:,None]),axis=1)
    feat_4=np.concatenate((Entr1[:,None],Entr2[:,None],Kurt1[:,None],Homog2[:,None],PSDrange1[:,None],PSDrange2[:,None],JBJFrange1[:,None],JBJFrange2[:,None]),axis=1)  
    
    # Assembel features together
    feat_all=np.concatenate((feat_1,feat_2,feat_3,feat_4),axis=1)

    return (feat_all)