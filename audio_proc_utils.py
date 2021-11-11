import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob


def readDir(dirname, Fs = 16000):
    
    '''
    Each audio clip should be upto 10s long; split larger audio files into many clips (non-overlapping) 

    Use load_audio(file) 
    
    Inputs: 
        dirname: (str) directory name
        Fs: (int) sampling rate
    Output: 
        x: np arrays of shape (Nclips, Nsamples) Nsamples correspond to 10s length. Use zero-padding for shorter clips.
    '''  

    required_duration = 10
    x = np.zeros(required_duration*Fs)
    
    for path in glob.iglob(f'{dirname}/*.wav'):
        audio = load_audio(path, Fs)
        num_per_clip = Fs*required_duration   # Number of point samples for 10s audio clip
        N = (int)(audio.shape[0]/num_per_clip)
        for i in range(0, N):
            x = np.vstack((x, audio[i*num_per_clip : (i+1)*num_per_clip]))
         
        # Last audio sample must be zero padded
        if(audio.shape[0]%num_per_clip != 0):
            seq = audio[N*num_per_clip:]
            seq = librosa.util.fix_length(seq, required_duration*Fs)
            x = np.vstack((x, seq))

    # We need to skip the initial zero array which was added for convenience
    x = x[1:, :]           # Nclips x Nsamples
    
    return x 


def load_audio(filename, Fs = 16000):
    '''
    Inputs: 
        filename: (str) filename
        Fs: (int) sampling rate
    Output: 
        x: 1D np array 
    '''
    
    x, sr = librosa.load(filename, sr=Fs)
    
    return x


def splitData(X, t, testFraction=0.2, randomize = False):
    """
    Split the data randomly into training and test sets
    Use numpy functions only
    Inputs:
        X: (np array of len Nclips) input feature vectors
        t: (np array of len Nclips) targets; one hot vectors
        testFraction: (float) Nclips_test = testFraction * Nclips
    Outputs:
        X_train: training set
        X_test: test set
        t_train: training labels
        t_test: test labels
    """

    np.random.seed(0)
    test_size = (int)(testFraction * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    training_index, test_index = indices[:X.shape[0] - test_size], indices[X.shape[0] - test_size:]
    X_train, t_train = X[training_index, :], t[training_index, :]
    X_test, t_test = X[test_index, :], t[test_index, :]

    return X_train, t_train, X_test, t_test


def audio2mfcc(x, n_mfcc = 20, Fs = 16000):
    
    '''
    Compute Mel-frequency cepstral coefficients (MFCCs)
    Inputs:
        x: np array of shape (Nclips,)
        Fs: (int) sampling rate
        n_mfcc: (int) number of MFCC features
    Output:
        X: (np array) MFCC sequence
    '''

    X = []
    for i in range(len(x)):
        mfccs = librosa.feature.mfcc(y=x[i], sr=Fs, n_mfcc=n_mfcc, hop_length=512)
        X.append(mfccs)
            
#     X = np.concatenate(X)
    X = np.stack(X, axis=0) # 3D array => Nsample 2D arrays of MFCCs
#     print("mfcc feat : ", X.shape)

    return X                # (Nclips, N_mfcc, N_frames)


def computeCM(y, y_hat):
    '''
    Compute confusion matrix to evaluate your model
    Inputs:
        y = labels 
        y_hat = predicted output
    Output:
        confusion matrix: confusion matrix
    '''

    #                True  
    #           Speech |  Music
    #          _________________
    #    Speech|       |       |
    # Pred     -----------------
    #    Music |       |       |
    #          -----------------
    confusion_matrix = np.zeros((2,2))
    for i in range(y.shape[0]):
        r = (int)(y_hat[i][0])
        c = (int)(y[i][0])
        confusion_matrix[r][c] += 1


    return confusion_matrix 