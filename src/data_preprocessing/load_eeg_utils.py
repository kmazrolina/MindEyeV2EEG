

def write_args_to_file(args, filename="args_output.txt"):
    """
        Quick Code to save parameters as a txt
    """
    with open(filename, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

            
def epoching(args,dataPath,dataPart,seed):
    
    """
        Load Your data and performs Trial Segmentation. Data that are provided are taken directly from raw_data files and must be a dictionary with keys:
            + raw_eeg_data - contains a matrix of chan x time
            + sfreq - current sampling freq of the data (DO NOT CONFUSE WITH RESAMPLING FREQUENCY)
            + ch_types - a type of channel (last one has to be a stim type)
            + ch_names - names of channels (Elecrodes!)

        
        Peprocessing steps are as follows
            + Load Data
            + Transform into raw format usingmne.create_info  and mne.io.RawArray
            + find events with method find_events using Stim channel
            + Segment to trial - either 0.2 to 0.8 s as intendet (But WRONG in my opinion) or 0.1 to 0.1 (as it should be!)
            + Resample data when the latter is true to a specified freequency (Reduced Dimensionality) 
            + Sort the data - Required, for images are differently rpesented in repetitions - select UTMOST 2x the repeated images! in case of train

        Output:
            +Epoched Data - a matrix of data containing Images x Repetitions*Sessions x Channels x Time
                + Test dataset  - 200  x 20*4 x 63 x 100
                + Train dataset - 8750 x 2*4  x 63 x 100
    
    """
    import os 
    import mne
    import numpy as np
    from sklearn.utils import shuffle

    sessions = args.sessionNumber;
    sfreq  = args.sfreq

     # Already Preprocessing as a position in a relative Timeseries (Computed with resampling freq, length of dataslice (based on trial))
    
    tbef = args.tbef; # In -t seconds -(How many seconds before)
    taft = args.taft; # In t seconds (How many seconds After)
    ### Loop across data collection sessions ###
    epoched_data = []
    img_conditions = []
    for s in range(sessions): # Iterate thorugh sessions


        ### ===  Load the EEG data and convert it to MNE raw format  === ###
        dataP = os.path.join(dataPath,'ses-'+format(s+1,'02'),dataPart)
        eeg_data = np.load(os.path.join(dataP),
        allow_pickle=True).item()

        # Get fields from dictionariry 
        ch_names = eeg_data['ch_names']
        sfreq2    = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']


        ### ===  Convert to MNE raw format - requires for smooth processing  === ###
        info = mne.create_info(ch_names, sfreq2, ch_types)
        raw = mne.io.RawArray(eeg_data, info)
        del eeg_data


        ### === Get events === ###
        events = mne.find_events(raw, stim_channel='stim')

        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        # Drop stim channel as it was used only in assessing the events:
        raw.drop_channels("stim") # It is done In-place


        ### === Epoching === ###
        epochs = mne.Epochs(raw, events, tmin=tbef, tmax=taft, baseline=(None,0),
        preload=True) # Using baseline and SWAP 
        del raw 


        ### === Resampling === ###
        if sfreq != 1000: # if necessery resample to a give sfreq - if sfreq = 1000 dont bother because nothing will change 
            epochs.resample(sfreq)
        ch_names = epochs.info['ch_names'] # save new Ch_name and TImes from Epochs
        times = epochs.times # 

        # #- Precautionary, and Important in Training Data (for Images were presented at random) and each session presents 2x half of Images
        data = epochs.get_data()
        events = epochs.events[:,2] # Gets ID of events
        img_cond = np.unique(events) # This is a list of All Image IDs in a given Session (sorted!)
        del epochs


       
       
        ### === Sort the data === ### 
    
        # Select only a maximum number of EEG repetitions (For Test it was 20 times)
        if dataPart == "raw_eeg_test.npy":
            max_rep = 20
        else:
            max_rep = 2 #2x half of images were presented each session

  
        # Sorted data matrix of shape: Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],data.shape[2]))
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
            # Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data) # Append for each Session
        img_conditions.append(img_cond) # append - resultin in 16750 image labels
        del sorted_data

    ### Output ###
    return epoched_data, img_conditions, ch_names, times, events


def mvnn(sessions, mvnn_dim,epoched_test, epoched_train):
    """
    It is almost as the original - I have my doubts - but it seems legit:
    Whiteniing data will make variance uniform across channels buuut i am unsure whether it will normalise artifacts
    
    Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch/repetitions of each image condition), and then average
    them across image conditions and data partitions. The inverse of the
    resulting averaged covariance matrix is used to whiten the EEG data
    (independently for each session).

    Parameters
    ----------
    sessions - number of sessions to include
    mvnn_dim - whether we perform whitening across times or across repetitions
    epoched_test, epoched_train - our data epoched by previous functioms 

    Returns
    -------
    whitened_test : list of float
    Whitened test EEG data.
    whitened_train : list of float
    Whitened training EEG data.

    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    ### Loop across data collection sessions ###
    whitened_test = []
    whitened_train = []
    for s in range(sessions):
        session_data = [epoched_test[s], epoched_train[s]]

        ### Compute the covariance matrices ###
        # Data partitions covariance matrix of shape:
        # Data partitions × EEG channels × EEG channels
        sigma_part = np.empty((len(session_data),session_data[0].shape[2],
        session_data[0].shape[2]))
        for p in range(sigma_part.shape[0]): # 
            # Image conditions covariance matrix of shape:
            # Image conditions × EEG channels × EEG channels
            sigma_cond = np.empty((session_data[p].shape[0],
            session_data[0].shape[2],session_data[0].shape[2]))
            for i in tqdm(range(session_data[p].shape[0])): # For nice progress Bar
                cond_data = session_data[p][i]
                # Compute covariace matrices at each time point, and then
                # average across time points
                if mvnn_dim == "time": # Compute MEAN Covariance for Whitening across Repetitions (There are only 2...) 
                    sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                        shrinkage='auto') for t in range(cond_data.shape[2])],
                        axis=0)
                # Compute covariace matrices at each epoch (EEG repetition),
                # and then average across epochs/repetitions
                elif mvnn_dim == "epochs":
                    sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]), # 100 x 64 -- time x Channels  for every Repetitions of stimuli
                        shrinkage='auto') for e in range(cond_data.shape[0])],
                        axis=0)
            # Average the covariance matrices across image conditions
            sigma_part[p] = sigma_cond.mean(axis=0)
        # Average the covariance matrices across image partitions
        sigma_tot = sigma_part.mean(axis=0) 
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5) # Compute Inverese Matrix as a final Whitening Step

        ### Whiten the data ###
        whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
            session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[0].shape))
        whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
            session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[1].shape))

    ### Output ###
    return whitened_test, whitened_train



def mergeData(args, whitened_test, whitened_train, img_conditions_train,
ch_names, times, seed):
    """
    Also almost unchange - i wanted it to be as close to the original as Possible!
    
    Merge the EEG data of all sessions together, shuffle the EEG repetitions
    across sessions and reshaping the data to the format:
    Image conditions × EEG repetitions × EEG channels × EEG time points.
    Then, the data of both test and training EEG partitions is saved.

    Parameters
    ----------
    session : how many sessions to inclue (although it IS required to be 4  because otherwise it will collapse)
    whitened_test : list of float  - Whitened test EEG data.
    whitened_train : list of float - Whitened training EEG data.
    img_conditions_train : list of int - Unique image conditions of the epoched and sorted train EEG data.
    ch_names : list of str -  EEG channel names.
    times : float EEG time points.
    repMean: Whethet to average the repetitions or not
    seed : int Random seed.

    Returns:
    test_dict  - a data dictionary that refleects preprocessed Train data
    train_dict - a data dictionary that refleects preprocessed Test data

    """

    import numpy as np
    from sklearn.utils import shuffle
    import os

   
    session = args.sessionNumber
    ### Merge and save the test data ###
    for s in range(session):
        if s == 0:
            merged_test = whitened_test[s]
        else:
            merged_test = np.append(merged_test, whitened_test[s], 1)
    del whitened_test

    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:,idx]


    


    # Insert the data into a dictionary
    test_dict = {
        'preprocessed_eeg_data': merged_test,
        'ch_names': ch_names,
        'times': times
    }
    del merged_test
    

    ### Merge and save the training data ###
    # Remember the data structure! It is not as straightforward as it seems: 4 Training sessions: Each with 8750 or so Images (Half Of All 165444) - Repeating 2 times
    # It means that ONLY after getting all 4 sessions we acheieve our 16540,4,64,100)

    for s in range(session):
        if s == 0:
            white_data = whitened_train[s]
            img_cond = img_conditions_train[s]
        else:
            white_data = np.append(white_data, whitened_train[s], 0)
            img_cond = np.append(img_cond, img_conditions_train[s], 0)
    del whitened_train, img_conditions_train
    # Data matrix of shape:
    # Image conditions × EEG repetitions × EEG channels × EEG time points
    merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2, # Bizzare way to structure you data but it makes sense!
        white_data.shape[2],white_data.shape[3]))
    for i in range(len(np.unique(img_cond))): # Aftrer gathering ALL images together we THEN try to order Out the data to reflect 4 different Image Viewing Condition
        # Find the indices of the selected category
        idx = np.where(img_cond == i+1)[0]
        for r in range(len(idx)):
            if r == 0:
                ordered_data = white_data[idx[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
        merged_train[i] = ordered_data
    
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:,idx]

   # Own Adjustmenet - across concatenating repetitions to a single Dataset:

    # Insert the data into a dictionary
    train_dict = {
    'preprocessed_eeg_data': merged_train,
    'ch_names': ch_names,
    'times': times
    }

    return test_dict, train_dict
def cutoffs(args,test_dict,train_dict):
    tbef = args.tbef; # In -t seconds -(How many seconds before)
    taft = args.taft; # In t seconds (How many seconds After)
    cutoff1 = int((args.cutoff1 - tbef)*args.sfreq*(taft-tbef)); # Input in -t seconds -(How many seconds before)
    cutoff2 = int((args.cutoff2 - tbef)*args.sfreq*(taft-tbef)); # Input in  t seconds (How many seconds After)
    train_dict2 = train_dict;
    test_dict2 = test_dict;

    train_dict2['preprocessed_eeg_data'] = train_dict['preprocessed_eeg_data'][:,:,:,cutoff1:cutoff2]
    test_dict2['preprocessed_eeg_data'] = test_dict['preprocessed_eeg_data'][:,:,:,cutoff1:cutoff2]

    return train_dict2, test_dict2

def checkData(test_dict, train_dict,argms, *args, **kwargs):

    """
        Performs data checks
        - Compares Our data with Preprocessed Data (the same channels) via difference to see whether they are the same. Only if the same Sampling Freq Is provided
        - 

         Average across Repetitions and IMAGES to get an ERP for a given individual on Ever channel:
         - Warning! Such ERPs are bound to be skewed in some cases! 

    """

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import os

    plt.ioff() # no interactive mode

    ### Create a Folder with Checks
    checkPath = os.path.join(argms.savePath,"checks",'sub-'+format(argms.currentSubjNum,'02'))
    if not os.path.isdir(checkPath):
        os.makedirs(checkPath)
    ### === Check whether Loaded Preprocessed data and our preprocessing is Coinciding === 
    # (!!!) Work in progress - not as intended -  

    ##--------------------------------------------------------##
    # ### ----  Check Difference of Our vs Preprocessed ---- ###  
    ##--------------------------------------------------------##

    # Check Whether Prep data are provided 
    dataPrepTest = args[0] if len(args) > 0 else None
    dataPrepTrain = args[1] if len(args) > 1 else None

    if argms.sfreq == 100 and (dataPrepTest is not None and dataPrepTrain is not None):
        
        cutoff1 = int((argms.cutoff1 - argms.tbef)*100*(argms.taft-argms.tbef)); # Input in -t seconds -(How many seconds before)
        cutoff2 = int((argms.cutoff2 - argms.tbef)*100*(argms.taft-argms.tbef)); # Input in  t seconds (How many seconds After)

        chanNames = ['Pz','Oz','POz']
        ### ger 3 reference Data for our own Things - for  PZ, Oz, POz
        refDataTest  =  [dataPrepTest['preprocessed_eeg_data'][:,:,dataPrepTest['ch_names'].index('Pz'),cutoff1:cutoff2], dataPrepTest['preprocessed_eeg_data'][:,:,dataPrepTest['ch_names'].index('Oz'),cutoff1:cutoff2], dataPrepTest['preprocessed_eeg_data'][:,:,dataPrepTest['ch_names'].index('POz'),cutoff1:cutoff2]]
        refDataTrain =  [dataPrepTrain['preprocessed_eeg_data'][:,:,dataPrepTest['ch_names'].index('Pz'),cutoff1:cutoff2], dataPrepTrain['preprocessed_eeg_data'][:,:,dataPrepTest['ch_names'].index('Oz'),cutoff1:cutoff2], dataPrepTrain['preprocessed_eeg_data'][:,:,dataPrepTest['ch_names'].index('POz'),cutoff1:cutoff2]]

        # True Data in Channels:
        trueDataTest = [test_dict['preprocessed_eeg_data'][:,:,test_dict['ch_names'].index('Pz'),:],
                        test_dict['preprocessed_eeg_data'][:,:,test_dict['ch_names'].index('Oz'),:],
                        test_dict['preprocessed_eeg_data'][:,:,test_dict['ch_names'].index('POz'),:]]

        trueDataTrain= [train_dict['preprocessed_eeg_data'][:,:,train_dict['ch_names'].index('Pz'),:],
                        train_dict['preprocessed_eeg_data'][:,:,train_dict['ch_names'].index('Oz'),:],
                        train_dict['preprocessed_eeg_data'][:,:,train_dict['ch_names'].index('POz'),:]]
        
        pdf1 = PdfPages(os.path.join(checkPath,'ERPs_Diff_test.pdf'))

        ### Plot Difference across 3 different channels for every Test Image 
        for j in range(trueDataTest[0].shape[0]):
            fig = plt.figure()
            for i in range(3):
                plt.plot(np.mean(trueDataTest[i][j,:,:]-refDataTest[i][j,:,:],0),label=chanNames[i])
            plt.legend()

            pdf1.savefig(fig)

            # Destroy the current figure to free up memory
            plt.close(fig)

        pdf1.close()

        ### Plot Overall Difference (Averaged across repetitions AND images)
        fig = plt.figure()
        for i in range(3):
            plt.plot( np.mean(np.mean(trueDataTrain[i] - refDataTrain[i],2),1),label=chanNames[i])
        plt.legend()
        plt.savefig(os.path.join(checkPath,"Train_Mean_SamplDiff"))
        

        
    ##--------------------------------------------------------------##
    # ### ----  Plot ERPs of Test Trials (Across Repetitions) ---- ###  
    ##--------------------------------------------------------------##


    # create a PdfPages object
    pdf = PdfPages(os.path.join(checkPath,'ERPs.pdf'))
    # define here the dimension of your figure

    # Retrieve the times array and calculate step for readable ticks
    times = test_dict["times"]
    step = len(times) // 10  # Adjust this as needed for readability (10 times less time xD)
    ticks = np.arange(0, len(times), step)
    tick_labels = np.round(times[ticks], 2)  # Limit to 2 decimal places for clarity


    for j in range(test_dict['preprocessed_eeg_data'].shape[0]): #Iterating through Repetitions

        fig = plt.figure(figsize=(20, 20))
        axes = fig.subplots(2, 2)

        plt.suptitle(f"Image {j}")
        #### Plot ERPs - For test purposes, assuming enough repetitions
            # Create a new figure
        
        #  For Oz 
        axes[0,0].plot(np.mean(test_dict['preprocessed_eeg_data'][j, :,train_dict['ch_names'].index('Oz'), :],axis=0))
        axes[0,0].set_title('Oz')
        axes[0,0].set_xticks(ticks, tick_labels)
        axes[0,0].set_xlabel("Time (s)")

        #  For POz 
        axes[0,1].plot(np.mean(test_dict['preprocessed_eeg_data'][j, :,train_dict['ch_names'].index('Pz'), :],axis=0))
        axes[0,1].set_title('POz')
        axes[0,1].set_xticks(ticks, tick_labels)
        axes[0,1].set_xlabel("Time (s)")

        #  For O1
        axes[1,0].plot(np.mean(test_dict['preprocessed_eeg_data'][j, :,train_dict['ch_names'].index('O1'), :],axis=0))
        axes[1,0].set_title('O1')
        axes[1,0].set_xticks(ticks, tick_labels)
        axes[1,0].set_xlabel("Time (s)")

        #  For O2
        axes[1,1].plot(np.mean(test_dict['preprocessed_eeg_data'][j, :,train_dict['ch_names'].index('O1'), :],axis=0))
        axes[1,1].set_title('O2')
        axes[1,1].set_xticks(ticks, tick_labels)
        axes[1,1].set_xlabel("Time (s)")
        # Save the current figure to the PDF
        pdf.savefig(fig)

        # Destroy the current figure to free up memory
        plt.close(fig)


    # Close the PdfPages object to ensure all plots are saved
    pdf.close()


    ##--------------------------------------------------------------------##
    # ### ----  Plot Correlation and Cosine  Similarity Differences ---- ###  
    ##--------------------------------------------------------------------##

    ### Check Data for repetitions:
    correlationMatrixRep = np.zeros([test_dict['preprocessed_eeg_data'].shape[0],test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Images X Channels
    correlationMatrixImg = np.zeros([test_dict['preprocessed_eeg_data'].shape[0],test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Images X Rep x  Channels
    cosineSimMatrixRep   = np.zeros([test_dict['preprocessed_eeg_data'].shape[0],test_dict['preprocessed_eeg_data'].shape[2]]) # Images X Channels
    cosineSimmatrixImg   = np.zeros([test_dict['preprocessed_eeg_data'].shape[0],test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Images X Rep x Channels
    
    ttestres     = np.zeros([test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels
    wilcoxres    = np.zeros([test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels
    ttestresCor  = np.zeros([test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels
    wilcoxresCor = np.zeros([test_dict['preprocessed_eeg_data'].shape[1],test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels

        
    ttestres2     = np.zeros([test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels
    wilcoxres2    = np.zeros([test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels
    ttestresCor2  = np.zeros([test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels
    wilcoxresCor2 = np.zeros([test_dict['preprocessed_eeg_data'].shape[2]]) # Rep X Channels


    from scipy import stats
    for j in range(test_dict['preprocessed_eeg_data'].shape[2]): #Iterating through Channels

        ### Get Similarity And Correlation  Every Repetitons to each other: Image(200) x Chan (63)  
        for i in range(test_dict['preprocessed_eeg_data'].shape[0]): #Iterating through Images
             # Average Correlation of Every repetion to each other across images: 
            correlationMatrixRep[i,:,j] = np.mean(np.corrcoef(test_dict['preprocessed_eeg_data'][i,:,j,:]),axis=1);
        
            # Compute Cosine Similarity of every Repetition to each other (Average it)
            matrixDot = np.dot(test_dict['preprocessed_eeg_data'][i,:,j,:],test_dict['preprocessed_eeg_data'][i,:,j,:].T);
            matrixDiag = np.diag(matrixDot)
            cosineSimMatrixRep[i,j] = np.mean(matrixDot/np.sqrt(np.dot(matrixDiag,matrixDiag.T))) # Get Mean Cosine Similarity for Given Image x Channel 

        ### Similarity across different Images : Image (200) x Rep (80) x Channel (63) 
        for i in range(test_dict['preprocessed_eeg_data'].shape[1]): # Iterate Through Repetitions
            # Average Correlation Coefficient across Different Images (No effect of repetitions should be visible)
            correlationMatrixImg[:,i,j] = np.mean(np.corrcoef(test_dict['preprocessed_eeg_data'][:,i,j,:]),axis=0);

            ### Compute Cosine Similarity between Different Images
            matrixDot = np.dot(test_dict['preprocessed_eeg_data'][:,i,j,:],test_dict['preprocessed_eeg_data'][:,i,j,:].T);
            matrixDiag = np.diag(matrixDot)
            cosineSimmatrixImg[:,i,j] = np.mean(matrixDot/np.sqrt(np.dot(matrixDiag,matrixDiag.T)),axis=1) # Get Cosine SImilarities for All Images , for given Repetition x Channel 

    # We get Images x Chan (Representing Similarity Across Repetitions) - representing for each image similarity in all repetitions 
    # Images x Rep x Chan (to be fair a mean of Rep here should be used, or chosen at random ) - representing for Each Image how it is similar to all the other Images (Rep x Chan additionally)

    ### Test - Both Similarity and Correlation

    # Check statistically  whether Across All Images (for a given Channel) (And Repetitions if there is one in Img condios)
    for j in range(test_dict['preprocessed_eeg_data'].shape[2]): #Iterating through Channels
        for i in range(test_dict['preprocessed_eeg_data'].shape[1]): #Iterating through repetitions

            ttestres[i,j] = stats.ttest_ind(cosineSimMatrixRep[:,j],cosineSimmatrixImg[:,i,j])[0]
            wilcoxres[i,j] = stats.wilcoxon(cosineSimMatrixRep[:,j],cosineSimmatrixImg[:,i,j])[1]
            ttestresCor[i,j] = stats.ttest_ind(correlationMatrixRep[:,i,j],correlationMatrixImg[:,i,j])[0]
            wilcoxresCor[i,j] = stats.wilcoxon(correlationMatrixRep[:,i,j],correlationMatrixImg[:,i,j])[1]

    ### Similarity Test Histogram (Channels X repetitions) - Tstat oF DIFFERENCE 
    fig = plt.figure(figsize=(20, 20))
    axes = fig.subplots(2, 2)

    axes[0,0].set_title("Tstats of CosSim between Repetitions and Each Image")
    currAx = axes[0,0].imshow(ttestres)
    axes[0,0].set_xlabel("Channel")
    axes[0,0].set_ylabel("Repetition")
    fig.colorbar(currAx, ax=axes[0,0])

    axes[0,1].set_title("Tstats of Corr between Repetitions and Each Image")
    currAx = axes[0,1].imshow(ttestresCor)
    axes[0,1].set_xlabel("Channel")
    axes[0,1].set_ylabel("Repetition")
    fig.colorbar(currAx, ax=axes[0,1])


    axes[1,0].set_title("Tstats of CosSim between Repetitions and Each Image")
    axes[1,0].hist(np.reshape(ttestres,[ttestres.shape[0]*ttestres.shape[1]]),50)
    axes[1,0].axvline(1.96,color="r")
    axes[1,1].set_title("Tstats of Corr between Repetitions and Each Image")
    axes[1,1].hist(np.reshape(ttestresCor,[ttestresCor.shape[0]*ttestresCor.shape[1]]),50)
    axes[1,1].axvline(1.96,color="r")

    plt.savefig(os.path.join(checkPath,"Corr_Sim_Ttests"))

    cosineSimmatrixImg2 = np.mean(cosineSimmatrixImg,axis=1)
    correlationMatrixImg2 = np.mean(correlationMatrixImg,axis=1)
    for j in range(test_dict['preprocessed_eeg_data'].shape[2]): #Iterating through Channels
        ttestres2[j] = stats.ttest_ind(cosineSimMatrixRep[:,j],cosineSimmatrixImg2[:,j])[0]
        wilcoxres2[j] = stats.wilcoxon(cosineSimMatrixRep[:,j],cosineSimmatrixImg2[:,j])[1]
        ttestresCor2[j] = stats.ttest_ind(correlationMatrixRep[:,i,j],correlationMatrixImg2[:,j])[0]
        wilcoxresCor2[j] = stats.wilcoxon(correlationMatrixRep[:,i,j],correlationMatrixImg2[:,j])[1]

    fig = plt.figure(figsize=(20, 20))
    axes = fig.subplots(1,2)

    axes[0].set_title(f"Tstats of CosSim between Repetitions and Each Image ({round(sum(ttestres2 < 1.96)/len(ttestres2)*100,2)} %)")
    axes[0].hist(ttestres2,8)
    axes[0].axvline(1.96,color="r")
    axes[1].set_title(f"Tstats of Corr between Repetitions and Each Image ({round(sum(ttestresCor2 < 1.96)/len(ttestresCor2)*100,2)} %)")
    axes[1].hist(ttestresCor2,8)
    axes[1].axvline(1.96,color="r")
    plt.savefig(os.path.join(checkPath,"Corr_Sim_TtestsAcross"))

    ### Check whether Significant number of those are > 5:
    
def saveData(save_dir,test_dict,train_dict,repMean):
# Saving directories
    import os
    import numpy as np

    if repMean: # ONLY NOW At the end you save your data as follows
        train_dict['preprocessed_eeg_data'] = np.mean(train_dict['preprocessed_eeg_data'] ,1)
        test_dict['preprocessed_eeg_data'] = np.mean(test_dict['preprocessed_eeg_data'] ,1)


    file_name_test = 'preprocessed_eeg_test.npy'
    file_name_train = 'preprocessed_eeg_training.npy'
    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_test), test_dict)

    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_train),train_dict)
