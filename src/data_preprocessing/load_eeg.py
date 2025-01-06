"""
    A Script for loading EEG data in Directory of THINGS-EEG2 Dataset:

    QUICK SETUPS:

    1) Things- EEG2 Proposed Preprocessing

        python loadEEG.py <dataPath> --subjIds 1 2 3  --sfreq 100 --sessionNumber 4

    2) ATM - proposed Preprocessing

        python loadEEG.py <dataPath> --subjIds 1 2 3 --sfreq 250 --sessionNumber 4 --repMean

    3) My Preprocessing 

        python loadEEG.py< dataPath> --subjIds 1 2 3 --repMean --sfreq 1000 --sessionNumber 4 --repMean


    Usage: (write down in a terminal: )
        + loadEEG dataPath --savePath --subjIds LIST ---repMean --sfreq INT --mnvdim --check INT --sessionNumber INT 

    Arguments:
        + dataPath - a path to a directory where the data are located: The directory should contain: raw_data folder in which raw_data are stored ()
            Warning! If you intendt to check with preprocessed data - yoou will have to also have a preprocessed_data folder! 
        + --savePath - if unspecified use dataPath
        + --subjNum LIST - (from 1 to 10) Number of a subject separated with space. Default is all subjects 1-10
        + --repMean -  default False - n analysis mode whether to Average the Repeating Evoked EEG to Image (ATMs thought that it improved performance)
        + --sfreq INT - default (1000), Resampling Frequency. If We want analysis to follow strictly the ImageNet and ATM papers:
                100 - will make the same analysis as in Image Net
                1000 - will make my own analysis, adjusting cuts and performing NO downsampling 
        + --mnvdim - In Which dimension do we compute our Multivariat Noise Normalization (Whitening) - either "time" or "epochs"
        + --check INT - accepts 0,1,2 --> performs data Checks:  
                0 - No data Check; 
                1 - Plot ERPs across all Image Samples 
                2 - Check Allignment with Preprocessed Data + Plot ERPs in a PDF (Warning! Requires preprocessed_data Folder in your path!)
        + --sessionsNumber - how many sessions do you want to include (up to 4)
        + --tbef FLOAT - in seconds (How many seconds before the stimulus onset are used to form a trial (negative- signiffying "before" )) - default -0.2
        + --taft FLOAT - in seconds (How many seconds before the stimulus onset are used to form a trial (positive)) - default: 0.8
        + --cutoff1 FLOAT -in seconds lower bound of period used in analysis 
        + --cutoff2 FLOAT -in seconds upper bound of period used in analysis 


"""

from load_eeg_utils import epoching,mvnn,mergeData,saveData,checkData,write_args_to_file
import argparse
import numpy  as np
import os


if __name__ == "__main__":

    #### ===== ARGUMENTS ===== ####

    ### Parse Input Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('dataPath')
    parser.add_argument('--savePath', help='Path to save the results')
    parser.add_argument('--subjIds', nargs='+', type=int, default=[i for i in range (1,11)])
    parser.add_argument('--sfreq',type=int,default=1000)
    parser.add_argument('--mnvdim',default="epochs")
    parser.add_argument('--repMean', action="store_true")
    parser.add_argument('--check',default=0,choices=[0,1,2],type=int)
    parser.add_argument('--sessionNumber', default = 4, choices=[1,2,3,4],type=int)
    parser.add_argument('--cutoff1', default = 0,  type = float, help = "Cutting trial to relevant subject (lower bound) in seconds from trial onset")
    parser.add_argument('--cutoff2', default = 0.2, type = float,help = "Cutting trial to relevant subject (upper bound) in seconds from trial onset")
    parser.add_argument('--tbef', default = -0.2, type = float,help = "number of seconds before trial onset (with minus)")
    parser.add_argument('--taft', default = 0.8, type = float,help = " number of seconds after trial onset")

    args = parser.parse_args()

    ### Create a Save Path 
    if not os.path.isdir(args.savePath):
        os.makedirs(args.savePath)

    ### Make parameter Log 
    write_args_to_file(args,os.path.join(args.savePath,"parameters.txt"))
    if args.savePath is None:
        args.savePath = args.dataPath

    for subjNum in args.subjIds:
        args.currentSubjNum = subjNum
        print("=== Getting EEG Folders ===\n")
        print(f"   Data from: {args.dataPath} ")
        print(f"   Subj numb: {subjNum}")

        ## Create a new Save - Folder - PreprocessedEEG  
        savePath = os.path.join(args.savePath, 'sub-'+format(subjNum,'02'))
        if not os.path.isdir(savePath):
            os.makedirs(savePath)


        ### Get Subject Raw Data
        print(f"   Getting Raw Data....")
        subjectPathRaw =  os.path.join(args.dataPath,'sub-'+format(subjNum,'02'))
        seednum = 20200220

        # Test
        dataPart = "raw_eeg_test.npy"
        print(args.sfreq)
        epoched_test, _, ch_names, times,_ = epoching(args,subjectPathRaw,dataPart,seednum)
        # Train:

        dataPart = "raw_eeg_training.npy"
        epoched_train, img_conditions_train, _, _,events = rawDict = epoching(args,subjectPathRaw,dataPart,seednum)

        ### Whiten Data (Good Practice!) - I need to check how to do it 
        whitened_test, whitened_train = mvnn(args.sessionNumber,args.mnvdim,epoched_test,epoched_train)


        ### Merge EEG Data

        ### Merge and save the test data ###

        test_dict, train_dict = mergeData(args.sessionNumber, whitened_test, whitened_train, img_conditions_train,
            ch_names, times, seednum)

        print(test_dict)

        ### Save EEG Data
        saveData(savePath,test_dict,train_dict, args.repMean)
        
        ###  ======Check Data with Original EEG: ======

        if args.check == 2: #Data Alignment with original + ERPs 
            ## - Does not work as intended!!! --> Clearly Shuffeling totally changes where the Representations Lay 
            ### Get Preprocessed_Folder 
            subjectPathPreprocessedTest =os.path.join(args.savePath,'sub-'+format(subjNum,'02'),"preprocessed_eeg_test.npy")
            subjectPathPreprocessedTrain =os.path.join(args.savePath,'sub-'+format(subjNum,'02'),"preprocessed_eeg_training.npy")

            ### Get Preprocessed Data (For Checking Purposes!)
            dataPrepTest = np.load(subjectPathPreprocessedTest,allow_pickle=True).item()
            dataPrepTrain = np.load(subjectPathPreprocessedTrain,allow_pickle=True).item()

            prepShape = dataPrepTest['preprocessed_eeg_data'].shape # Images x Sessions x Chan (17) x Time (100)
            print(f"Test  Prep Shape:       Trials: {prepShape[0]};\n       Sessions: {prepShape[1]};\n       Channels: {prepShape[2]};\n       Timepoints: {prepShape[0]};\n" ,prepShape)
            prepShape = dataPrepTrain['preprocessed_eeg_data'].shape # Images x Sessions x Chan (17) x Time (100)
            print(f"Train Prep Shape:       Trials: {prepShape[0]};\n       Sessions: {prepShape[1]};\n       Channels: {prepShape[2]};\n       Timepoints: {prepShape[0]};\n" ,prepShape)
            checkData(test_dict,train_dict,args,dataPrepTest,dataPrepTrain)


        elif args.check == 1: # ERPs Only
            checkData(test_dict,train_dict,args)

        









