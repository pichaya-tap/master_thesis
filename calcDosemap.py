import numpy as np
import pandas as pd
import glob
import os
import sys
import getopt
import itertools
import time


def main(argv):

    opts, args = getopt.getopt(argv,"hi:e:",["ifile=","ofile="])
    for opt, arg in opts:
        if opt == "-e":
            ENERGY = arg
        else:
            print('Unknown Argument!')

    #####################################Refer to "'notes_dicom_images.txt"########################################################
    DATASET = 'DATASET'

   
    PHANTOM_SIZE = 400
    DELTA_X = 2.5 #mm
    DELTA_Y = 0.707031*2
    DELTA_Z = 0.707031*2
    
    VOXELNUMBER_X = int(PHANTOM_SIZE/DELTA_X)
    #+1 weil ich in der Simu aufgerundet habe und int abrundet...
    VOXELNUMBER_Y = int(PHANTOM_SIZE/DELTA_Y)+1
    VOXELNUMBER_Z = int(PHANTOM_SIZE/DELTA_Z)+1

    SOURCE_POSITION_XY = 0
    SOURCE_POSITION_Z = -200

    
    DENSITY_WATER = 1000 #kg/m^3
    WATERMASS_PER_VOXEL = DENSITY_WATER*DELTA_X*DELTA_X*DELTA_Z*(1e-9) #kg # need to convert mm^3 to m^3.

    elementarladung = 1.602176634e-19

    ##########################################################################################

    # Set the directory path where the files are located
    directory_path = "/gpfs001/scratch/schwar14/simudosemapheavyions/Watersimulation/Hadr01/Dataset_REAL_WATER/Dataset_Miriam/{}/".format(ENERGY)

    save_path = '/scratch/tappay01/watersimulation/{}'.format(DATASET)
    # Get a list of files matching the pattern "output_nt_Hits_t*.csv"

    file_pattern = os.path.join(directory_path, "output_nt_Hits_*.csv")

    inputfiles = glob.glob(file_pattern)

    dosemap = np.zeros(shape=(VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z))
    dosemap_sum = np.zeros(shape=(VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z))
    # create empty array to collect the sum of dosemaps, file name and path for saving
    filename_sum = '{}_{}.npy'.format(DATASET, ENERGY)
    output_path = os.path.join(save_path, filename_sum)

    for file in inputfiles:
        print('-- checking File {} --'.format(file))
        # Construct the save_filename based on the file name and position values
        filename = '{}_{}.npy'.format(ENERGY,file.split(".")[0].split("_")[-1])
        file_path = os.path.join(save_path, filename)
        # check if the target file exists already, skip
        if os.path.exists(file_path):
            print(f"Warning: File '{file_path}' already exists. Skipping calculation.")
            dosemap_sum += np.load(file_path)
            continue

        # Read csv and perform calculations
        print('-- Loading File {} --'.format(file))           
        start_time = time.time()  # Record the start time         
        data = pd.read_csv(file, sep = ',', skiprows = 11, names = ['X', 'Y', 'Z', 'Edep', 'Process', 'particle', 'E'])
        data_edep = data['Edep'].values
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        print(f"Time taken to read file: {elapsed_time} seconds")  

    
        # source position_brain (-110, POSITIONY,POSITIONZ).New source position_water (0,0,-200) 
        data['Z'] = data['Z'] + 90                                    
        
        data_for_hist = data[['X', 'Y', 'Z']]

        dosemap, edges = np.histogramdd(data_for_hist.values, bins = (VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z), range = ((-200, 200),(-200, 200),(-200, 200)), weights = data_edep)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        print(f"Time taken for histogram: {elapsed_time} seconds")
        #dosemap_shape = (283, 283, 160)
        print(dosemap.shape)

        # Umrechnen in Gray (Gy = J/kg und eV = e* J )
        dosemap = dosemap*(1e6)*elementarladung / WATERMASS_PER_VOXEL # from MeV
        # Save current dosemap
        np.save(file_path, dosemap)
        print('{} file added successfully'.format(filename))   
        # Add current dosemap to the overall dosemap
        #dosemap_sum += dosemap
    
    # Save the overall dose map        
    #np.save(output_path, dosemap_sum)
    #print('{} file saved successfully'.format(filename_sum))   


if __name__ == "__main__":
    main(sys.argv[1:])


    

