import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy.io
from astropy.table import Table, Column
import argparse
from astropy.io import fits
from multiprocessing import Pool
import re
import sys

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def make_data_txt_file(files, txt_filename):
    totality_darks_filename = []
    totality_darks_mean_values  =[]
    totality_darks_med_values   =[]
    totality_darks_stddev_values=[]
    totality_darks_max_values   =[]
    totality_darks_min_values  =[]
    file_number=[]
    i=0

    totality_dark_files = natural_sort(files)
    for tif in totality_dark_files:
        i+=1
        file_number.append(i)
        image = Image.open(tif)
        total_dark_array=np.array(image)
        total_dark_mean = np.mean(total_dark_array)
        total_dark_med = np.median(total_dark_array)
        total_dark_stddev = np.std(total_dark_array)
        total_dark_max = np.max(total_dark_array)
        total_dark_min = np.min(total_dark_array)
        
        totality_darks_filename.append(tif)
        totality_darks_mean_values.append(total_dark_mean)  
        totality_darks_med_values.append(total_dark_med)   
        totality_darks_stddev_values.append(total_dark_stddev)
        totality_darks_max_values.append(total_dark_max)   
        totality_darks_min_values.append(total_dark_min)   
        #print(image)
    #filename=totality_darks_filename[0][0:11]
    filename = txt_filename
    #print(filename)
    t=Table()
    t['#File_number'] = file_number
    t['Filename'] = totality_darks_filename
    t['Max Values (Analog-Digital units)'] = totality_darks_max_values
    t['Min Values (Analog-Digital units)'] = totality_darks_min_values
    t['Mean Values (Analog-Digital units)'] = totality_darks_mean_values
    t['Median Values (Analog-Digital units)'] = totality_darks_med_values
    t['STDDEV Values (Analog-Digital units)'] = totality_darks_stddev_values
    t.write('{}_totality_dark_statistics.txt'.format(filename),format='ascii.basic', overwrite=True)
    t.write('{}_totality_dark_statistics_neat.txt'.format(filename),format='ascii.fixed_width', overwrite=True)

def plot_statistics():
    #os.system('pwd')
    file_num_list = []
    image_num_list= []
    #os.system('mkdir bad_exptime')
    header_file = glob.glob('*_totality_dark_statistics.txt')
    for txt in header_file:
        with open(txt) as f:
            mean = np.loadtxt(txt, usecols=(4))
            median = np.loadtxt(txt, usecols=(5))
            stddev = np.loadtxt(txt, usecols=(6))
            filename = np.loadtxt(txt, dtype='str', usecols=(1))
            file_num_list = np.loadtxt(txt, usecols=(0))
        #print(len(mean), len(file_num_list))
        plt.scatter(file_num_list,mean)
        plt.plot(file_num_list,mean)
        plt.grid(color='b')
        plt.title('{} Mean vs Frame'.format(filename[0][:11]))
        plt.xlabel('Frame')
        plt.ylabel('Frame Mean (ADU)')
        plt.savefig('plot {} Frame Mean vs Frame.png'.format(filename[0][:11]))
        plt.show()
        plt.clf()

        
        plt.scatter(file_num_list,median)
        plt.scatter(file_num_list,mean,c='orange')
        plt.plot(file_num_list,median)
        plt.plot(file_num_list,mean,c='orange')
        plt.grid(color='b')
        plt.title('{} Median vs Frame'.format(filename[0][:11]))
        plt.xlabel('Frame')
        plt.ylabel('Frame Median (ADU)')
        plt.savefig('plot {} Frame Median vs Frame.png'.format(filename[0][:11]))
        plt.show()
        plt.clf()

def ds9_multi_exts (list_of_files):
    """
    Used to open a specfied extension of a fits file 

    """
    #files = [x + ext for x in list_of_files]
    #print(files)
    #files= ' '.join(files)
    #print(files)
    os.system('ds9 -zscale -zoom 0.5 {} &'.format(files))








def parse_args():
    """Parses command line arguments.

    Parameters:
        nothing

    Returns:
        args : argparse.Namespace object
            An argparse object containing all of the added arguments.

    Outputs:
        nothing
    """

    #Create help string:
    path_help = 'Path to the folder with files to run tweakreg.'
    filter_help = 'The filter that is being used'
    new_or_old_version_help= 'Are you using old or new idctab, npol and d2im files'
    # Add arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-path', dest = 'path', action = 'store',
                        type = str, required = True, help = path_help)
    # Parse args:
    args = parser.parse_args()


    return args
# -------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    path=args.path 
    os.chdir(path)
    files = glob.glob('*.tif')
    txt_filename = 'test'
    make_data_txt_file(files,txt_filename)
    plot_statistics()
    list_of_files=natural_sort(files)
    #for x in list_of_files:
        #print(list_of_files[0])
        #os.system('ds9 -tiff -zscale -zoom 0.5 {} &'.format(x))
    #ext1='[0]'
    #ds9_multi_exts(list_of_files,ext1)





