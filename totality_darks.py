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
"""
Sorts totality dark files in numerical order (1-100) 

""" 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def make_data_txt_file(totality_dark_files, txt_filename):
""" 
calculate the mean,standard deviation and median of darks files and stores the data into 2 .txt files for observer and code. 


"""
    totality_darks_filename = []
    totality_darks_mean_values  =[]
    totality_darks_med_values   =[]
    totality_darks_stddev_values=[]
    totality_darks_max_values   =[]
    totality_darks_min_values  =[]
    file_number=[]
    i=0

    totality_dark_files = natural_sort(totality_dark_files)
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
    t.write('{}__totality_dark_statistics_neat.txt'.format(filename),format='ascii.fixed_width', overwrite=True)

def plot_dark_statistics():
"""
Uses the data from the .txt files made perviously to plot the frome value vs. frame and saves the plot as .png file. 
"""
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
        plt.savefig('Frame Mean vs Frame.png')
        plt.show()
        plt.clf()

        #plt.scatter(file_num_list,median)
        #plt.grid(color='b')
        #plt.title('{} data Median vs filenumber'.format(filename[0]))
        #plt.xlabel('File Number')
        #plt.ylabel('Median (ADU)')
        #plt.savefig('Median vs number.png')
        #plt.show()
        #plt.clf()


def seperate_exp():

"""
Seperates the dark frames into 8 different directory by file number to repersent the different exposures 

Ex. 1st frame ---> exp1
    2nd frame ---> exp2
    etc. 
"""
    #os.system('rm -r exp1 1_3ms_exp 4ms_exp 13_exp 40_exp 130_exp 400_exp 1300_exp')
    #os.system('mkdir exp1 1_3ms_exp 4ms_exp 13_exp 40_exp 130_exp 400_exp 1300_exp')
    os.system('rm -r exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8')
    os.system('mkdir exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8')
    header_file = glob.glob('*_totality_dark_statistics.txt')
    for txt in header_file:
        with open(txt) as f:
            filename = np.loadtxt(txt, dtype='str', usecols=(1))
            filenum = np.loadtxt(txt, usecols=(0))

    for image,num in zip(filename,filenum):
        n = num
        m = (n-1)%8
        print(image)
        if m == 0:
            print ('Moving {} to exp1_dir'.format(image))
            os.system('cp {} exp1/'.format(image))
              
        if m == 1:
            print ('Moving {} to exp2_dir'.format(image))
            os.system('cp {} exp2/'.format(image))
            
        if m == 2:
            print ('Moving {} to exp3_dir'.format(image))
            os.system('cp {} exp3/'.format(image))
        
        
        if m == 3:
            print ('Moving {} to exp4_dir'.format(image))
            os.system('cp {} exp4/'.format(image))
            
        if m == 4:
            print ('Moving {} to exp5_dir'.format(image))
            os.system('cp {} exp5/'.format(image))
            
            
        if m == 5:
            print ('Moving {} to exp6_dir'.format(image))
            os.system('cp {} exp6/'.format(image))
            
        if m == 6:
            print ('Moving {} to exp7_dir'.format(image))
            os.system('cp {} exp7/'.format(image))
        
        
        if m == 7:
            print ('Moving {} to exp8_dir'.format(image))
            os.system('cp {} exp8/'.format(image))
            
def convert_tif_to_fits(list_of_tif):

"""
Converts the tif files into .fits.
"""
    for image in list_of_tif:
        im = Image.open(image)
        dark_image_array=np.array(im)
        #print(total_array.shape)
        #print(image[:-4])
        new_fits = fits.HDUList()
        new_fits.append(fits.ImageHDU(dark_image_array))
        new_fits.writeto('{}.fits'.format(image[:-4]),overwrite=True)


def mean_stack(j):
    """This is a function to create a mean stack of a data cube through the z axis. It will print the column it is currently working on.

    mean_stack1(j)
    Parameters:
        j: int
            The number of columns to mean stack together at a time


    """
    col_mean=np.nanmean(data_cube[:,:,j],axis=0)
    #print(j)
    return col_mean




def stacking(dark_fits):

"""
Stacks the dark frames by column and outputs the result as a .fits file
"""
    i=0
    for dark_im in dark_fits:
        #print(dark_im)
        h = fits.open(dark_im)
        #h.info()
        sci_chip=h[0].data
        h.close()
    
    
    #Inputs the masked data in to the data cube
        data_cube[i] = sci_chip
        i+=1
    #Using parellel computing to median stack the columns for faster results (Do not use 8 when running the locally)    
    p=Pool(9)
    result = p.map(mean_stack,range(2448))
    
    
    #Puts the final 2d list into a numpy array
    result=np.array(result)
    
    #The data deminsions changes when the median stack occures so we transpose the data to re shape the array
    result =np.transpose(result)
    #print(result.shape)
    #print(darkim)
    stacked_hdul = fits.HDUList()
    stacked_hdul.append(fits.ImageHDU(result))
    stacked_hdul.writeto('sd_cate*darkexp.fits'.format(dark_im[:10]),overwrite=True)
    stacked_hdul.close() 

def superdark_stats_plots(superdarks):

"""
Plots the the mean of the stacked files(superbias) vs. frame number and makes a .txt file with the statistics for each file.

The plots are used to identify the the lowest mean value because idealy that would reperesent the starting exposure.  
"""
    superdark_filename = []
    superdark_mean_values  =[]
    superdark_med_values   =[]
    superdark_stddev_values=[]
    superdark_max_values   =[]
    superdark_min_values  =[]
    file_number=[]
    i=0

    superdarks = natural_sort(superdarks)
    for sd in superdarks:
        i+=1
        file_number.append(i)
        #image = Image.open(sd)
        hdu = fits.open(sd)
        sd_array = hdu[0].data
        #sd_array=np.array(image)
        sd_mean = np.mean(sd_array)
        sd_med = np.median(sd_array)
        sd_stddev = np.std(sd_array)
        sd_max = np.max(sd_array)
        sd_min = np.min(sd_array)
        
        superdark_filename.append(sd)
        superdark_mean_values.append(sd_mean)  
        superdark_med_values.append(sd_med)   
        superdark_stddev_values.append(sd_stddev)
        superdark_max_values.append(sd_max)   
        superdark_min_values.append(sd_min)   
        #print(image)
    filename=superdark_filename[0][:12]
    #print(filename)
    t=Table()
    t['#File_number'] = file_number
    t['Filename'] = superdark_filename
    t['Max Values (Analog-Digital units)'] = superdark_max_values
    t['Min Values (Analog-Digital units)'] = superdark_min_values
    t['Mean Values (Analog-Digital units)'] = superdark_mean_values
    t['Median Values (Analog-Digital units)'] = superdark_med_values
    t['STDDEV Values (Analog-Digital units)'] = superdark_stddev_values
    t.write('{}_totality_dark_statistics.txt'.format(filename),format='ascii.basic', overwrite=True)
    t.write('{}__totality_dark_statistics_neat.txt'.format(filename),format='ascii.fixed_width', overwrite=True)



def superdark_name_change():

"""
Once the user has idntified the starting exposure they will be ask to submit the number and name of the site. The function will
change the names to the supeerdark files to the correct name.
"""

    start = input("What number is the starting image(1-8)? ")
    start = int(start)
    sitename = input("What is the sitename(cate17-[])?")
    sitename=str(sitename)
#    site_dir = os.getcwd()
#    site_name=site_dir[34:]
#    site_name=site_name[:-30]
#    print(site_name)


    if start == 1:
        print('Started at the 0.4ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_0_4ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_1_3ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_4ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_13ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_40ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_130ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_400ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_1300ms.fits'.format(sitename))        

    elif start == 2:
        print('Started at the 1.3ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_1_3ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_4ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_13ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_40ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_130ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_400ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_0_4ms.fits'.format(sitename))     

    elif start == 3:
        print('Started at the 4.0ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_4ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_13ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_40ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_130ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_400ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_0_4ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_1_3ms.fits'.format(sitename))     
    
    elif start == 4:
        print('Started at the 13.0ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_13ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_40ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_130ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_400ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_0_4ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_1_3ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_4ms.fits'  .format(sitename))     
    
    elif start == 5:
        print('Started at the 40ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_40ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_130ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_400ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_0_4ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_1_3ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_4ms.fits'   .format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_13ms.fits'  .format(sitename))     
    
    elif start == 6:
        print('Started at the 130ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_130ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_400ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_0_4ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_1_3ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_4ms.fits'   .format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_13ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_40ms.fits'  .format(sitename))     

    elif start == 7:
        print('Started at the 400ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_400ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_0_4ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_1_3ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_4ms.fits'   .format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_13ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_40ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_130ms.fits'  .format(sitename))     

    elif start == 8:
        print('Started at the 1300ms exposure')
        os.system('mv sd_cate*darkexp1.fits {}_superdark_1300ms.fits'.format(sitename))
        os.system('mv sd_cate*darkexp2.fits {}_superdark_0_4ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp3.fits {}_superdark_1_3ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp4.fits {}_superdark_4ms.fits'   .format(sitename))
        os.system('mv sd_cate*darkexp5.fits {}_superdark_13ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp6.fits {}_superdark_40ms.fits'  .format(sitename))
        os.system('mv sd_cate*darkexp7.fits {}_superdark_130ms.fits' .format(sitename))
        os.system('mv sd_cate*darkexp8.fits {}_superdark_400ms.fits' .format(sitename))     

#def superdark_plot(superdark_files):
#    frame_mean_list = []
#    frame_med_list = []
#    frame_stddev_list = []
#    frame_max_list = []
#    frame_min_list = []
#    filename_list = []
#    frame_number = []
#    i = 0
#
#    for i in superdark_files:
#        i += 1
#        print(i)
#        hdu = fits.open(i)
#        sci_data = hdu[1].data
#        filename = i[:12]
#        frame_mean = np.mean(sci_data)
#        frame_med = np.median(sci_data)
#        frame_stddev = np.std(sci_data)
#        frame_max = np.max(sci_data)
#        frame_min = np.min(sci_data)
#
#        filename_list.append(filename)
#        frame_mean_list.append(frame_mean)
#        frame_med_list.append(frame_med)
#        frame_stddev_list.append(frame_stddev)
#        frame_max_list.append(frame_max)
#        frame_min_list.append(frame_min)

def plot_superdark_statistics():

"Replots the supdarks mean values vs. frame"
    #os.system('pwd')
    file_num_list = []
    image_num_list= []
    #os.system('mkdir bad_exptime')
    header_file = glob.glob('cate*_totality_dark_statistics.txt')
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
        plt.savefig('Frame Mean vs Frame.png')
        plt.show()
        plt.clf()
        

def plot_darksets_statistics():
"""
Make statistics table and plots the frame mean vs. frame for the individual exposure sets.
"""    
    #os.system('pwd')
    file_num_list = []
    image_num_list= []
    os.getcwd()
    #os.system('mkdir bad_exptime')
    header_file = glob.glob('*_totality_dark_statistics.txt')
    for txt in header_file:
        with open(txt) as f:
            mean = np.loadtxt(txt, usecols=(4))
            median = np.loadtxt(txt, usecols=(5))
            stddev = np.loadtxt(txt, usecols=(6))
            filename_list = np.loadtxt(txt, dtype='str', usecols=(1))
            file_num_list = np.loadtxt(txt, usecols=(0))
        #print(len(mean), len(file_num_list))
    dirname=os.getcwd()
    print(dirname)
    #print(dirname[:-4])
    if dirname == '{}exp1'.format(dirname[:-4]):
        filename = 'exposure1'
    elif dirname == '{}exp2'.format(dirname[:-4]):
        filename = 'exposure2'
    elif dirname == '{}exp3'.format(dirname[:-4]):
        filename = 'exposure3'
    elif dirname == '{}exp4'.format(dirname[:-4]):
        filename = 'exposure4'
    elif dirname == '{}exp5'.format(dirname[:-4]):
        filename = 'exposure5'
    elif dirname == '{}exp6'.format(dirname[:-4]):
        filename = 'exposure6'
    elif dirname == '{}exp7'.format(dirname[:-4]):
        filename = 'exposure7'
    elif dirname == '{}exp8'.format(dirname[:-4]):
        filename = 'exposure8'

    filename1 = filename
    plt.scatter(file_num_list,mean)
    plt.plot(file_num_list,mean)
    plt.grid(color='b')
    plt.title('{} Mean vs Frame'.format(filename1))
    plt.xlabel('Frame')
    plt.ylabel('Frame Mean (ADU)')
    plt.savefig('{} Frame Mean vs Frame.png'.format(filename1))
    #plt.show()
    plt.clf()
        










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
os.system('mkdir totality_dark')
os.system('mv *_totaldarks*.tif totality_dark')
os.system('pwd')
os.chdir('{}totality_dark'.format(path))
os.system('rm -r superdark')
os.system('mkdir ../superdark')
os.system('pwd')
i = 1
totality_dark_files = glob.glob("*_totaldarks*.tif")
txt_filename = totality_dark_files[0][:12]
make_data_txt_file(totality_dark_files, txt_filename)
plot_dark_statistics()
txt = input("Do you want to continue on to stacking the files?(yes/no or y/n)")
#print(sys.argv)
txt = str(txt)

if txt == 'yes' or txt=='y':
    print('---------------Seperating files------------------')
    seperate_exp()
    dark_dir = '{}/totality_dark'.format(path)
    for subdir, dirs, files in os.walk(dark_dir):
        for dir in sorted(dirs):
            path1=os.path.join(subdir,dir)
            print(path1)
            os.chdir(path1)
            print('-------------Converting .tif files to .fits-------------')
            list_of_tif = glob.glob('*.tif')
            convert_tif_to_fits(list_of_tif)
            dark_fits = glob.glob('*.fits')
            make_data_txt_file(list_of_tif, list_of_tif[0][:12])
            plot_darksets_statistics()
            #print(dark_fits)
            hdr = fits.getheader(dark_fits[0])
            nx = hdr['NAXIS1']
            ny = hdr['NAXIS2']
            nf = len(dark_fits)
            #print(nx,ny,nf)
            data_cube = np.zeros((nf, ny, nx), dtype=float)
            print('-----------Making Superdark files---------------')
            stacking(dark_fits)
            os.system('mv sd_cate*darkexp.fits sd_cate*darkexp{}.fits'.format(i))
            os.system('cp sd*.fits ../../superdark')
            i+=1
    
    os.system('mv {}/superdark {}/totality_dark/'.format(path, path))
    os.chdir('{}totality_dark/superdark'.format(path))
    os.system('pwd')
    superdarks = glob.glob("*.fits")
    superdark_stats_plots(superdarks)
    plot_dark_statistics()
    superdark_name_change()
    superdarks = glob.glob("*.fits")
    superdark_stats_plots(superdarks)
    plot_superdark_statistics()

    #superdarks_newname = glob.glob("*.fits")
    #superdark_name_change()
    #txt_filename = superdarks_newname[0][:12]
    #make_data_txt_file(superdarks_newname, txt_filename)

    

elif txt == 'no' or txt=='n':
    print ('Another problem -____-........ go fix it')






