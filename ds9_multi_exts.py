import os
import glob
import argparse

def ds9_multi_exts (list_of_files,ext):
	"""
	Used to open a specfied extension of a fits file 

	"""
	files = [x + ext for x in list_of_files]
	#print(files)
	files= ' '.join(files)
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
    os.chdir(args.path)
    list_of_files = glob.glob('*.fits')
#    list_of_files = glob.glob('*_vs_*.fits')
    ext1='[1]'
    ds9_multi_exts(list_of_files,ext1)
    ext0='[4]'
    ds9_multi_exts(list_of_files,ext0)

