import os
import argparse
import glob


def make_totality():

	dir_list = glob.glob('cate*')
	# Making directory for the 
	for dir,files in dir_list:
		os.system ('pwd')
		print('Making dir ./totality/{}'.format(dir))
		os.system('mkdir ./totality_data/{}'.format(dir))

def move_totality():
	totality = glob.glob('./*/*_total_*')
	os.system('mkdir totality_data')
	for im in totality:
		print(im)
		os.system('mv {} totality_data'.format(im))
		#files=glob.glob('')











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
    path=args.path
    os.chdir(path)
    os.system('mkdir zzz_derp_dir')
    #make_totality()for subdir, dirs, files in os.walk(path1):
    for subdir, dirs, files in os.walk(path):
    	for dir in dirs:
    	    path=os.path.join(subdir,dir)
    	    os.chdir(path)
    	    move_totality()
    	    #os.system('pwd')
    	    #os.system('mkdir totality_data')
    		#os.system('mkdir {}_totality_data'.format)
		    #os.chdir(dir_path)


