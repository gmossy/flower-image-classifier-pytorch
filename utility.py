# Get a random flower fullfilename from the a data directory 
import random
import os, sys
import logging

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths 

def pick_random_flower():
    # generate random integer
    ridx = random.randint(0,102)  #102 is the number of different classes of flowers
    return ridx

def get_randomimage(ridx, data_directory):
    # take the random number and data dir & index into it randomly, output the image_path (filename) of a flower in the test dir
    get_filepaths(data_directory)
    full_file_paths = get_filepaths(data_directory)
    image_path = full_file_paths[ridx]
    return image_path

rand_value = pick_random_flower()
#print(rand_value)
#image_path = get_randomimage(rand_value, test_dir)
#print(image_path)
# Run the above function and store its results in a variable.   
#full_file_paths = get_filepaths(train_dir)
#print(full_file_paths[rand_value])

def mylogger(LOG_FILENAME = 'log', level='NEWLOG'):
    if level=='APPEND':
        f = open(LOG_FILENAME, 'a')
        f.write("log")
    if level=='NEWLOG':
        f = open(LOG_FILENAME, 'w')
        f.write("log")
    if level=='DEBUG':
        DEBUG = True
    return
def close_log():
    f.close
    return

    # pause 
def pause():
    yn = str(input("Would you like to continue?  "))
    if yn == 'y' or 'Y' or 'YES' or 'yes': l = 0
    else: sys.exit()
    return l
def ask_user():
    # 
    print("Would you like to continue? Y/N ")
    input = sys.stdin.readline()
    return input
        

