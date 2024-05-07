# Script to compare file names from two directories
import os

dir1 = 'output_img'
dir2 = '../trainval/test/OUT'

dir1_files = os.listdir(dir1)
dir2_files = os.listdir(dir2)

files1_2 = [] # Files in dir1 but not in dir2
files2_1 = [] # Files in dir2 but not in dir1

for file in dir1_files:
    if file not in dir2_files:
        files1_2.append(file)   
        
for file in dir2_files:
    if file not in dir1_files:
        files2_1.append(file)
        
if len(files1_2) == 0 and len(files2_1) == 0:
    print('All files in both directories are the same.')
