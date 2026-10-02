import glob
import os


sdf_files = glob.glob('*.sdf')

for sdf_file in sdf_files:
    num = int(sdf_file.split('.')[0])
    if int(num) >= 39:
        num += 7
    elif 30 <= num <= 38:
        num += 3
    else:
        num += 1
    os.system('obabel ' + sdf_file + ' -O ' + str(num) + '_charged.sdf' + ' -p 7.4')
