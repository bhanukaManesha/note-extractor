
import os
from os import listdir
from os.path import isfile, join
import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush() 

if __name__ == "__main__":
    if len(sys.argv) != 4:
       print("Please input the args as <folder_path> <file_type> <resume_from>")
       exit()
    else:

        # Get the folder path
        folderpath = sys.argv[1]

        # Move into the directory
        os.chdir(folderpath)

        # Get the output file type
        filetype = sys.argv[2]
        len_file_type = len(filetype)

        # Get resume from value
        resume_from = int(sys.argv[3])

        # Get the names of all the files
        files = [f for f in listdir(".") if isfile(join(".", f))]

        # Get the total files
        total_files = len(files)

        # Calculate the start point
        c = 0
        if resume_from != -1:
            c = resume_from

        for f in files:

            x = int(f[:len(filetype)])

            if x >= resume_from :
                # Update the progress
                progress(c, total_files)

                # Get the name of the new file
                new_file = str(c)+ filetype

                # Rename the file
                os.rename(f,new_file)
                
                # Increment the count
                c += 1

        print("Done! Please wait a while until the finder updates the changes")
