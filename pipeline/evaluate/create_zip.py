from zipfile import ZipFile 
import os 
import shutil
  
def get_all_file_paths(directory): 
  
    # initializing empty file paths list 
    file_paths = [] 
  
    # crawling through directory and subdirectories 
    for root, directories, files in os.walk(directory): 
        for filename in files: 
            # join the two strings in order to form the full filepath. 
            filepath = os.path.join(root, filename) 
            file_paths.append(filepath) 
  
    # returning all file paths 
    return file_paths         
  
def main(config): 
    # path to folder which needs to be zipped 
    directory = config['submit_path']
  
    # calling function to get all file paths in the directory 
    file_paths = os.listdir(directory)
  
    # printing the list of all files to be zipped 
    
  
    os.chdir(directory) 
    
    with ZipFile('submit.zip','w') as zip: 
        # writing each file one by one 
        for file in file_paths:
            zip.write(file) 
  
    os.chdir(config['evaluate_path'])
    shutil.move(directory+'/submit.zip','submit.zip')
    os.chdir(config['parent_path'])

  
  
if __name__ == "__main__": 
    main()