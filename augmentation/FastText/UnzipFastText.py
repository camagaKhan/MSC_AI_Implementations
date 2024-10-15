from zipfile import ZipFile
import os 

path = r"C:\\Users\\liamm\\OneDrive\\Documents\\GitRepositories\\Thesis\\Augmentation"
model_path = os.path.join(path, "crawl-300d-2M-subword.zip") # this is the model path
model_output_path = path  

with ZipFile(model_path, "r") as zObject:
    # ZipFile Will attempt to extract the 
    # contents of the zipped file.
    # Notice the model_output_path; this will
    # store the fast text model in the directory provided
    zObject.extractall(path=model_output_path)
    print('Success!')