# +
from osfclient.api import OSF
import os
import zipfile

# Set up folders
base_path = os.getcwd()  # ~/scratch/anphy_sleep_data
patient_folder = os.path.join(base_path, "patient_records")
os.makedirs(patient_folder, exist_ok=True)  # create folder if it doesn't exist

# Connect to OSF
osf = OSF()
project = osf.project('r26fh')

for storage in project.storages:
    print(f"Accessing storage: {storage.name}")
    for file in storage.files:
        filename = file.name
        
        # Only download EPCTL *.zip files
        if filename.startswith("EPCTL") and filename.endswith(".zip"):
            local_path = os.path.join(patient_folder, filename)
            
            print(f"Downloading {filename} to {local_path}...")
            with open(local_path, 'wb') as f:
                file.write_to(f)
            print("Download complete.")
            
            # Unzip the file
            print(f"Unzipping {filename}...")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(patient_folder)
            print("Unzip complete.")
            
            # Remove the original zip file to save space
            os.remove(local_path)
            print(f"{filename} removed.")
