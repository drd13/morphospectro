import torch torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import skimage.io
import pandas as pd

class GalaxyDataset(Dataset):
    def __init__(self,spectra_file,image_folder):
        """
        A pytorch dataset class used for training a neural network. Will automatically generate cutout photometry images based on ra,dec.
        spectra_file: str
            A filepath towards an h5 file containg the spectra.
            
        image_folder: str
            A filepath towards a directory in which images are found. If not existing will be created
        """
        self.spectra_file = spectra_file
        self.image_folder = image_folder
        self.flux,self.obj_ids = self.load_spectra(self.spectra_file)

        
    def load_spectra(self,spectra_file):
        flux = pd.read_hdf(spectra_file,key="flux")
        obj_ids = pd.read_hdf(spectra_file,key="obj_ids")
        return flux,obj_ids
        
        
    def download_cutout(self,ra,dec,savepath,pixscale=0.3):
        """downloads a cutout image."""
        url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer=sdss&pixscale={pixscale}"
        img = skimage.io.imread(url)
        skimage.io.imsave(savepath,img)

    def get_spectra(self,idx):
        return torch.FloatTensor(flux.loc[idx].values)
    
    def get_params(self,idx):
        ra = obj_ids.loc[idx]["ra"]
        dec = obj_ids.loc[idx]["dec"]
        dr8objid = int(obj_ids.loc[idx]["dr8objid"])
        return ra,dec,dr8objid
   
    def get_image(self,idx):
        ra,dec,objid = self.get_params(idx)
        image_name = f"{objid}.jpg"
        image_path = os.path.join(self.image_folder,image_name)

        try:
            return skimage.io.imread(image_path)
        except:
            print("failed")
            return self.download_cutout(ra,dec,image_path)
   
    def __len__(self):
        return len(fluxes)
    
    
    def __getitem__(self,idx):
        spectra = self.get_spectra(idx)
        image = self.get_image(idx)

        return spectra,image,idx
