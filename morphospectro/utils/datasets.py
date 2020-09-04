import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import skimage.io
import pandas as pd

class GalaxyDataset(Dataset):
    def __init__(self,spectra_file,image_folder,debug = False,keys_spectra=["flux","obj_ids"],size=None,idx_start=0):
        """
        A pytorch dataset class used for training a neural network. Will automatically generate cutout photometry images based on ra,dec.
        spectra_file: str
            A filepath towards an h5 file containg the spectra.
            
        image_folder: str
            A filepath towards a directory in which images are found. If not existing will be created
            
        keys_spectra: list
            list/tupple where keys[0] is the header name for fluxes and keys[1] for obj_ids
        
        """
        self.spectra_file = spectra_file
        self.image_folder = image_folder
        self.flux,self.obj_ids = self.load_spectra(self.spectra_file,keys_spectra)
        self.size = size
        self.idx_start = idx_start
        self.transform = transforms.Compose([
            torchvision.transforms.ToPILImage(),
            #torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            torchvision.transforms.Resize(size=[32,32]),
            #torchvision.transforms.RandomResizedCrop(size=32,scale=(0.6,1)),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.RandomRotation(180),
            #torchvision.transforms.Resize(size=[28,28]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        )
        self.debug = debug

        
    def load_spectra(self,spectra_file,keys=["flux","obj_ids"]):
        flux = pd.read_hdf(spectra_file,key=keys[0])
        obj_ids = pd.read_hdf(spectra_file,key=keys[1])
        return flux,obj_ids
        
    def download_cutout(self,ra,dec,pixscale=0.1,savepath=None):
        """downloads a cutout image."""
        url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer=sdss&pixscale={pixscale}"
        if self.debug is True:
            print(f"url is {url}")
        img = skimage.io.imread(url)
        if savepath:
            skimage.io.imsave(savepath,img)
        return img

    def get_spectra(self,idx):
        return torch.FloatTensor(self.flux.loc[idx].values)
    
    def get_params(self,idx):
        ra = self.obj_ids.loc[idx]["ra"]
        dec = self.obj_ids.loc[idx]["dec"]
        dr8objid = int(self.obj_ids.loc[idx]["dr8objid"])
        return ra,dec,dr8objid
    
   
    def get_image(self,idx,pixscale=0.1):
        ra,dec,objid = self.get_params(idx)
        image_name = f"{objid}.jpg"
        image_path = os.path.join(self.image_folder,image_name)

        try:
            return skimage.io.imread(image_path)
        except:
            print("failed")
            return self.download_cutout(ra,dec,savepath=image_path)
   
    def __len__(self):
        if self.size is None:
            return len(self.flux)
        else:
            return self.size
    
    
    def __getitem__(self,idx):
        idx = self.idx_start+idx
        spectra = self.get_spectra(idx)
        image = self.get_image(idx)
        image = self.transform(image)

        return spectra,image,idx

    

    
    
    
