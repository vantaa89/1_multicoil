import random
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
from scipy.ndimage import rotate as rot
from copy import deepcopy
import matplotlib.pyplot as plt

class MotionParameters:
    def __init__(self, img_size, SNR, lb, ub, PE_ran, rot = 0.5, flip=0):
        self.img_size = img_size
        
        assert len(SNR) == 2 and SNR[0] <= SNR[1]
        self.snr_min, self.snr_max = SNR[0], SNR[1]
        
        assert lb <= ub
        self.m_lb, self.m_ub = lb, ub
        
        assert len(PE_ran) == 2 and PE_ran[0] <= PE_ran[1]
        self.PE_lb, self.PE_ub = PE_ran
        
        self.rot = rot
        self.flip = flip
    
    @staticmethod
    def random_between(lower, upper):
        return (lower + (upper - lower) * random.random()) * (-1 if random.random() > 0.5 else 1)
    
    def get_parameters(self):
        PE_occurred = random.randint(self.PE_lb, self.PE_ub)
        trans_0 = MotionParameters.random_between(self.m_lb[0], self.m_ub[0])
        trans_x_diff = MotionParameters.random_between(self.m_lb[1], self.m_ub[1])
        trans_y_diff = MotionParameters.random_between(self.m_lb[2], self.m_ub[2])
        
        rot_angle = 0
        if self.rot > 0:
            rot_angle = (random.random() - 0.5) * 2 * self.rot
        
        flip = (random.random() < self.flip)
        
        determined_params = {
            "PE_occurred": PE_occurred,
            "trans_dir": (trans_0, trans_x_diff, trans_y_diff),
            "rot_angle": rot_angle, 
            "flip": flip, 
            "SNR_range": (self.snr_min, self.snr_max)
        }
        
        return determined_params
    

    
class SimulatedDataset(Dataset):
    '''
    Dataset class that provides simulated motion-corrupted DICOM MR images

    Args:
        image_path (str): Path to the folder that contains uncombined multicoil MR dataset in (patient number).mat format
        sensmap_path (str): Path to the folder that contains the ESPIRiT-generated sensitivity map uncombined multicoil MR dataset in (patient number).mat format. the patient number must match with those in image_path.
        params (MotionParameters): Parameters related to the motion
        multiple (int): The number of times the same data is repeatedly used. In each repeat, the generated data is different because of the random motion parameters.
        slices (int): The number of slices contained in the single .mat file.
    '''
    def __init__(self, image_path: str, sensmap_path: str, params: MotionParameters, split='train', multiple=10, slices=16):
        

        self.slices = slices
        self.image_paths = []
        self.sensmap_paths = []
        self.multiple = multiple
        self.splits = {
            'train': [(str(i) + ".mat") for i in range(0, 10)],
            'test': [(str(i) + ".mat") for i in range(10, 11)],
        }
        assert split in self.splits.keys()
        
        image_files = os.listdir(image_path)
        sensmap_files = os.listdir(sensmap_path)
        for image_file in image_files:
            if image_file in self.splits[split]:
                self.image_paths.append(os.path.join(image_path, image_file))
        for sensmap_file in sensmap_files:
            if sensmap_file in self.splits[split]:
                self.sensmap_paths.append(os.path.join(sensmap_path, sensmap_file))
        self.image_paths.sort()
        self.sensmap_paths.sort()
        self.len = len(self.image_paths)
        
        self.params = deepcopy(params)
        self.img_size = self.params.img_size
        
    def __getitem__(self, idx):
        # out dataset dimension: 224 x 224 x 16 x 7 x 32
        image = loadmat(self.image_paths[idx%self.len])['uncomb_img'][:,:,:,0,:]              # leave only the first echo
        sens_map = loadmat(self.sensmap_paths[idx%self.len])['sensitivity'][:,:,:,0,:]        
        
        assert len(image.shape) == 4                      # 4D, (224, 224, 16, 32). 32: coil
        dicom_image = (image * np.conj(sens_map)).sum(axis=-1) # (224, 224, 16)
        
        num_layers = dicom_image.shape[-1]
        PE_occurred = []
        image_kspace = np.fft.fftn(image, axes=(0, 1))
        
        
        corrupted_multicoil_kspace = np.zeros_like(image_kspace)
        
        for i in range(num_layers): # apply different motion parameters to each coil
            determined_motion_params = self.params.get_parameters()
            PE_occurred.append(determined_motion_params['PE_occurred'])

            postmotion_dicom = self.apply_motion(deepcopy(dicom_image[:,:,i]), determined_motion_params) # (224, 224)
            postmotion_multicoil = np.einsum('xy,xyc->xyc', postmotion_dicom, sens_map[:,:,i,:]) # (224, 224, 32). single layer
            postmotion_multicoil_kspace = np.fft.fftn(postmotion_multicoil, axes=(0, 1))
            
            corrupted_multicoil_kspace[:PE_occurred[i],:,i,:] = image_kspace[:PE_occurred[i],:,i,:]
            corrupted_multicoil_kspace[PE_occurred[i]:,:,i,:] = postmotion_multicoil_kspace[PE_occurred[i]:,:,:]
            
        corrupted_dicom_kspace = np.fft.fftn((np.conjugate(sens_map) * np.fft.ifftn(corrupted_multicoil_kspace, axes=(0, 1))).sum(axis=-1), axes=(0, 1))   # multiplying conjuated sens_map and take coil-sum
        
        # add noise
        snr_min, snr_max = determined_motion_params['SNR_range']
        if(snr_max > 0):
            SNR = snr_min + (snr_max - snr_min) * random.random()
            mean_intensity = np.mean(np.abs(corrupted_dicom_kspace))
            noise_std = mean_intensity / SNR
            corrupted_dicom_kspace += (np.random.normal(0,noise_std,size=(corrupted_dicom_kspace.shape)) + 1j* np.random.normal(0,noise_std,size=(corrupted_dicom_kspace.shape)))
        
        return corrupted_dicom_kspace, PE_occurred        # , postmotion_dicom, dicom_image
        
    def apply_motion(self, image, determined_params: dict):
        
        # load parameters
        PE_occurred = determined_params['PE_occurred'] # unused
        trans_0, trans_x_diff, trans_y_diff = determined_params['trans_dir']
        rot_angle = determined_params['rot_angle']
        flip = determined_params['flip']
        snr_min, snr_max = determined_params['SNR_range']
        
        # apply rotation and flip
        image = rot(np.real(image),rot_angle,reshape=False)+rot(np.imag(image),rot_angle,reshape=False)*1j
        if flip:
            image = image[:,::-1]
        
        # translation
        trans_vec = np.zeros((self.img_size[1], 3))
        trans_vec[:,0] = trans_0
        trans_vec[:,1] = trans_x_diff
        trans_vec[:,2] = trans_y_diff
        
        phase_vec_x = np.zeros((self.img_size[1],1))
        phase_vec_y = np.zeros((self.img_size[1],1))

        phase_vec_x[:,0] = trans_0 + np.arange(0,self.img_size[1],1) * trans_x_diff
        phase_vec_y[:,0] = np.arange(-self.img_size[1]//2+1,self.img_size[1]//2+1,1) * trans_y_diff
        
        phase = np.ones(image.shape)
        phase = phase * np.exp(1j * phase_vec_x.transpose(1,0)) # x phase
        phase[:,:] = phase[:,:] * np.exp(1j * phase_vec_y ) # y phase
        #after_k_data = np.zeros(k_data.shape)
        #after_k_data[:,PE_occurred:] = k_data[:,PE_occurred]
        
        k_data = np.fft.fftn(image, axes=(0,1)) * np.fft.ifftshift(phase, axes=(0,1))
        return np.fft.ifft2(k_data)
        
    def __len__(self):
        return self.len * self.multiple