''' Script to load data in batches from h5 files and create training and validation datasets 
Replacement for CASE Data Generator and CASE data reader
Author: Aritra Bal
Date: 2rd July 2024'''


import tensorflow as tf
import numpy as np
import h5py
import os,glob
from collections import namedtuple
    
class JetDataset(tf.data.Dataset):
    def __new__(cls, filelist, batch_size=32, shuffle_buffer_size=1000,input_shape=(100,3)):
        return cls.from_generator(
            cls.generator,
            output_signature=(
                tf.TensorSpec(shape=(None,)+input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            ),args=(filelist, batch_size, shuffle_buffer_size, input_shape)
        )

    @staticmethod
    def generator(filelist:list[str], batch_size:int, shuffle_buffer_size:int,input_shape: tuple[int,int]):    
        def load_and_preprocess_file(file_path):
            with h5py.File(file_path, 'r') as file:
                jet1_pxpypz = file['jet1_PFCands'][:,:input_shape[0],:]
                jet2_pxpypz = file['jet2_PFCands'][:,:input_shape[0],:]
                truth_label = file['truth_label'][()]
            
            stacked_data = np.vstack([jet1_pxpypz, jet2_pxpypz])
            stacked_labels = np.concatenate([truth_label,truth_label],axis=0)
            return stacked_data, stacked_labels

        for file_path in filelist:
            data, labels = load_and_preprocess_file(file_path)
            
            # Shuffle the data from this file
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
            
            # Yield data in batches
            for i in range(0, len(data), batch_size):
                end=i+batch_size
                if end>len(data):
                    end=len(data)
                batch_data = data[i:end]
                batch_labels = labels[i:end]
                #import pdb;pdb.set_trace()
                #import pdb;pdb.set_trace()
                # Ensure the batch size is consistent
                yield batch_data, batch_labels


def create_training_datasets(directory:str, batch_size:int=32, shuffle_buffer_size:int=1000, train_split:float=0.8,input_shape:tuple[int,int]=(100,3)) -> tuple[JetDataset, JetDataset]:
    filelist = (glob.glob(os.path.join(directory,'train', '*.h5')))
    print(f"Found {len(filelist)} files in {os.path.join(directory,'train')}")
    train_files_count = int(len(filelist) * train_split)
    print(f"Using {train_files_count}/{len(filelist)} files for training")
    train_files= filelist[:train_files_count]
    valid_files= filelist[train_files_count:]
    # print('\n\n ##############################################  \n\n')
    # print('These files are used for training: ', [os.path.split(f)[-1] for f in train_files])
    # time.sleep(5)
    # print('\n\n ##############################################  \n\n')
    # print('These files are used for validation: ', [os.path.split(f)[-1] for f in valid_files])
    # time.sleep(5)
    train_dataset = JetDataset(train_files, batch_size, shuffle_buffer_size,input_shape)
    val_dataset = JetDataset(valid_files, batch_size, shuffle_buffer_size,input_shape)
    
    return train_dataset, val_dataset

def create_test_dataset(directory:str, batch_size:int=32, shuffle_buffer_size:int=1000,input_shape:tuple[int,int]=(10,3)) -> JetDataset:
    filelist = (glob.glob(os.path.join(directory,'test', '*.h5')))
    print(f"Found {len(filelist)} files in {os.path.join(directory,'test')}")
    
    test_dataset = JetDataset(filelist, batch_size, shuffle_buffer_size,input_shape)
    
    return test_dataset

def read_dataset(dirpath:str=None,event_wise:bool=False,input_shape:tuple[int,int]=(100,3),read_vae_loss:bool=False) -> tuple[np.ndarray, np.ndarray]:
    filelist=glob.glob(os.path.join(dirpath,'*.h5'))
    for i,filename in enumerate(filelist):
        with h5py.File(filename, 'r') as f:
            jet1_pxpypz = f['jet1_PFCands'][:,:input_shape[0],:]
            jet2_pxpypz = f['jet2_PFCands'][:,:input_shape[0],:]
            truth_label = f['truth_label'][()]
            if read_vae_loss:
                jet1_loss=f['jet_kinematics'][:,15]+0.5*f['jet_kinematics'][:,16]
                jet2_loss=f['jet_kinematics'][:,18]+0.5*f['jet_kinematics'][:,19]
        if i==0:
            stacked_data = np.vstack([jet1_pxpypz, jet2_pxpypz])
            stacked_labels = np.concatenate([truth_label,truth_label], axis=0)
            if read_vae_loss:
                stacked_loss = np.concatenate([jet1_loss,jet2_loss],axis=0)
            
        else:
            stacked_data = np.concatenate([stacked_data,np.vstack([jet1_pxpypz, jet2_pxpypz])],axis=0)
            stacked_labels=np.concatenate([stacked_labels,np.concatenate([truth_label,truth_label], axis=0)],axis=0)
            if read_vae_loss:
                stacked_loss=np.concatenate([stacked_loss,np.concatenate([jet1_loss,jet2_loss],axis=0)],axis=0)
    if read_vae_loss:
        return stacked_data,stacked_labels,stacked_loss
    return stacked_data, stacked_labels
def load_params(filepath):
    with h5py.File(os.path.join(filepath, 'model_params.h5'), 'r') as f:
        params_group = f['params']
        
        # Extract attribute names and values from the HDF5 file
        field_names = list(params_group.attrs.keys())
        field_values = [params_group.attrs[name] for name in field_names]

        # Dynamically create the named tuple class with extracted field names
        Params = namedtuple('Params', field_names)
        
        # Instantiate the named tuple with the values from the HDF5 file
        params_instance = Params(*field_values)
        return params_instance
# Usage example

def get_label_and_losses(signal='XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow'):
    base_dir='/storage/9/abal/CASE/VAE_results/events/run_141098/'
    signal_dir=os.path.join(base_dir,signal+"_RECO",'nominal')
    qcd_dir=os.path.join(base_dir,'qcd_data_SR_Oz_RECO')
    qcd_filelist=glob.glob(os.path.join(qcd_dir,'*.h5'))
    signal_filelist=glob.glob(os.path.join(signal_dir,'*.h5'))
    filelist=signal_filelist+qcd_filelist
    stacked_jet_loss=[]
    stacked_labels=[]
    for file in filelist:
        signal=False
        if file in signal_filelist:
            signal=True
        with h5py.File(file, 'r') as f:
            j1Loss=f['eventFeatures'][:,15]+0.5*f['eventFeatures'][:,16]
            j2Loss=f['eventFeatures'][:,18]+0.5*f['eventFeatures'][:,19]
        jet_loss=np.concatenate([j1Loss,j2Loss],axis=0)
        if signal:
            jet_label=np.ones_like(jet_loss)
        else:
            jet_label=np.zeros_like(jet_loss)
        stacked_jet_loss.append(jet_loss)
        stacked_labels.append(jet_label)
    stacked_jet_loss=np.concatenate(stacked_jet_loss,axis=0)
    stacked_labels=np.concatenate(stacked_labels,axis=0)
    indices = np.arange(stacked_jet_loss.shape[0])
    np.random.shuffle(indices)
    stacked_jet_loss = stacked_jet_loss[indices]
    stacked_labels = stacked_labels[indices]
    return stacked_jet_loss,stacked_labels

if __name__ == "__main__":
        
    directory = '/ceph/abal/CASE/supervised_training_data/XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow/'
    batch_size = 1024
    train_dataset, val_dataset = create_training_datasets(directory, batch_size)

    # Iterate through the datasets
    for inputs, targets in train_dataset.take(1):
        print("Training batch - Inputs shape:", inputs.shape, "Targets shape:", targets.shape)
        #import pdb;pdb.set_trace()
    for inputs, targets in val_dataset.take(1):
        print("Validation batch - Inputs shape:", inputs.shape, "Targets shape:", targets.shape)




    