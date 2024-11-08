import os,glob,tqdm
import h5py,pathlib
import numpy as np
import argparse
    
def process_and_save_h5_files(output_dir,signal='XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow', batch_size=5000,sideband=False,read_n=1e4,from_reco=False):
    # Initialize lists to store concatenated arrays
    all_jet_kinematics = []
    all_jet1_PFCands = []
    all_jet2_PFCands = []
    all_truth_labels = []
    if '3000' in signal:
        mjj_lower=2725.0#2450.#2725.0
        mjj_upper=3331.0#3550.#
    if '5000' in signal:
        mjj_lower=4500.0
        mjj_upper=5500.0
    output_dir=os.path.join(output_dir,signal)

    
    pathlib.Path(os.path.join(output_dir,'train')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_dir,'test')).mkdir(parents=True, exist_ok=True)

    # List all h5 files in the directory
    
    if from_reco:
        sig_dir=f'/storage/9/abal/CASE/VAE_results/events/run_141098/{signal}_RECO/nominal'
        data_dir='/storage/9/abal/CASE/VAE_results/events/run_141098/qcd_data_SR_Oz_RECO/'
        if sideband:
            print("No reco exists for sideband")
            print("Set sideband=False, continuing with signal region data")
            #data_dir='/storage/9/abal/CASE/new_signals/run2_data_side/merged'

    else:        
        sig_dir=f'/ceph/abal/CASE/Lundv2_preprocessed_signals/{signal}/nominal'
        data_dir='/storage/9/abal/CASE/new_signals/run2_data_SR_Oz'
        if sideband:
            data_dir='/storage/9/abal/CASE/new_signals/run2_data_side/merged'

    
    directories = [sig_dir,data_dir]
    filelist = []
    for directory in directories:
        filelist=filelist+(glob.glob(os.path.join(directory, '*.h5')))
    num_evts_QCD=0
    num_evts_sig=0
    # Loop over each file and read the datasets
    
    for file_path in tqdm.tqdm(filelist):
        with h5py.File(file_path, 'r') as file:
            if (num_evts_QCD>=read_n) and (num_evts_sig>=read_n):
                break
            if from_reco:   
                jet_kinematics = file['eventFeatures'][:]
                jet1_PFCands = file['jetOrigConstituentsList'][0,...]
                jet2_PFCands = file['jetOrigConstituentsList'][1,...]
                
            else:
                jet_kinematics = file['jet_kinematics'][:]
                jet1_PFCands = file['jet1_PFCands'][:]
                jet2_PFCands = file['jet2_PFCands'][:]
            
            try:
                truth_label = file['truth_label'][:]
            except:
                if ('data' in file_path) or ('qcd' in file_path):
                    truth_label=np.zeros((jet1_PFCands.shape[0],1))
                    print("Inferred: QCD. Truth label --> 0")
                else:
                    truth_label=np.ones((jet1_PFCands.shape[0],1))
                    print("Inferred: signal. Truth label --> 1")

            mask=(jet_kinematics[:,0]>mjj_lower)&(jet_kinematics[:,0]<mjj_upper)
            jet_kinematics=jet_kinematics[mask]
            jet1_PFCands=jet1_PFCands[mask]
            jet2_PFCands=jet2_PFCands[mask]
            truth_label=truth_label[mask]

            if 'data' in file_path:
                if num_evts_QCD>=read_n: continue
                evts_left=int(read_n-num_evts_QCD)
                num_evts_QCD+=truth_label.shape[0]
                num_evts_QCD=min(read_n,num_evts_QCD)
                print(f"{evts_left} events left to read of type: QCD")
            else:
                if num_evts_sig>=read_n: continue
                evts_left=int(read_n-num_evts_sig)
                num_evts_sig+=truth_label.shape[0]
                num_evts_sig=min(read_n,num_evts_sig)
                print(f"{evts_left} events left to read of type: signal")
            
                
            # if 'run2_data_SR_Oz' in file_path:
            #     mask=(jet_kinematics[:,0]>1800.)&(jet_kinematics[:,0]<2300.)
            # else:
            #     mask=(jet_kinematics[:,0]>mjj_lower)&(jet_kinematics[:,0]<mjj_upper)
            
            
            
            # Append data to lists
            all_jet_kinematics.append(jet_kinematics[:evts_left])
            all_jet1_PFCands.append(jet1_PFCands[:evts_left])
            all_jet2_PFCands.append(jet2_PFCands[:evts_left])
            all_truth_labels.append(truth_label[:evts_left])

    # Concatenate all data from lists into single arrays
    all_jet_kinematics = np.concatenate(all_jet_kinematics, axis=0)
    all_jet1_PFCands = np.concatenate(all_jet1_PFCands, axis=0)
    all_jet2_PFCands = np.concatenate(all_jet2_PFCands, axis=0)
    all_truth_labels = np.concatenate(all_truth_labels, axis=0)
    
    # Shuffle the data
    indices = np.arange(all_jet_kinematics.shape[0])
    np.random.shuffle(indices)
    
    all_jet_kinematics = all_jet_kinematics[indices]
    all_jet1_PFCands = all_jet1_PFCands[indices]
    all_jet2_PFCands = all_jet2_PFCands[indices]
    all_truth_labels = all_truth_labels[indices]
    all_jet1_PFCands = np.take_along_axis(all_jet1_PFCands,np.argsort(all_jet1_PFCands[..., 0]*(-1), axis=1)[...,None],axis=1)
    all_jet2_PFCands = np.take_along_axis(all_jet2_PFCands,np.argsort(all_jet2_PFCands[..., 0]*(-1), axis=1)[...,None],axis=1)
    # jet1_pxpypz=np.zeros_like(all_jet1_PFCands)
    # jet2_pxpypz=np.zeros_like(all_jet2_PFCands)

    # jet1_pxpypz[...,0]=all_jet1_PFCands[:,:,0]*np.cos(all_jet1_PFCands[:,:,2])
    # jet1_pxpypz[...,1]=all_jet1_PFCands[:,:,0]*np.sin(all_jet1_PFCands[:,:,2])
    # jet1_pxpypz[...,2]=all_jet1_PFCands[:,:,0]*np.sinh(all_jet1_PFCands[:,:,1])
    
    # jet2_pxpypz[...,0]=all_jet2_PFCands[:,:,0]*np.cos(all_jet2_PFCands[:,:,2])
    # jet2_pxpypz[...,1]=all_jet2_PFCands[:,:,0]*np.sin(all_jet2_PFCands[:,:,2])
    # jet2_pxpypz[...,2]=all_jet2_PFCands[:,:,0]*np.sinh(all_jet2_PFCands[:,:,1])
    #Save the data in batches to new h5 files
    num_batches = (len(indices) + batch_size - 1) // batch_size  # Calculate the number of batches
    constituents_j1j2 = np.vstack([all_jet1_PFCands,all_jet2_PFCands])
    mean,std=get_mean_and_stdev(constituents_j1j2)
    np.savez(os.path.join(output_dir, 'mean_stdev.npz'), mean=mean, std=std)
    print(f'mean = {mean} \n sigma = {std}')
    belongs_to='train'
    print(f"Will begin saving the data in {num_batches} batches at location: {output_dir}")
    for i in tqdm.tqdm(range(num_batches)):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_jet_kinematics = all_jet_kinematics[start_index:end_index]
        batch_jet1_PFCands = all_jet1_PFCands[start_index:end_index] #jet1_pxpypz[start_index:end_index]
        batch_jet2_PFCands = all_jet2_PFCands[start_index:end_index]#jet2_pxpypz[start_index:end_index]
        batch_truth_labels = all_truth_labels[start_index:end_index]
        if i > (int)(num_batches*0.8):
            belongs_to='test'
        # Create a new h5 file for each batch
        with h5py.File(os.path.join(output_dir,belongs_to, f'batch_{i+1}.h5'), 'w') as batch_file:
            batch_file.create_dataset('jet_kinematics', data=batch_jet_kinematics)
            batch_file.create_dataset('jet1_PFCands', data=batch_jet1_PFCands)
            batch_file.create_dataset('jet2_PFCands', data=batch_jet2_PFCands)
            batch_file.create_dataset('truth_label', data=batch_truth_labels[:,0])



def get_mean_and_stdev(dat): # nd.array [N,K,num-features]-> nd.array [num-features], nd.array [num-features]
	''' compute mean and std-dev of each feature (axis 2) of a datasample [N_examples, K_elements, F_features]
	'''
	std = np.nanstd(dat, axis=(0,1))
	mean = np.nanmean(dat, axis=(0,1))
	print('computed mean {} and std-dev {}'.format(mean, std))
	std[std == 0.0] = 0.001 # handle zeros
	return mean, std


if __name__ == '__main__':
    directory = '/ceph/abal/CASE/supervised_training_data/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal",default='XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow',help="Set signal against which to train classifier")
    parser.add_argument("--from_reco",action="store_true")
    args = parser.parse_args()
    process_and_save_h5_files(directory,batch_size=1000,signal=args.signal,from_reco=args.from_reco)
