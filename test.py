import os,json
#import setGPU
import numpy as np
from collections import namedtuple
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
tf.config.run_functions_eagerly(True)
import case_supervised.models.classifier as classifier
import case_supervised.data.data_processors as dp
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from vande.vae.layers import StdNormalization

#import debug_utils.get_free_gpu_id as gpu; gpu.set_gpu()
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
def fetch_mean_and_stdev(filepath):
    mean=np.load(os.path.join(filepath,'mean_stdev.npz'))['mean']
    std=np.load(os.path.join(filepath,'mean_stdev.npz'))['std']
    return mean,std
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--batchsize",type=int,default=256,help="Set Batch Size")
    parser.add_argument("-s","--seed",type=str,default='12345',help="Set seed")
    parser.add_argument("-l","--load",action='store_true',help="Load saved model from specified run number if true")
    parser.add_argument("--signal",default='XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow',help="Set signal against which to train classifier")
    parser.add_argument("--add_decoder",action='store_true',help="Choose whether to use the full VAE architecture as a classifier or not")
    parser.add_argument("--plot_vae_scores",action='store_true')
    args = parser.parse_args()
    seed=args.seed

    BATCH_SIZE=args.batchsize
    label_dict={'YtoHH_Htott_Y3000_H400':'$M_{Y}= 3$ TeV, $M_{H}=400$ GeV',\
    'XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow':'$M_{X}=3$ TeV, $M_{Y} = M_{Y\'} = 170$ GeV',\
    'WkkToWRadionToWWW_M3000_Mr170':'$M_{W_{KK}}=3$ TeV, $M_{R}=170$ GeV'}

    data_dir=f'/ceph/abal/CASE/supervised_training_data/{args.signal}'
    if args.add_decoder:
        model_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder+decoder',args.signal,args.seed,'model')
        result_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder+decoder',args.signal,args.seed,'results')
    else:
        model_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder_only',args.signal,args.seed,'model')
        result_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder_only',args.signal,args.seed,'results')
        
    mean,stdev=fetch_mean_and_stdev(data_dir)
    print("Mean=",mean)
    print("sigma=",stdev)
    custom_objects={'StdNormalization':StdNormalization(mean_x=mean,std_x=stdev)}
    if args.add_decoder:
        custom_objects['Sampling']=classifier.Sampling()
    
    Parameters = namedtuple('Parameters', 'run_n input_shape dense_factor1 dense_factor2 kernel_sz kernel_1D_sz kernel_ini_n beta epochs batch_n train_split batch_shuffle_size z_sz activation initializer learning_rate max_lr_decay lambda_reg comments signal data_or_MC')
    params = Parameters(run_n=seed, 
                        input_shape=(100,3),
                        kernel_sz=(1,3), 
                        kernel_1D_sz=3,
                        kernel_ini_n=12,
                        beta=25,
                        epochs=4,
                        batch_n=BATCH_SIZE,
                        train_split=0.8,
                        batch_shuffle_size=1000,
                        z_sz=12,
                        dense_factor1=17,#17
                        dense_factor2=4,#4
                        activation='leaky_relu',
                        initializer='he_uniform',
                        learning_rate=1.0e-4,
                        max_lr_decay=8, 
                        lambda_reg=0.01,
                        comments='supervised classifier',
                        signal=args.signal,
                        data_or_MC='data + signal MC') 

    #if args.add_decoder:
    #    classifier_model=classifier.VAELargeClassifier(params,mean,stdev)
    #else:
    #    classifier_model=classifier.VAEClassifier(params,mean,stdev)
    classifier_model=tf.keras.models.load_model(model_dir,custom_objects=custom_objects)
    
    x_test,y_test,vae_score_test = dp.read_dataset(os.path.join(data_dir,'test'),input_shape=params.input_shape,read_vae_loss=True)
    #y_preds=classifier_model.predict(x_test)
    y_pred=classifier_model.predict(x_test)
    
    y_pred=y_pred[:,0]

    bce_score=tf.keras.metrics.BinaryCrossentropy()
    keras_auc_scorer=tf.keras.metrics.AUC()
    fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred)
    auc=metrics.auc(fpr,tpr)
    auc=metrics.auc(fpr,tpr)

    vae_fpr,vae_tpr,vae_thresholds=metrics.roc_curve(y_test,vae_score_test)
    vae_auc=metrics.auc(vae_fpr,vae_tpr)
    vae_auc=metrics.auc(vae_fpr,vae_tpr)

    
    
    hep.style.use("CMS")
    hep.cms.label('Preliminary',data=True, lumi=137.2, year='2016-18')
    plt.plot(1-fpr, tpr, 'b', label = 'AUC (supervised) = %0.3f' % auc)
    plt.plot(1-vae_fpr, vae_tpr, 'g', label = 'AUC (VAE loss) = %0.3f' % vae_auc)
    
    plt.legend(loc = 'lower left')
    plt.plot([0, 1], [1, 0],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Background rejection")
    plt.ylabel("Signal Efficiency")
    try:
        sig_str=label_dict[args.signal]
    except:
        sig_str='placeholder signal name'
        print("Add signal string to label_dict. Using placeholder for now.")
    
    plt.text(0.05, 0.3, sig_str,fontsize=17)
    plt.savefig(os.path.join(result_dir,'ROC_curve.png'))
    print('AUC: ',auc)
    
    bce=bce_score(y_test,y_pred).numpy()
    
    bkg_preds=y_pred[y_test==0]
    bkg_truth=y_test[y_test==0]
    sig_preds=y_pred[y_test==1]
    sig_truth=y_test[y_test==1]
    metrics_dict={'AUC':float(auc),'BCE':float(bce)}
    bkg_efficiencies=[1,10,25]
    for eff in bkg_efficiencies:
        threshold=np.percentile(bkg_preds,100.-eff)
        bkg_eff=np.sum(bkg_preds>threshold)/bkg_preds.shape[0]
        sig_eff=np.sum(sig_preds>threshold)/sig_preds.shape[0]
        print('Bkg eff: ',bkg_eff)
        print('Sig eff: ',sig_eff)
        print('Threshold: ',threshold)
        sic=sig_eff/np.sqrt(bkg_eff)
        metrics_dict['SIC_'+str(eff)]=float(sic)
        metrics_dict['threshold_'+str(eff)]=float(threshold)    
    #threshold=np.percentile(bkg_preds,99)
    #bkg_eff=np.sum(bkg_preds>threshold)/bkg_preds.shape[0]
    #sig_eff=np.sum(sig_preds>threshold)/sig_preds.shape[0]
    #sic=sig_eff/np.sqrt(bkg_eff)
    print('SIC: ',sic)
    print('Threshold: ',threshold)
    print('BCE:',bce)
    write=True
    if write:    
        with open(os.path.join(result_dir,'metrics.json'),'w') as f:
            json.dump(metrics_dict, f)
    print('Metrics written to file')