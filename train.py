import os,pathlib,argparse
#import setGPU
import numpy as np
from collections import namedtuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
tf.config.run_functions_eagerly(True)
import case_supervised.models.classifier as classifier
import case_supervised.data.data_processors as dp
#import debug_utils.get_free_gpu_id as gpu; gpu.set_gpu()

os.environ["CUDA_VISIBLE_DEVICES"]="1,3"
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
    
    args = parser.parse_args()
    seed=args.seed

    BATCH_SIZE=args.batchsize

    data_dir=f'/ceph/abal/CASE/supervised_training_data/{args.signal}'
    if args.add_decoder:
        model_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder+decoder',args.signal,args.seed,'model')
        result_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder+decoder',args.signal,args.seed,'results')
    else:
        model_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder_only/',args.signal,args.seed,'model')
        result_dir=os.path.join('/work/abal/CASE/CASE_supervised_classifiers/encoder_only/',args.signal,args.seed,'results')
    pathlib.Path(result_dir).mkdir(parents=True,exist_ok=True)
    pathlib.Path(os.path.join(model_dir,'best_so_far')).mkdir(parents=True,exist_ok=True)

    mean,stdev=fetch_mean_and_stdev(data_dir)
    
    Parameters = namedtuple('Parameters', 'run_n input_shape dense_factor1 dense_factor2 kernel_sz kernel_1D_sz kernel_ini_n beta epochs batch_n train_split batch_shuffle_size z_sz activation initializer learning_rate max_lr_decay lambda_reg comments signal data_or_MC')
    params = Parameters(run_n=seed, 
                        input_shape=(100,3),
                        kernel_sz=(1,3), 
                        kernel_1D_sz=3,
                        kernel_ini_n=12,
                        beta=25,
                        epochs=50,
                        batch_n=BATCH_SIZE,
                        train_split=0.8,
                        batch_shuffle_size=1000,
                        z_sz=12,
                        dense_factor1=17,#17
                        dense_factor2=4,#4
                        activation='leaky_relu',
                        initializer='he_uniform',
                        learning_rate=5.0e-4,
                        max_lr_decay=4, 
                        lambda_reg=0.01,
                        comments='supervised classifier',
                        signal=args.signal,
                        data_or_MC='data + signal MC') 



    with open(os.path.join(model_dir,"params.json"),'w') as f:
        print(f"Dumping model parameters to JSON file")
        import json;json.dump(params._asdict(),f,indent=4) 



    if args.add_decoder:
        classifier_model=classifier.VAELargeClassifier(params,mean,stdev)#.build_model()
    else:
        classifier_model=classifier.VAEClassifier(params,mean,stdev)#.build_model()
    train_dataset, val_dataset = dp.create_training_datasets(data_dir, BATCH_SIZE, params.batch_shuffle_size, params.train_split,params.input_shape)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001,patience=3, restore_best_weights=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir,'best_so_far','best_so_far.h5'), monitor='val_auc', save_best_only=True,save_weights_only=True)
    #classifier_model.build_graph()
    #class_weight = {0: BATCH_SIZE/700, 1: BATCH_SIZE/300}
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params.learning_rate,decay_steps=500,decay_rate=0.1,staircase=True)
    classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),\
                            metrics=['accuracy',tf.keras.metrics.AUC()])
    classifier_model.summary()
    #classifier_model.summary()
    
    #import pdb;pdb.set_trace()
    classifier_model.fit(train_dataset, epochs=params.epochs, validation_data=val_dataset,\
                         callbacks=[early_stopping_callback,checkpoint_callback])
    #test_dataset = dp.create_test_dataset(data_dir, BATCH_SIZE, params.batch_shuffle_size,params.input_shape)
    classifier_model.save_model(model_dir)  
    classifier_model.save(model_dir)
    # my_model=classifier.VAEClassifier(params,mean,stdev)
    # my_model.load_model(model_dir)
    # bce_score=tf.keras.metrics.BinaryCrossentropy()
    # auc_score=tf.keras.metrics.AUC()

    #x_test,y_test=dp.read_dataset(os.path.join(data_dir,'test'))
    
  
