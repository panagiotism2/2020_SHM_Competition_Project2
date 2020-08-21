import os
import numpy as np
import random as rn
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
import keras.callbacks
from IPython.display import clear_output
from keras import optimizers
import tensorflow as tf
from numpy.testing import assert_allclose
from keras.callbacks import ModelCheckpoint



# def loss(y_true,y_pred):
#     diff=y_pred - y_true
#     mask = K.less(y_pred, y_true) #i.e. y_pred - y_true < 0
#     # with tf.Session() as sess:  aaa=mask.eval()
   
#     #if sess.run(mask):
#     # if  aaa:
#     RR=K.sum(K.exp(1/10*K.abs(diff)))  
#     # else:
#     #     RR=K.sum(K.exp(K.abs(diff)))  
#     # return RR



# # def loss(y_true,y_pred):
# #     diff=y_pred - y_true

# #     a=1/13
# #     RR=K.sum(K.exp(a*K.abs(diff)))  
#     return RR




def create_supervised(input_shape, activ, cell, dropout,n_classes):
    """
    """
    # Set-up
    N_hl = len(cell)-1
    
    # Define the input placeholder as a tensor with shape input_shape    
    X_input = Input(input_shape)    
  
    X = X_input
    # Hidden layers
    for i in range(N_hl):
        r_s = True
        if i==N_hl-1:
            r_s = False
        X = LSTM(cell[i], return_sequences=r_s, activation=activ, kernel_initializer=glorot_uniform(seed=0))(X) 

    # Fully connected
    X = Dense(cell[-1], activation=activ, kernel_initializer=glorot_uniform(seed=0), name='fc')(X)
    if dropout != 0:
        X = Dropout(dropout)(X)
    X = Dense(n_classes, activation='softmax', name='Output')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='LSTM')    
   
    return model

def supXLSTM(ModelName,X_train, Y_train, X_valid, Y_valid, epochs_s, batch_size_s, cell, activ, dropout,n_classes,class_weight,monitor_metric):
    # updatable plot
    class PlotLosses(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []        
            self.fig = plt.figure()        
            self.logs = []
    
        def on_epoch_end(self, epoch, logs={}):        
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1
            
            clear_output(wait=True)
            plt.plot(self.x, np.sqrt(self.losses), label="loss")
            plt.plot(self.x, np.sqrt(self.val_losses), label="val_loss")
            plt.ylabel('loss - RMSE')
            plt.xlabel('epoch')
            plt.legend(['train','validation'], loc='upper left')
            plt.title('model loss = ' + str(min(np.sqrt(self.val_losses))))
            plt.show();
            
    #plot_losses = PlotLosses()
    # Set-up
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(229)
    rn.seed(229)
    tf.random.set_seed(229)
    K.clear_session()

    # Create model    
    input_shape= X_train.shape[1:]
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    opt = 'Adam'
    supervised = create_supervised(input_shape, activ, cell, dropout,n_classes)   

    #supervised.compile(optimizer=opt, loss="mean_squared_error")
    supervised.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy)

    # define the checkpoint
    filepath = ModelName+'_best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor=monitor_metric, verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #supervised.compile(optimizer=opt, loss=lf_NASA)
    # Report model summary
    supervised.summary()

    # Fit model   
#    history = supervised.fit(X_train, Y_train, validation_split=pr_valid,epochs=epochs_s, batch_size=batch_size_s,  class_weight=class_weight, callbacks=callbacks_list)
    history = supervised.fit(X_train, Y_train, validation_data = (X_valid, Y_valid),epochs=epochs_s, batch_size=batch_size_s,  class_weight=class_weight, callbacks=callbacks_list)

    # # Collect outputs
    # predsTr = supervised.evaluate(x = X_train, y = Y_train)
    # val_los = history.history['val_loss'][-1]
    # tr_los = history.history['loss'][-1]
    # val_acc=history.history['sparse_categorical_accuracy'][-1]
    # tr_acc=history.history['val_sparse_categorical_accuracy'][-1]
    
    # predsTs = supervised.evaluate(x = X_test, y = Y_test)
    # y_hat = supervised.predict(x = X_test)
    
    # Summarize history for loss
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # # Save model
    # model_json = supervised.to_json()

    # #with open("PM_model_LSTM.json", "w") as json_file:
    # with open(ModelName+".json", "w") as json_file:
    #      json_file.write(model_json)
    
    # # Serialize weights to HDF5
    # supervised.save_weights(ModelName+".h5")
    # print("Saved model to disk")
    #K.clear_session()
    
    return supervised


def plot_confusion_matrix(cm, class_names):
    import itertools
    cm_abs=cm.astype(np.float)
    cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Feature_'+str(rr))
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
     

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, format(cm[i, j], '.2f' )+" ("+format(cm_abs[i,j], '.0f' )+')', horizontalalignment="center", color=color)
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def ModelAssess(model,x_test,y_test,class_names):
    ###### INPUT
    model=model 
    x_test=x_test # Test Input data: n_data x n_features x 1 (Array of float64)
    y_test=y_test # Test Output data: n_data x n_classes (Array of float64)
    class_names=class_names
    ######
    
    
    # Confusion matrix
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score 
    
    
    # Confusion Matrix
    #plt.figure(11,figsize = (18,12))
    plt.figure(11)
    model_preds = model.predict(x = x_test)
    test_preds = np.argmax(model_preds , axis=-1) # Get prediction
    test_probabilities=model_preds[test_preds] # get the probability of the prediction
    
    # Reverse one hot encoding for labels
    test_labels = np.argmax(y_test, axis=1)        
    cm = confusion_matrix(test_labels, test_preds ) 
    plot_confusion_matrix(cm, class_names=class_names)
    
    a_score=accuracy_score(test_labels,test_preds)
    
    print(a_score)
    
    # Get los metrics
    val_los =model.history.history['val_loss'][-1]
    tr_los = model.history.history['loss'][-1]
    n_epochs = model.history.params['epochs']
    # val_acc=model.history.history['sparse_categorical_accuracy'][-1]
    # tr_acc=model.history.history['val_sparse_categorical_accuracy'][-1]
    
    

    return(cm,a_score,val_los,tr_los,n_epochs)


def supXLSTM_trainfurther(filepath,X_train, Y_train, X_valid, Y_valid, epochs_s, batch_size_s,class_weight,n_epochs_old,monitor_metric):
    from keras.models import  load_model
    from keras.callbacks import ModelCheckpoint
    
    # updatable plot
    class PlotLosses(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []        
            self.fig = plt.figure()        
            self.logs = []
    
        def on_epoch_end(self, epoch, logs={}):        
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1
            
            clear_output(wait=True)
            plt.plot(self.x, np.sqrt(self.losses), label="loss")
            plt.plot(self.x, np.sqrt(self.val_losses), label="val_loss")
            plt.ylabel('loss - RMSE')
            plt.xlabel('epoch')
            plt.legend(['train','validation'], loc='upper left')
            plt.title('model loss = ' + str(min(np.sqrt(self.val_losses))))
            plt.show();
    
            
    ###### INPUT
    filepath = filepath
    x_train=X_train
    y_train=Y_train
    x_valid=X_valid
    y_valid=Y_valid
    
    epochs=epochs_s
    batch_size=batch_size_s
    class_weight=class_weight
    n_epochs_old=n_epochs_old # number of epochs of previous training just for the name
    ######
    
    #plot_losses = PlotLosses()
    # Set-up
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(229)
    rn.seed(229)
    tf.random.set_seed(229)
    K.clear_session()

    # Load Model  
    new_model = load_model(filepath)
    n_epochs_new=n_epochs_old+epochs
    
    # Define new model name
    CustomName=filepath.split('_')[0]
    filepath_new=CustomName+'_LSTM_e'+str(n_epochs_new)+'_best.hdf5'


    # Train the new model
    checkpoint = ModelCheckpoint(filepath_new, monitor=monitor_metric, verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history=new_model.fit(x_train, y_train,validation_data = (x_valid, y_valid) , epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks=callbacks_list)
    
    # Summarize history for loss
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # # Save model
    # model_json = supervised.to_json()
    
    # #with open("PM_model_LSTM.json", "w") as json_file:
    # with open(ModelName+".json", "w") as json_file:
    #       json_file.write(model_json)
        
    
    return new_model





