from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import acc_f1_from_binary_confusion_mat
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import os


cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

def train_LR(LR_datapath):

    df = pd.read_csv(os.path.join(LR_datapath, 'train_LR.csv'))
    X = df.iloc[:,:2].values
    Y = df.iloc[:,2:].values

    df2 = pd.read_csv( os.path.join(LR_datapath, 'val_LR.csv'))
    X2 = df.iloc[:,:2].values
    Y2 = df.iloc[:,2:].values

    X_train = np.concatenate([X,X2])

    Y_train = np.concatenate([Y,Y2])
    #X_train, Y_train = X, Y

    df3 = pd.read_csv(os.path.join(LR_datapath, 'test_LR.csv'))
    X_test = df.iloc[:,:2].values
    Y_test = df.iloc[:,2:].values

    # Model training    
    model = LogisticRegression(penalty='l1',C=1.2,solver='saga',max_iter=200)
    model.fit( X_train, Y_train )  

    Y_pred = model.predict( X_test ) 

    # measure performance    
    correctly_classified = 0    
    tp, fp, tn, fn = 0, 0, 0, 0
    # counter     
    count_0, count_1 = 0, 0
    for count in range( np.size( Y_pred ) ) :  
            
        if Y_test[count] == 1 and Y_pred[count] == 1:            
            tp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 1:
            fp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 0:
            tn += 1
        elif Y_test[count] == 1 and Y_pred[count] == 0:
            fn += 1  
                
        if Y_test[count] == 0: count_0 += 1
        elif Y_test[count] == 1: count_1 += 1

    acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
        
    print("Acc: {} and f1: {}".format(acc,f1))
    print("Count_0: {}, Count_1: {}".format(count_0, count_1))

    return model

def train_svm():

    df = pd.read_csv( "../data/train_LR/train_LR.csv" )
    X = df.iloc[:,:2].values
    Y = df.iloc[:,2:].values

    df2 = pd.read_csv( "../data/train_LR/val_LR.csv" )
    X2 = df.iloc[:,:2].values
    Y2 = df.iloc[:,2:].values

    X_train = np.concatenate([X,X2])

    Y_train = np.concatenate([Y,Y2])

    df3 = pd.read_csv( "../data/train_LR/test_LR.csv" )
    X_test = df.iloc[:,:2].values
    Y_test = df.iloc[:,2:].values

    # Model training      
    model = SVC(gamma='auto', probability = True)
    model.fit( X_train, Y_train)  

    Y_pred = model.predict( X_test ) 

    # measure performance    
    correctly_classified = 0    
    tp, fp, tn, fn = 0, 0, 0, 0
    # counter     
    count_0, count_1 = 0, 0
    for count in range( np.size( Y_pred ) ) :  
            
        if Y_test[count] == 1 and Y_pred[count] == 1:            
            tp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 1:
            fp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 0:
            tn += 1
        elif Y_test[count] == 1 and Y_pred[count] == 0:
            fn += 1  
                
        if Y_test[count] == 0: count_0 += 1
        elif Y_test[count] == 1: count_1 += 1

    acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
        
    print("Acc: {} and f1: {}".format(acc,f1))
    print("Count_0: {}, Count_1: {}".format(count_0, count_1))

    return model

def train_mlp():

    df = pd.read_csv( "../data/train_LR/train_LR.csv" )
    X = df.iloc[:,:2].values
    Y = df.iloc[:,2:].values

    df2 = pd.read_csv( "../data/train_LR/val_LR.csv" )
    X2 = df.iloc[:,:2].values
    Y2 = df.iloc[:,2:].values

    X_train = np.concatenate([X,X2])

    Y_train = np.concatenate([Y,Y2])

    df3 = pd.read_csv( "../data/train_LR/test_LR.csv" )
    X_test = df.iloc[:,:2].values
    Y_test = df.iloc[:,2:].values

    # Model training    
    model = MLPClassifier(random_state=1, max_iter=300)
    model.fit( X_train, Y_train )  

    Y_pred = model.predict( X_test ) 

    # measure performance    
    correctly_classified = 0    
    tp, fp, tn, fn = 0, 0, 0, 0
    # counter     
    count_0, count_1 = 0, 0
    for count in range( np.size( Y_pred ) ) :  
            
        if Y_test[count] == 1 and Y_pred[count] == 1:            
            tp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 1:
            fp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 0:
            tn += 1
        elif Y_test[count] == 1 and Y_pred[count] == 0:
            fn += 1  
                
        if Y_test[count] == 0: count_0 += 1
        elif Y_test[count] == 1: count_1 += 1

    acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
        
    print("Acc: {} and f1: {}".format(acc,f1))
    print("Count_0: {}, Count_1: {}".format(count_0, count_1))

    breakpoint()

    plt.scatter(X_test[:,0], X[:,1], c=Y_test)
    plt.show()
    plt.savefig("scatterplot")

    return model


def train_rbf():

    df = pd.read_csv( "../data/train_LR/train_LR.csv" )
    X = df.iloc[:,:2].values
    Y = df.iloc[:,2:].values

    df2 = pd.read_csv( "../data/train_LR/val_LR.csv" )
    X2 = df.iloc[:,:2].values
    Y2 = df.iloc[:,2:].values

    X_train = np.concatenate([X,X2])

    Y_train = np.concatenate([Y,Y2])

    df3 = pd.read_csv( "../data/train_LR/test_LR.csv" )
    X_test = df.iloc[:,:2].values
    Y_test = df.iloc[:,2:].values

    # Model training 
    kernel = 1.0 * RBF(1.0)   
    model = GaussianProcessClassifier(kernel=kernel, random_state=0)

    print("Training.....")
    model.fit( X_train, Y_train )  

    Y_pred = model.predict( X_test ) 
    breakpoint()

    # measure performance    
    correctly_classified = 0    
    tp, fp, tn, fn = 0, 0, 0, 0
    # counter     
    count_0, count_1 = 0, 0
    for count in range( np.size( Y_pred ) ) :  
            
        if Y_test[count] == 1 and Y_pred[count] == 1:            
            tp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 1:
            fp += 1
        elif Y_test[count] == 0 and Y_pred[count] == 0:
            tn += 1
        elif Y_test[count] == 1 and Y_pred[count] == 0:
            fn += 1  
                
        if Y_test[count] == 0: count_0 += 1
        elif Y_test[count] == 1: count_1 += 1

    acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
        
    print("Acc: {} and f1: {}".format(acc,f1))
    print("Count_0: {}, Count_1: {}".format(count_0, count_1))

    breakpoint()
    return model




