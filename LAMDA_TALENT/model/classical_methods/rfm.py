from model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
import time
from sklearn.metrics import accuracy_score, mean_squared_error

import sys
sys.path.append('/u/dbeaglehole/recursive_feature_machines')
from rfm import LaplaceRFM, GeneralizedLaplaceRFM
import torch
import numpy as np
import gc


from model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_norm_process,
    num_enc_process,
    data_label_process,
    get_categories
)

def one_hot_encode(labels, num_classes):
    """Convert class labels to one-hot encoded format.
    
    Args:
        labels: tensor of class labels
    Returns:
        one_hot: tensor of one-hot encoded labels
    """
    one_hot = torch.zeros((labels.size(0), num_classes), device=labels.device)
    one_hot[torch.arange(labels.size(0)), labels.long()] = 1
    return one_hot

class RFMMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy not in ['indices'])

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']

        self.args.cat_policy = model_config['cat_policy']
        self.args.normalization = model_config['normalization']
        iters = model_config['iters']
        bandwidth = model_config['bandwidth']
        reg = model_config['reg']
        kernel_type = model_config['kernel_type']
        exponent = model_config['exponent']

        self.kernel_type = kernel_type
        self.exponent = exponent

        if kernel_type == 'laplace':
            self.model = LaplaceRFM(device='cuda', reg=reg, bandwidth=bandwidth, iters=iters)   
        elif kernel_type == 'gen_laplace':
            exponent = model_config['exponent']
            p_batch_size = 512
            self.model = GeneralizedLaplaceRFM(device='cuda', reg=reg, bandwidth=bandwidth, iters=iters, exponent=exponent, p_batch_size=p_batch_size)


    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config:
            self.reset_stats_withconfig(config)
        self.args.cat_policy = self.args.config['model']['cat_policy']
        self.args.normalization = self.args.config['model']['normalization']
        print("Training with cat_policy:", self.args.cat_policy)
        print("Training with normalization:", self.args.normalization)
        self.data_format(is_train = True)
        self.construct_model()
        
        if self.C is None:
            assert self.N is not None
            X_train = torch.from_numpy(self.N['train'])
            X_val = torch.from_numpy(self.N['val'])
        elif self.N is None:
            assert self.C is not None
            X_train = torch.from_numpy(self.C['train'])
            X_val = torch.from_numpy(self.C['val'])
        else:
            assert self.C is not None and self.N is not None
            X_train = torch.from_numpy(np.concatenate((self.C['train'], self.N['train']), axis=1))
            X_val = torch.from_numpy(np.concatenate((self.C['val'], self.N['val']), axis=1))
        
        X_train = X_train.to(dtype=torch.float32, device='cuda')
        X_val = X_val.to(dtype=torch.float32, device='cuda')
        y_train = torch.from_numpy(self.y['train']).to(dtype=torch.float32, device='cuda')
        y_val = torch.from_numpy(self.y['val']).to(dtype=torch.float32, device='cuda') 

        is_classification = not self.is_regression

        num_classes = len(torch.unique(y_train))
        
        if len(y_train.shape)==1:
            if self.is_regression or num_classes == 2:
                y_train = y_train.unsqueeze(-1)
                y_val = y_val.unsqueeze(-1)
            elif is_classification and num_classes > 2:
                y_train = one_hot_encode(y_train, num_classes)
                y_val = one_hot_encode(y_val, num_classes)

        tic = time.time()
        self.model.fit((X_train, y_train), 
                       (X_val, y_val), 
                       loader=False, 
                       classif=is_classification, 
                       method='lstsq', 
                       M_batch_size=2048)
        
        y_val_pred = self.model.predict(X_val).cpu()

        if is_classification:
            if len(y_val.shape)<= 2 and len(torch.unique(y_val)) == 2:
                y_val_pred = torch.where(y_val_pred > 0.5, 1, 0).reshape(y_val.shape)
            else:
                y_val = y_val.argmax(dim=1)
                y_val_pred = torch.argmax(y_val_pred, dim=1).reshape(y_val.shape)
            self.trlog['best_res'] = accuracy_score(y_val.cpu(), y_val_pred)
        else:
            # Calculate RMSE manually without the squared parameter
            mse = mean_squared_error(y_val.cpu(), y_val_pred.reshape(y_val.shape))
            self.trlog['best_res'] = (mse ** 0.5) * self.y_info['std']  # Convert MSE to RMSE and scale
        time_cost = time.time() - tic
        # Save just the essential model attributes
        if hasattr(self.model, 'sqrtM'):
            sqrtM = self.model.sqrtM
        else:
            sqrtM = None
        torch.save({
            'weights': self.model.weights,
            'M': self.model.M,
            'sqrtM': sqrtM,
            'bandwidth': self.model.bandwidth,
            'is_classification': is_classification,
            'kernel_type': self.kernel_type,
            'exponent': self.exponent,
            'cat_policy': self.args.cat_policy,
            'normalization': self.args.normalization
        }, ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))

        torch.cuda.empty_cache()
        gc.collect()
        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)

        # Load saved attributes
        checkpoint = torch.load(ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))
        if checkpoint['kernel_type'] == 'laplace':
            self.model = LaplaceRFM(device='cuda')
        elif checkpoint['kernel_type'] == 'gen_laplace':
            self.model = GeneralizedLaplaceRFM(device='cuda', exponent=checkpoint['exponent'])
        self.model.weights = checkpoint['weights']
        self.model.M = checkpoint['M']
        if checkpoint['sqrtM'] is not None:
            self.model.sqrtM = checkpoint['sqrtM']
        self.model.bandwidth = checkpoint['bandwidth']
        self.args.cat_policy = checkpoint['cat_policy']
        self.args.normalization = checkpoint['normalization']
        print("Predicting with cat_policy:", self.args.cat_policy)
        print("Predicting with normalization:", self.args.normalization)

        if self.C is None:
            assert self.N is not None
            X_train = torch.from_numpy(self.N['train'])
        elif self.N is None:
            assert self.C is not None
            X_train = torch.from_numpy(self.C['train'])
        else:
            assert self.C is not None and self.N is not None
            X_train = torch.from_numpy(np.concatenate((self.C['train'], self.N['train']), axis=1))
        
        X_train = X_train.to(dtype=torch.float32, device='cuda')
        self.model.centers = X_train
        is_classification = checkpoint['is_classification']
        
        # Convert test data and labels to tensors
        if self.C_test is None:
            assert self.N_test is not None
            X_test = torch.from_numpy(self.N_test)
        elif self.N_test is None:
            assert self.C_test is not None
            X_test = torch.from_numpy(self.C_test)
        else:
            assert self.C_test is not None and self.N_test is not None
            X_test = torch.from_numpy(np.concatenate((self.C_test, self.N_test), axis=1))

        X_test = X_test.to(dtype=torch.float32, device='cuda')
        test_label = torch.from_numpy(self.y_test).to(dtype=torch.float32)
        test_logit = self.model.predict(X_test).cpu()

        if is_classification:
            if len(test_label.shape)<= 2 and len(torch.unique(test_label)) == 2:
                test_logit = torch.where(test_logit > 0.5, 1, 0).reshape(test_label.shape)
            else:
                test_logit = torch.argmax(test_logit, dim=1).reshape(test_label.shape)

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit
    
    def metric(self, predictions, labels, y_info):
        from sklearn import metrics as skm
        if self.is_regression:
            mae = skm.mean_absolute_error(labels, predictions)
            rmse = skm.mean_squared_error(labels, predictions) ** 0.5
            r2 = skm.r2_score(labels, predictions)
            if y_info['policy'] == 'mean_std':
                mae *= y_info['std']
                rmse *= y_info['std']
            return (mae,r2,rmse), ("MAE", "R2", "RMSE")
        else:
            accuracy = skm.accuracy_score(labels, predictions)
            avg_precision = skm.precision_score(labels, predictions, average='macro')
            avg_recall = skm.recall_score(labels, predictions, average='macro')
            f1_score = skm.f1_score(labels, predictions, average='binary' if self.is_binclass else 'macro')
            return (accuracy, avg_precision, avg_recall, f1_score), ("Accuracy", "Avg_Precision", "Avg_Recall", "F1")