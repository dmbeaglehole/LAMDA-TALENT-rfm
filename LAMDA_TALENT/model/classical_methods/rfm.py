from model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
import time
from sklearn.metrics import accuracy_score, mean_squared_error

import sys
sys.path.insert(0, '/u/dbeaglehole/recursive_feature_machines/')
from rfm import LaplaceRFM, GeneralizedLaplaceRFM, NTKModel, matrix_sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import os


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

class SmoothClampedReLU(nn.Module):
    def __init__(self, beta=50):
        super(SmoothClampedReLU, self).__init__()
        self.beta = beta
        
    def forward(self, x):
        # Smooth transition at x=0 (using softplus with high beta)
        activated = F.softplus(x, beta=self.beta)
        
        # Smooth transition at x=1 (using sigmoid scaled and shifted)
        # As x approaches infinity, this approaches 1
        clamped = activated - F.softplus(activated - 1, beta=self.beta)
        
        return clamped



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

        self.use_feature_learning = model_config['use_feature_learning']
        self.use_ntk = model_config['use_ntk']
        self.kernel_type = kernel_type
        self.exponent = exponent
        self.diag = model_config['diag']
        self.agop_power = model_config['agop_power']

        if kernel_type == 'laplace':
            self.model = LaplaceRFM(device='cuda', reg=reg, bandwidth=bandwidth, iters=iters, diag=self.diag)   
        elif kernel_type == 'gen_laplace':
            exponent = model_config['exponent']
            self.model = GeneralizedLaplaceRFM(device='cuda', reg=reg, bandwidth=bandwidth, iters=iters, exponent=exponent, diag=self.diag)


    def fit(self, data, info, train=True, config=None, train_on_subset=False, agop_path='./agops/'):
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config:
            self.reset_stats_withconfig(config)
        
        if 'model' in self.args.config:
            if 'cat_policy' in self.args.config['model']:
                self.args.cat_policy = self.args.config['model']['cat_policy']
            if 'normalization' in self.args.config['model']:
                self.args.normalization = self.args.config['model']['normalization']

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


        is_classification = self.is_binclass or self.is_multiclass
        if is_classification:
            print("Getting num classes")
            if 'n_classes' in info:
                num_classes = info['n_classes']
            elif 'num_classes' in info:
                num_classes = info['num_classes']
            else:
                num_classes = len(torch.unique(y_train))
                
        if len(y_train.shape)==1:
            if self.is_regression or self.is_binclass:
                y_train = y_train.unsqueeze(-1)
                y_val = y_val.unsqueeze(-1)
            elif self.is_multiclass:
                y_train = one_hot_encode(y_train, num_classes)
                y_val = one_hot_encode(y_val, num_classes)

        if not self.use_feature_learning:
            self.model.iters = 0

        if len(X_train) <= 50000:
            # use lstsq for small datasets
            fit_method = 'lstsq'
            iters_to_use = self.model.iters
        elif len(X_train) <= 75000:
            fit_method = 'eigenpro'
            iters_to_use = self.model.iters
        elif train_on_subset:
            # large dataset, but tune on subset of the training data, using lstsq
            fit_method = 'lstsq'
            train_subset_ids = np.random.choice(range(X_train.shape[0]), size=50000, replace=False)
            X_train = X_train[train_subset_ids]
            y_train = y_train[train_subset_ids]
            iters_to_use = self.model.iters
        else:
            # large dataset, train on the entire training data, using eigenpro and zero iterations of RFM
            # pre-trained M matrix is used
            fit_method = 'eigenpro'
            iters_to_use = 0
            if self.use_feature_learning:
                M_path = os.path.join(agop_path, f'agop_{self.args.dataset}_trial_{self.args.best_trial}.pt')
                self.model.M = torch.load(M_path).cuda()
                print("Loaded M matrix from", M_path)

        tic = time.time()
        self.model.fit((X_train, y_train), 
                       (X_val, y_val), 
                       classification=is_classification, 
                       method=fit_method, 
                       M_batch_size=8192 if self.kernel_type=='laplace' else 256,
                       return_best_params=True,
                       bs=4096,
                       top_q=196,
                       epochs=60,
                       lr_scale=1.0,
                       n_subsamples=16384,
                       verbose=True,
                       iters=iters_to_use)
        
        self.trlog['best_iter'] = self.model.best_iter

        if self.use_feature_learning and train_on_subset:
            print("Saving M matrix")
            M_path = os.path.join(agop_path, f'agop_{self.args.dataset}_trial_{info["trial"]}.pt')
            torch.save(self.model.M.cpu(), M_path)
        
        if self.use_ntk:
            if self.model.sqrtM is None:
                sqrtM = matrix_sqrt(self.model.M)
            else:
                sqrtM = self.model.sqrtM
            ntk_model = NTKModel(sqrtM=sqrtM, reg=self.model.reg, device='cuda')
            ntk_model.fit(X_train, y_train)
            y_val_pred = ntk_model.predict(X_val).cpu()
        else:
            y_val_pred = self.model.predict(X_val).cpu()

        if self.is_binclass:
            print("Binary classification")
            y_val_pred = torch.where(y_val_pred > 0.5, 1, 0).reshape(y_val.shape)
            self.trlog['best_res'] = accuracy_score(y_val.cpu(), y_val_pred)
        elif self.is_multiclass:
            print("Multi-class classification")
            y_val = torch.argmax(y_val, dim=1)
            y_val_pred = torch.argmax(y_val_pred, dim=1).reshape(y_val.shape)
            self.trlog['best_res'] = accuracy_score(y_val.cpu(), y_val_pred)
        else:
            print("Regression")
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
            'normalization': self.args.normalization,
            'use_ntk': self.use_ntk,
            'diag': self.diag
        }, ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))

        torch.cuda.empty_cache()
        gc.collect()
        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)

        # Load saved attributes
        checkpoint = torch.load(ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))
        if checkpoint['use_ntk']:
            self.model = NTKModel()
        elif checkpoint['kernel_type'] == 'laplace':
            self.model = LaplaceRFM(device='cuda')
        elif checkpoint['kernel_type'] == 'gen_laplace':
            self.model = GeneralizedLaplaceRFM(device='cuda', exponent=checkpoint['exponent'])
        self.model.weights = checkpoint['weights']
        self.model.M = checkpoint['M']
        if checkpoint['sqrtM'] is not None:
            self.model.sqrtM = checkpoint['sqrtM']
        else:
            self.model.sqrtM = matrix_sqrt(self.model.M)

        self.model.bandwidth = checkpoint['bandwidth']
        self.args.cat_policy = checkpoint['cat_policy']
        self.args.normalization = checkpoint['normalization']
        self.model.diag = checkpoint['diag']

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
            # Create the activation function
            smooth_clamped = SmoothClampedReLU(beta=10)
            predictions = smooth_clamped(predictions)

            if self.is_binclass:
                predicted_classes = torch.where(predictions > 0.5, 1, 0).reshape(labels.shape)
            elif self.is_multiclass:
                predicted_classes = torch.argmax(predictions, dim=1).reshape(labels.shape)
                predictions /= predictions.sum(dim=1, keepdim=True)

            accuracy = skm.accuracy_score(labels, predicted_classes)
            avg_precision = skm.precision_score(labels, predicted_classes, average='binary' if self.is_binclass else 'macro')
            avg_recall = skm.recall_score(labels, predicted_classes, average='binary' if self.is_binclass else 'macro')
            f1_score = skm.f1_score(labels, predicted_classes, average='binary' if self.is_binclass else 'macro')
            auc_score = skm.roc_auc_score(labels, predictions, multi_class='ovr')
            log_loss = skm.log_loss(labels, predictions)
            return (accuracy, avg_precision, avg_recall, f1_score, log_loss, auc_score), ("Accuracy", "Avg_Precision", "Avg_Recall", "F1", "LogLoss", "AUC")