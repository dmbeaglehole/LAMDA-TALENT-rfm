import os
import shutil
import time
import errno
import pprint
import torch
import numpy as np
import random
import json
import os.path as osp
from hashlib import md5
import traceback


THIS_PATH = os.path.dirname(__file__)

def mkdir(path):
    """
    Create a directory if it does not exist.

    :path: str, path to the directory
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_gpu(x):
    """
    Set environment variable CUDA_VISIBLE_DEVICES
    
    :x: str, GPU id
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path, remove=True):
    """
    Ensure a path exists.

    path: str, path to the directory
    remove: bool, whether to remove the directory if it exists
    """
    if os.path.exists(path):
        if remove:
            if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
                shutil.rmtree(path)
                os.mkdir(path)
    else:
        os.mkdir(path)


#  --- criteria helper ---
class Averager():
    """
    A simple averager.

    """
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        """
        
        :x: float, value to be added
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        """
        Measure the time since the last call to measure.

        :p: int, period of printing the time
        """

        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

#  ---- import from lib.util -----------
def set_seeds(base_seed: int, one_cuda_seed: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    :base_seed: int, base seed
    :one_cuda_seed: bool, whether to set one seed for all GPUs
    """
    assert 0 <= base_seed < 2 ** 32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)

def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import sklearn.metrics as skm
def rmse(y, prediction, y_info):
    """
    
    :y: np.ndarray, ground truth
    :prediction: np.ndarray, prediction
    :y_info: dict, information about the target variable
    :return: float, root mean squared error
    """
    rmse = skm.mean_squared_error(y, prediction) ** 0.5  # type: ignore[code]
    if y_info['policy'] == 'mean_std':
        rmse *= y_info['std']
    return rmse
    
def load_config(args, config=None, config_name=None):
    """
    Load the config file.

    :args: argparse.Namespace, arguments
    :config: dict, config file
    :config_name: str, name of the config file
    :return: argparse.Namespace, arguments
    """
    if config is None:
        config_path = os.path.join(os.path.abspath(os.path.join(THIS_PATH, '..')), 
                                   'configs', args.dataset, 
                                   '{}.json'.format(args.model_type if args.config_name is None else args.config_name))
        with open(config_path, 'r') as fp:
            config = json.load(fp)

    # set additional parameters
    args.config = config 

    # save the config files
    with open(os.path.join(args.save_path, 
                           '{}.json'.format('config' if config_name is None else config_name)), 'w') as fp:
        args_dict = vars(args)
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return args

# parameter search
def sample_parameters(trial, space, base_config):
    """
    Sample hyper-parameters.

    :trial: optuna.trial.Trial, trial
    :space: dict, search space
    :base_config: dict, base configuration
    :return: dict, sampled hyper-parameters
    """
    def get_distribution(distribution_name):
        if distribution_name == 'float':
            return lambda label, *args: trial.suggest_float(
                label, 
                *args, 
                log=(args[1] / args[0] >= 10)  # Use log scale if range spans order of magnitude
            )
        return getattr(trial, f'suggest_{distribution_name}')

    result = {}
    for label, subspace in space.items():
        if isinstance(subspace, dict):
            result[label] = sample_parameters(trial, subspace, base_config)
        else:
            assert isinstance(subspace, list)
            
            # If list has exactly 2 elements, use second element as fixed value
            if len(subspace) == 2 and subspace[0] == "fixed":
                result[label] = subspace[1]
                continue
                
            distribution, *args = subspace

            if distribution == "categorical":
                # If args is a list of options, sample from those options
                if len(args) == 1 and isinstance(args[0], list):
                    result[label] = trial.suggest_categorical(label, args[0])
                else:
                    result[label] = trial.suggest_categorical(label, args)
                continue

            if distribution.startswith('?'):
                default_value = args[0]
                result[label] = (
                    get_distribution(distribution.lstrip('?'))(label, *args[1:])
                    if trial.suggest_categorical(f'optional_{label}', [False, True])
                    else default_value
                )

            elif distribution == '$mlp_d_layers':
                min_n_layers, max_n_layers, d_min, d_max = args
                n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
                suggest_dim = lambda name: trial.suggest_int(name, d_min, d_max)  # noqa
                d_first = [suggest_dim('d_first')] if n_layers else []
                d_middle = (
                    [suggest_dim('d_middle')] * (n_layers - 2) if n_layers > 2 else []
                )
                d_last = [suggest_dim('d_last')] if n_layers > 1 else []
                result[label] = d_first + d_middle + d_last

            elif distribution == '$d_token':
                assert len(args) == 2
                try:
                    n_heads = base_config['model']['n_heads']
                except KeyError:
                    n_heads = base_config['model']['n_latent_heads']

                for x in args:
                    assert x % n_heads == 0
                result[label] = trial.suggest_int('d_token', *args, n_heads)  # type: ignore[code]

            elif distribution in ['$d_ffn_factor', '$d_hidden_factor']:
                if base_config['model']['activation'].endswith('glu'):
                    args = (args[0] * 2 / 3, args[1] * 2 / 3)
                result[label] = trial.suggest_uniform('d_ffn_factor', *args)

            else:
                result[label] = get_distribution(distribution)(label, *args)
    return result

def merge_sampled_parameters(config, sampled_parameters):
    """
    Merge the sampled hyper-parameters.

    :config: dict, configuration
    :sampled_parameters: dict, sampled hyper-parameters
    """
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            # If there are parameters in the default config, the value of the parameter will be overwritten.
            config[k] = v

def get_classical_args():
    """
    Get the arguments for classical models.

    :return: argparse.Namespace, arguments
    """

    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    with open('configs/classical_configs.json','r') as file:
        default_args = json.load(file)
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--dataset', type=str, default=default_args['dataset'])
    parser.add_argument('--model_type', type=str, 
                        default=default_args['model_type'], 
                        choices=['LogReg', 'NCM', 'RandomForest', 
                                 'xgboost', 'catboost', 'lightgbm',
                                 'svm','knn', 'NaiveBayes',"dummy","LinearRegression",
                                 "rfm","kernel"
                                 ])
    
    # optimization parameters 
    parser.add_argument('--normalization', type=str, default=default_args['normalization'], choices=['none', 'standard', 'minmax', 'quantile', 'maxabs', 'power', 'robust'])
    parser.add_argument('--num_nan_policy', type=str, default=default_args['num_nan_policy'], choices=['mean', 'median'])
    parser.add_argument('--cat_nan_policy', type=str, default=default_args['cat_nan_policy'], choices=['new', 'most_frequent'])
    parser.add_argument('--cat_policy', type=str, default=default_args['cat_policy'], choices=['indices', 'ordinal', 'ohe', 'binary', 'hash', 'loo', 'target', 'catboost'])
    parser.add_argument('--num_policy',type=str, default=default_args['num_policy'],choices=['none','Q_PLE','T_PLE','Q_Unary','T_Unary','Q_bins','T_bins','Q_Johnson','T_Johnson'])
    parser.add_argument('--n_bins', type=int, default=default_args['n_bins'])
    parser.add_argument('--cat_min_frequency', type=float, default=default_args['cat_min_frequency'])

    # other choices
    parser.add_argument('--n_trials', type=int, default=default_args['n_trials'])    
    parser.add_argument('--seed_num', type=int, default=default_args['seed_num'])
    parser.add_argument('--gpu', default=default_args['gpu'])
    parser.add_argument('--tune', action='store_true', default=default_args['tune'])  
    parser.add_argument('--retune', action='store_true', default=default_args['retune'])  
    parser.add_argument('--dataset_path', type=str, default=default_args['dataset_path'])  
    parser.add_argument('--model_path', type=str, default=default_args['model_path'])
    parser.add_argument('--evaluate_option', type=str, default=default_args['evaluate_option']) 
    args = parser.parse_args()
    
    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type])
    
    save_path2 = 'Norm-{}'.format(args.normalization)
    save_path2 += '-Nan-{}-{}'.format(args.num_nan_policy, args.cat_nan_policy)
    save_path2 += '-Cat-{}'.format(args.cat_policy)

    if args.cat_min_frequency > 0.0:
        save_path2 += '-CatFreq-{}'.format(args.cat_min_frequency)
    if args.tune:
        save_path1 += '-Tune'

    save_path = osp.join(save_path1, save_path2)
    args.save_path = osp.join(args.model_path, save_path)
    mkdir(args.save_path)    
    
    # load config parameters
    args.seed = 0
    
    config_default_path = os.path.join('configs','default',args.model_type+'.json')
    config_opt_path = os.path.join('configs','opt_space',args.model_type+'.json')
    with open(config_default_path,'r') as file:
        default_para = json.load(file)  
    
    with open(config_opt_path,'r') as file:
        opt_space = json.load(file)

    args.config = default_para[args.model_type]
    set_seeds(args.seed)
    if torch.cuda.is_available():     
        torch.backends.cudnn.benchmark = True
    pprint(vars(args))
    
    args.config['fit']['n_bins'] = args.n_bins
    return args,default_para,opt_space   

def get_deep_args():  
    """
    Get the arguments for deep learning models.

    :return: argparse.Namespace, arguments
    """
    import argparse 
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    # basic parameters
    with open('configs/deep_configs.json','r') as file:
        default_args = json.load(file)
    parser.add_argument('--dataset', type=str, default=default_args['dataset'])
    parser.add_argument('--model_type', type=str, 
                        default=default_args['model_type'],
                        choices=['mlp', 'resnet', 'ftt', 'node', 'autoint',
                                 'tabpfn', 'tangos', 'saint', 'tabcaps', 'tabnet',
                                 'snn', 'ptarl', 'danets', 'dcn2', 'tabtransformer',
                                 'dnnr', 'switchtab', 'grownet', 'tabr', 'modernNCA',
                                 'hyperfast', 'bishop', 'realmlp', 'protogate', 'mlp_plr',
                                 'excelformer', 'grande','amformer','tabptm','trompt','tabm',
                                 'PFN-v2'
                                 ])
    
    # optimization parameters
    parser.add_argument('--max_epoch', type=int, default=default_args['max_epoch'])
    parser.add_argument('--batch_size', type=int, default=default_args['batch_size'])  
    parser.add_argument('--normalization', type=str, default=default_args['normalization'], choices=['none', 'standard', 'minmax', 'quantile', 'maxabs', 'power', 'robust','MncaPFN'])
    parser.add_argument('--num_nan_policy', type=str, default=default_args['num_nan_policy'], choices=['mean', 'median'])
    parser.add_argument('--cat_nan_policy', type=str, default=default_args['cat_nan_policy'], choices=['new', 'most_frequent'])
    parser.add_argument('--cat_policy', type=str, default=default_args['cat_policy'], choices=['indices', 'ordinal', 'ohe', 'binary', 'hash', 'loo', 'target', 'catboost','tabr_ohe'])
    parser.add_argument('--num_policy',type=str, default=default_args['num_policy'],choices=['none','Q_PLE','T_PLE','Q_Unary','T_Unary','Q_bins','T_bins','Q_Johnson','T_Johnson'])
    parser.add_argument('--n_bins', type=int, default=default_args['n_bins'])  
    parser.add_argument('--cat_min_frequency', type=float, default=default_args['cat_min_frequency'])

    # other choices
    parser.add_argument('--n_trials', type=int, default=default_args['n_trials'])    
    parser.add_argument('--seed_num', type=int, default=default_args['seed_num'])
    parser.add_argument('--workers', type=int, default=default_args['workers'])
    parser.add_argument('--gpu', default=default_args['gpu'])
    parser.add_argument('--tune', action='store_true', default=default_args['tune'])  
    
    parser.add_argument('--retune', action='store_true', default=default_args['retune'])  
    parser.add_argument('--evaluate_option', type=str, default=default_args['evaluate_option'])   
    parser.add_argument('--dataset_path', type=str, default=default_args['dataset_path'])  
    parser.add_argument('--model_path', type=str, default=default_args['model_path'])
    parser.add_argument('--use_float', action='store_true', default=False,
                    help='Whether to use float type for model and data (default: True)')
    args = parser.parse_args()
    
    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type])
    save_path2 = 'Epoch{}BZ{}'.format(args.max_epoch, args.batch_size)
    save_path2 += '-Norm-{}'.format(args.normalization)
    save_path2 += '-Nan-{}-{}'.format(args.num_nan_policy, args.cat_nan_policy)
    save_path2 += '-Cat-{}'.format(args.cat_policy)

    if args.cat_min_frequency > 0.0:
        save_path2 += '-CatFreq-{}'.format(args.cat_min_frequency)
    if args.tune:
        save_path1 += '-Tune'

    save_path = osp.join(save_path1, save_path2)
    args.save_path = osp.join(args.model_path, save_path)
    mkdir(args.save_path)    
    
    # load config parameters
    config_default_path = os.path.join('configs','default',args.model_type+'.json')
    config_opt_path = os.path.join('configs','opt_space',args.model_type+'.json')
    with open(config_default_path,'r') as file:
        default_para = json.load(file)  
    
    with open(config_opt_path,'r') as file:
        opt_space = json.load(file)
    args.config = default_para[args.model_type]
    
    args.seed = 0
    set_seeds(args.seed)
    if torch.cuda.is_available():     
        torch.backends.cudnn.benchmark = True
    pprint(vars(args))
    
    args.config['training']['n_bins'] = args.n_bins
    return args,default_para,opt_space   

def show_results_classical(args,info,metric_name,results_list,time_list):
    """
    Show the results for classical models.

    :args: argparse.Namespace, arguments
    :info: dict, information about the dataset
    :metric_name: list, names of the metrics
    :results_list: list, list of results
    :time_list: list, list of time
    """
    metric_arrays = {name: [] for name in metric_name}  


    for result in results_list:
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])

    metric_arrays['Time'] = time_list
    metric_name = metric_name + ('Time', )

    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    

    # Printing results
    print(f'{args.model_type}: {args.seed_num} Trials')
    for name in metric_name:
        if info['task_type'] == 'regression' and name != 'Time':
            formatted_results = ', '.join(['{:.8e}'.format(e) for e in metric_arrays[name]])
            print(f'{name} Results: {formatted_results}')
            print(f'{name} MEAN = {mean_metrics[name]:.8e} ± {std_metrics[name]:.8e}')
        else:
            formatted_results = ', '.join(['{:.8f}'.format(e) for e in metric_arrays[name]])
            print(f'{name} Results: {formatted_results}')
            print(f'{name} MEAN = {mean_metrics[name]:.8f} ± {std_metrics[name]:.8f}')

    print('-' * 20, 'GPU info', '-' * 20)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU Available.")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}")
            print(f"  Total Memory:          {gpu_info.total_memory / 1024**2} MB")
            print(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
            print(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
    else:
        print("CUDA is unavailable.")
    print('-' * 50)



def show_results(args,info,metric_name,loss_list,results_list,time_list):
    """
    Show the results for deep learning models.

    :args: argparse.Namespace, arguments
    :info: dict, information about the dataset
    :metric_name: list, names of the metrics
    :loss_list: list, list of loss
    :results_list: list, list of results
    :time_list: list, list of time
    """
    metric_arrays = {name: [] for name in metric_name}  


    for result in results_list:
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])

    metric_arrays['Time'] = time_list
    metric_name = metric_name + ('Time', )

    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    mean_loss = np.mean(np.array(loss_list))

    # Printing results
    print(f'{args.model_type}: {args.seed_num} Trials')
    for name in metric_name:
        if info['task_type'] == 'regression' and name != 'Time':
            formatted_results = ', '.join(['{:.8e}'.format(e) for e in metric_arrays[name]])
            print(f'{name} Results: {formatted_results}')
            print(f'{name} MEAN = {mean_metrics[name]:.8e} ± {std_metrics[name]:.8e}')
        else:
            formatted_results = ', '.join(['{:.8f}'.format(e) for e in metric_arrays[name]])
            print(f'{name} Results: {formatted_results}')
            print(f'{name} MEAN = {mean_metrics[name]:.8f} ± {std_metrics[name]:.8f}')

    print(f'Mean Loss: {mean_loss:.8e}')
    
    print('-' * 20, 'GPU info', '-' * 20)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU Available.")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}")
            print(f"  Total Memory:          {gpu_info.total_memory / 1024**2} MB")
            print(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
            print(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
    else:
        print("CUDA is unavailable.")
    print('-' * 50)

def tune_hyper_parameters(args, opt_space, train_val_data, info, cache_path='/work/hdd/bbjr/dbeaglehole/LAMDA-TALENT/'):
    """
    Tune hyper-parameters using TPE.
    Caches results and saves best hyperparameters after each improvement.
    """
    import optuna
    import optuna.samplers
    print("TUNING HYPER-PARAMETERS")

    # Create dataset directory if it doesn't exist
    dataset_dir = osp.join(cache_path, args.dataset)
    mkdir(dataset_dir)
    
    # Cache file path for storing trial results 
    cache_path = osp.join(dataset_dir, f'{args.model_type}-{args.dataset}-trial-cache.json')
    
    # Path for best hyperparameters
    best_params_path = osp.join(args.save_path, f'{args.model_type}-tuned.json')
    
    # Load existing cache if available
    if osp.exists(cache_path):
        with open(cache_path, 'r') as f:
            trial_cache = json.load(f)
    else:
        trial_cache = {}

    # Track best score and config
    best_score = float('inf') if info['task_type'] == 'regression' else float('-inf')
    best_config = None

    def get_config_hash(config):
        """Generate a unique hash for a config dictionary"""
        # Sort the dictionary to ensure consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        return md5(config_str.encode()).hexdigest()

    def save_best_config(config, score, trial_number):
        """Save config if it's the best so far and better than previously saved best"""
        nonlocal best_score, best_config
        is_regression = info['task_type'] == 'regression'
        
        # Try to load existing saved config if it exists
        saved_score = None
        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, 'r') as fp:
                    saved_config = json.load(fp)
                    # Look for saved score in the config
                    saved_score = saved_config.get('_best_score')
            except json.JSONDecodeError:
                print("Warning: Could not read existing best config file")
        
        # Compare against current run's best
        is_better_than_current = score < best_score if is_regression else score > best_score
        
        # Compare against saved best if it exists
        is_better_than_saved = True
        if saved_score is not None:
            is_better_than_saved = score < saved_score if is_regression else score > saved_score
        
        if is_better_than_current and is_better_than_saved:
            best_score = score
            best_config = config
            # Add score to config before saving
            config_to_save = config.copy()
            config_to_save['_best_score'] = score
            config_to_save['trial_number'] = trial_number
            # Save the new best config
            with open(best_params_path, 'w') as fp:
                json.dump(config_to_save, fp, sort_keys=True, indent=4)
            print(f"New best score: {score:.6f} - Saved best config")

    def objective(trial):
        config = {}
        try:
            opt_space[args.model_type]['training']['n_bins'] = [
                    "int",
                    2, 
                    256
            ]
        except:
            opt_space[args.model_type]['fit']['n_bins'] = [
                    "int",
                    2, 
                    256
            ]
        
        merge_sampled_parameters(
            config, sample_parameters(trial, opt_space[args.model_type], config)
        )

        if args.model_type == 'rfm':
            config['model'].setdefault('iters', 4)  
            
        elif args.model_type == 'kernel':
            config['model'].setdefault('iters', 0)

        trial_configs.append(config)
        
        # Generate hash for this config
        config_hash = get_config_hash(config)
        agop_path = osp.join(args.save_path, f'best-sqrtM-{trial.number}.pt')
        config['model']['agop_path'] = agop_path
        # Check if we have cached results
        if config_hash in trial_cache:
            print("Found cached result for this configuration")
            cached_result = trial_cache[config_hash]
            # Check if cached result is best so far
            save_best_config(config, cached_result["result"], trial.number)
            return cached_result["result"]

        try:
            print("Fitting method. Config:", config)
            info['trial'] = trial.number
            
            method.fit(train_val_data, info, train=True, config=config, train_on_subset=True)    
            result = method.trlog['best_res']
            
            # Cache the result
            trial_cache[config_hash] = {
                "trial_number": trial.number,
                "result": result,
                **{k2: v2 for k2, v2 in config['model'].items()}
            }
            
            # Save cache after each new result
            with open(cache_path, 'w') as f:
                json.dump(trial_cache, f, indent=4)
            
            # Check if current result is best so far
            save_best_config(config, result, trial.number)
            
            print("RESULT", result)
            return result
        except Exception as e:
            print("ERROR", e)
            print(f"Full error traceback:\n{traceback.format_exc()}")
            return 1e9 if info['task_type'] == 'regression' else 0.0

    if osp.exists(best_params_path) and args.retune == False:
        with open(best_params_path, 'rb') as fp:
            args.config = json.load(fp)
    else:
        direction = 'minimize' if info['task_type'] == 'regression' else 'maximize'
        print("direction", direction)
        
        sampler = optuna.samplers.TPESampler(seed=0)
        n_trials = args.n_trials

        method = get_method(args.model_type)(args, info['task_type'] == 'regression')      
        trial_configs = []
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )
        
        best_trial_index = study.best_trial.number
        print("BEST TRIAL INDEX", best_trial_index)
        best_config = trial_configs[best_trial_index]

        print('Best Hyper-Parameters')
        print(best_config)
        args.config = best_config
        args.best_trial = best_trial_index
    return args

def get_method(model):
    """
    Get the method class.

    :model: str, model name
    :return: class, method class
    """
    if model == "rfm":
        from model.classical_methods.rfm import RFMMethod
        return RFMMethod
    elif model == "kernel":
        from model.classical_methods.rfm import RFMMethod
        return RFMMethod
    elif model == "mlp":
        from model.methods.mlp import MLPMethod
        return MLPMethod
    elif model == 'resnet':
        from model.methods.resnet import ResNetMethod
        return ResNetMethod
    elif model == 'node':
        from model.methods.node import NodeMethod
        return NodeMethod
    elif model == 'ftt':
        from model.methods.ftt import FTTMethod
        return FTTMethod
    elif model == 'tabptm':
        from model.methods.tabptm import TabPTMMethod
        return TabPTMMethod
    elif model == 'tabpfn':
        from model.methods.tabpfn import TabPFNMethod
        return TabPFNMethod
    elif model == 'tabr':
        from model.methods.tabr import TabRMethod
        return TabRMethod
    elif model == 'modernNCA':
        from model.methods.modernNCA import ModernNCAMethod
        return ModernNCAMethod
    elif model == 'tabcaps':
        from model.methods.tabcaps import TabCapsMethod
        return TabCapsMethod
    elif model == 'tabnet':
        from model.methods.tabnet import TabNetMethod
        return TabNetMethod
    elif model == 'saint':
        from model.methods.saint import SaintMethod
        return SaintMethod
    elif model == 'tangos':
        from model.methods.tangos import TangosMethod    
    elif model == 'snn':
        from model.methods.snn import SNNMethod
        return SNNMethod
    elif model == 'ptarl':
        from model.methods.ptarl import PTARLMethod
        return PTARLMethod
    elif model == 'danets':
        from model.methods.danets import DANetsMethod
        return DANetsMethod
    elif model == 'dcn2':
        from model.methods.dcn2 import DCN2Method
        return DCN2Method
    elif model == 'tabtransformer':
        from model.methods.tabtransformer import TabTransformerMethod
        return TabTransformerMethod
    elif model == 'grownet':
        from model.methods.grownet import GrowNetMethod
        return GrowNetMethod
    elif model == 'autoint':
        from model.methods.autoint import AutoIntMethod
        return AutoIntMethod
    elif model == 'dnnr':
        from model.methods.dnnr import DNNRMethod
        return DNNRMethod
    elif model == 'switchtab':
        from model.methods.switchtab import SwitchTabMethod
        return SwitchTabMethod
    elif model == 'hyperfast':
        from model.methods.hyperfast import HyperFastMethod
        return HyperFastMethod
    elif model == 'bishop':
        from model.methods.bishop import BiSHopMethod
        return BiSHopMethod
    elif model == 'protogate':
        from model.methods.protogate import ProtoGateMethod
        return ProtoGateMethod
    elif model == 'realmlp':
        from model.methods.realmlp import RealMLPMethod
        return RealMLPMethod
    elif model == 'mlp_plr':
        from model.methods.mlp_plr import MLP_PLRMethod
        return MLP_PLRMethod
    elif model == 'excelformer':
        from model.methods.excelformer import ExcelFormerMethod
        return ExcelFormerMethod
    elif model == 'grande':
        from model.methods.grande import GRANDEMethod
        return GRANDEMethod
    elif model == 'amformer':
        from model.methods.amformer import AMFormerMethod
        return AMFormerMethod
    elif model == 'trompt':
        from model.methods.trompt import TromptMethod
        return TromptMethod
    elif model == 'tabm':
        from model.methods.tabm import TabMMethod
        return TabMMethod
    elif model == 'xgboost':
        from model.classical_methods.xgboost import XGBoostMethod
        return XGBoostMethod
    elif model == 'LogReg':
        from model.classical_methods.logreg import LogRegMethod
        return LogRegMethod
    elif model == 'NCM':
        from model.classical_methods.ncm import NCMMethod
        return NCMMethod
    elif model == 'lightgbm':
        from model.classical_methods.lightgbm import LightGBMMethod
        return LightGBMMethod
    elif model == 'NaiveBayes':
        from model.classical_methods.naivebayes import NaiveBayesMethod
        return NaiveBayesMethod
    elif model == 'knn':
        from model.classical_methods.knn import KnnMethod
        return KnnMethod
    elif model == 'RandomForest':
        from model.classical_methods.randomforest import RandomForestMethod
        return RandomForestMethod
    elif model == 'catboost':
        from model.classical_methods.catboost import CatBoostMethod
        return CatBoostMethod
    elif model == 'svm':
        from model.classical_methods.svm import SvmMethod
        return SvmMethod
    elif model == 'dummy':
        from model.classical_methods.dummy import DummyMethod
        return DummyMethod
    elif model == 'LinearRegression':
        from model.classical_methods.lr import LinearRegressionMethod
        return LinearRegressionMethod
    elif model == 'PFN-v2':
        from model.methods.PFN_v2 import TabPFNMethod
        return TabPFNMethod
    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")
