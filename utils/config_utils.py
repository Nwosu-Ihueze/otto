# utils/config_utils.py
import yaml
from dotmap import DotMap
import copy

def load_config(file_path):
    """Loads a YAML file and returns a DotMap object."""
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DotMap(config_dict)

def merge_configs(base_config, search_space_config):
    """Merges the base config with the search space config."""
    full_config = copy.deepcopy(base_config)
    full_config.hyperopt_search_space = search_space_config
    return full_config

def create_trial_config(base_config, trial_params):
    """Creates a specific configuration for a single HPO trial."""
    trial_config = copy.deepcopy(base_config)
    
    
    trial_config.model.params.n_layer = trial_params['n_layer']
    trial_config.model.params.n_head = trial_params['n_head']
    trial_config.model.params.n_embd = trial_params['n_embd']
    trial_config.model.params.dropout = trial_params['dropout']
    trial_config.trainer.batch_size = trial_params['batch_size']
    trial_config.optimizer.params.learning_rate = trial_params['learning_rate']
    trial_config.scheduler.params.min_lr = trial_params['min_lr']

    return trial_config