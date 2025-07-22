# training/hyperopt_manager.py
import optuna
import os
import json
from .trainer import TrialTrainer
from utils.config_utils import create_trial_config

class HyperoptManager:
    def __init__(self, config, data_info, output_dir, n_trials=20):
        self.config = config
        self.data_info = data_info
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.study_path = os.path.join(output_dir, "optuna_study.db")
        self.study_name = "automl_slm_study"
        self._precompute_valid_architectures()

    def _precompute_valid_architectures(self):
        """
        Pre-computes the valid (n_embd, n_head) pairs to create a static
        search space for Optuna.
        """
        search_space = self.config.hyperopt_search_space
        n_embd_choices = search_space.model_params.n_embd['values']
        n_head_choices = search_space.model_params.n_head['values']
        
        self.valid_architectures = []
        for n_embd in n_embd_choices:
            for n_head in n_head_choices:
                if n_embd % n_head == 0:
   
                    self.valid_architectures.append(f"{n_embd}_{n_head}")
        
        print(f"Generated {len(self.valid_architectures)} valid n_embd/n_head combinations for the search space.")

    def run(self):
        print("\nStarting Hyperparameter Optimization...")

        def objective(trial):

            arch_str = trial.suggest_categorical('architecture', self.valid_architectures)
            n_embd, n_head = map(int, arch_str.split('_'))
            

            search_space = self.config.hyperopt_search_space

            trial_params = {
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': trial.suggest_categorical('n_layer', search_space.model_params.n_layer['values']),
                'dropout': trial.suggest_float('dropout', search_space.model_params.dropout['lower'], search_space.model_params.dropout['upper']),
                'batch_size': trial.suggest_categorical('batch_size', search_space.trainer_params.batch_size['values']),
                'learning_rate': trial.suggest_float('learning_rate', search_space.optimizer_params.learning_rate['lower'], search_space.optimizer_params.learning_rate['upper'], log=True),
                'min_lr': trial.suggest_float('min_lr', search_space.scheduler_params.min_lr['lower'], search_space.scheduler_params.min_lr['upper'], log=True),
            }


            trial_config = create_trial_config(self.config, trial_params)
            

            trainer = TrialTrainer(trial_config, self.data_info, trial.number, self.output_dir)

            val_loss = trainer.train(trial) 
            
            return val_loss

        storage = f"sqlite:///{self.study_path}"

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=2)
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=pruner
        )
        
        try:
            study.optimize(objective, n_trials=self.n_trials, catch=(Exception,))
        except Exception as e:
            print(f"An unexpected error occurred during study optimization: {e}")
            raise

        print("\nHyperparameter Optimization Complete.")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("No trials were completed successfully. Cannot determine the best model.")
            return None

        best_trial = study.best_trial
        
        best_trial_info = {
            "trial_number": best_trial.number,
            "value (val_loss)": best_trial.value,
            "params": best_trial.params
        }
        with open(os.path.join(self.output_dir, "best_trial.json"), "w") as f:
            json.dump(best_trial_info, f, indent=4)
        
        return best_trial