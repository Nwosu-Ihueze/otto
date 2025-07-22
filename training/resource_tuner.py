# training/resource_tuner.py
import torch
import copy

class ResourceTuner:
    def tune_for_memory(self, config):
        if not torch.cuda.is_available():
            print("WARNING: No GPU detected. Skipping memory tuning.")
            return config
        
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU Memory: {available_memory_gb:.2f} GB")

        tuned_config = copy.deepcopy(config)
        search_space = tuned_config.hyperopt_search_space
        
        params_to_tune = ['batch_size', 'n_embd', 'n_layer']
        
        for i in range(10):
            max_params = self._get_max_hyperparams(search_space)
            estimated_gb = self._estimate_memory_usage_gb(max_params, tuned_config)

            print(f"Checking max config: {max_params}. Estimated VRAM: {estimated_gb:.2f} GB")

            if estimated_gb <= available_memory_gb * 0.90: 
                print("Max configuration fits in memory. No changes needed to search space.")
                return tuned_config
            
            found_param_to_shrink = False
            for param_name in params_to_tune:
                space_key = 'trainer_params' if param_name == 'batch_size' else 'model_params'
                param_space = search_space[space_key][param_name]
                
                if param_space['type'] == 'choice' and len(param_space['values']) > 1:
                    param_space['values'].pop() 
                    print(f"Reduced search space for '{param_name}'. New choices: {param_space['values']}")
                    found_param_to_shrink = True
                    break
            
            if not found_param_to_shrink:
                raise RuntimeError("Cannot shrink search space further to fit in memory. The smallest model is too large.")
        
        raise RuntimeError("Failed to find a memory-fitting configuration after 10 attempts.")

    def _get_max_hyperparams(self, search_space):
        max_params = {}
        for section in search_space.values():
            for param_name, space in section.items():
                if space['type'] == 'choice':
                    max_params[param_name] = max(space['values'])
                elif space['type'] in ['uniform', 'loguniform']:
                    max_params[param_name] = space['upper']
        return max_params

    def _estimate_memory_usage_gb(self, params, base_config):
  
        C = params['n_embd']
        L = params['n_layer']
        V = base_config.model.params.vocab_size
        T = base_config.data.block_size
        B = params['batch_size']
        G = base_config.trainer.gradient_accumulation_steps

        
        model_params = (L * 12 * C**2) + (V * C)
        model_mem_gb = (model_params * 2) / (1024**3)

        
        optimizer_mem_gb = model_mem_gb * 2
        
        # Activation memory (highly dependent on batch size and context length)
        # Formula from: https://medium.com/mlearning-ai/transformer-memory-analysis-and-reduction-1d7b3a245c3b
        activation_mem_gb = (B * T * C * L * (10 + 24/G + 5*params['n_head']*T/C)) * 2 / (1024**3)

        return model_mem_gb + optimizer_mem_gb + activation_mem_gb