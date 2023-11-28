import argparse
import copy
import gc
import os
import random
import sys
import torch
import shutil
import transformers
import yaml

from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.utils import print_ascii_art, format_context, load_config, PrintAndStoreLogger
from modules.models import load_model, save_model, NoInit
from modules.probability import calculate_word_probabilities, print_phrase_probabilities, convert_to_new_phrase_format, auto_adjust_weights
from modules.composition import calculate_final_composition, aggregate_composition
from modules.merging import merge_tensors, merge_header_tensors

def merge_monster(config_path):
    original_stdout = sys.stdout
    logger = PrintAndStoreLogger(original_stdout)
    sys.stdout = logger  # Redirect stdout to the logger instance
    
    config = load_config(config_path)

    device = config['device']
    model_path1 = config['directories']['model_path1']
    output_directory = config['directories']['output_directory']

    if 'models_to_merge' in config: models_to_merge = config['models_to_merge']
    else: models_to_merge = []

    if 'model_directory' in config['directories']: model_directory = config['directories']['model_directory']
    else: model_directory = []

    if len(models_to_merge) == 0 and len(model_directory) == 0:
        sys.exit("ERROR: No model directory or models to merge variable has been found in the YAML config.")

    if 'bad_phrases' in config: bad_phrases = config['bad_phrases']
    else: bad_phrases = []

    if 'good_phrases' in config: good_phrases = config['good_phrases']
    else: good_phrases = []

    if 'merge_ratios' in config: merge_ratios = config['merge_ratios']
    else: merge_ratios = [0.2, 0.4, 0.6, 0.8]

    if 'merge_method' in config: merge_method = config['merge_method']
    else: merge_method = "lerp"

    if merge_method not in ["lerp", "slerp"]:
        sys.exit("ERROR: Please use a valid merging method! (lerp/slerp)")

    if 'merge_headers' in config: merge_headers = config['merge_headers']
    else: merge_headers = True

    if 'random_seed' in config: random_seed = config['random_seed']
    else: random_seed = 512

    if 'auto_weights' in config: auto_weights = config['auto_weights']
    else: auto_weights = False

    if 'strategy' in config: strategy = config['strategy']
    else: strategy = "cumulative"
    if 'strategy_threshold' in config: strategy_threshold = config['strategy_threshold']
    else: strategy_threshold = 0.6

    print_ascii_art("modules/logo.ascii")
    print(f"{datetime.now().strftime('%H:%M:%S')} - THE MERGE MONSTER HUNGERS")
    print("------------------------------------")
    print(f"Device           : {device}")
    print(f"Random seed      : {random_seed}")
    print(f"Starting model   : {model_path1}")

    if len(models_to_merge) > 0:
        print(f"Models to merge  : {models_to_merge}")
    else:
        print(f"Model directory  : {model_directory}")
    
    print(f"Output directory : {output_directory}")
    print(f"Phrases loaded   : {len(bad_phrases)+len(good_phrases)}")
    print(f"Auto weights     : {auto_weights}")
    print(f"Merge ratios     : {merge_ratios}")
    print(f"Merge method     : {merge_method}")
    print(f"Merge headers    : {merge_headers}")
    print(f"Strategy used    : {strategy}")

    with torch.no_grad():
        if device == "cpu": torch.set_default_dtype(torch.float32)
        else: torch.set_default_dtype(torch.float16)
            
        torch.set_default_device(device)
    
        # Seed 512
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load the base model
        model1 = load_model(model_path1, device)
        model1name = model_path1.split('/')[-1]
        header_chosen = [1.0, model1name]
        
        tokenizer = AutoTokenizer.from_pretrained(model_path1)

        # Convert to new phrase format
        bad_phrases = convert_to_new_phrase_format(bad_phrases)
        good_phrases = convert_to_new_phrase_format(good_phrases)

        if auto_weights == True:
            bad_phrases, good_phrases = auto_adjust_weights(model1, tokenizer, bad_phrases, good_phrases, device)

        # Let's get our starting probabilities
        print_phrase_probabilities(model1, tokenizer, bad_phrases, good_phrases, device)

        # Get a list of all model paths in the directory, or otherwise just use the list of repo's
        if len(models_to_merge) > 0:
            model_paths = models_to_merge
        else:
            model_paths = [os.path.join(model_directory, f) for f in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, f)) and f.startswith('.') == False]

        # Create our origins dict
        layer_origins = {}

        # How many layers we have to iterate through
        layerCount = model1.config.num_hidden_layers

        # Pre-populate our layer origins dict at startup
        for i in range(layerCount):
            layer_origins[i] = [[1.0, model1name]]
        layer_origins[999] = [[1.0, model1name]]

        # Start of the main monster loop
        for model_path2 in model_paths:
            model2name = model_path2.split('/')[-1]
            
            # Avoid merging the same model
            if model_path2 == model_path1:
                continue
    
            model2 = load_model(model_path2, device)

            model2.config.eos_token_id = model1.config.eos_token_id

            # Debugging purposes
            skip_layers = False
            
            # Start of layer processing loop
            for i in range(layerCount):
                if skip_layers == False:
                    # Save a copy of the unchanged dict at start, otherwise probabilities get messed up
                    model1dict = copy.deepcopy(model1.model.state_dict())
                    
                    orig_probs = calculate_word_probabilities(model1, tokenizer, bad_phrases, good_phrases, device)
                    best_probs = orig_probs
                    best_layer = model1.model.layers[i].state_dict()
                    best_ratio = 1.0
                    layer_changed = False
    
                    # We go along the scale of ratios and test each possibility
                    for ratio in tqdm(merge_ratios, desc="Testing Merge Ratios"):
                        layer1 = model1.model.layers[i].state_dict()
                        layer2 = model2.model.layers[i].state_dict()
                        merged_layer = layer1
                        
                        for key in merged_layer.keys():
                            merged_layer[key] = merge_tensors(merge_method, layer1[key], layer2[key], ratio)
    
                        # Restore our original dict copy, otherwise probabilities get messed up - Very expensive in terms of efficiency, but necessary
                        model1.model.load_state_dict(model1dict)
                        model1.model.layers[i].load_state_dict(merged_layer)
    
                        new_probs = calculate_word_probabilities(model1, tokenizer, bad_phrases, good_phrases, device)
                
                        if strategy == "cumulative":
                            if sum(p for _, _, p in new_probs) < sum(p for _, _, p in best_probs):
                                best_probs = new_probs
                                best_layer = merged_layer
                                best_ratio = ratio
                                layer_changed = True
                        elif strategy == "all_phrases":
                            if all(new_p <= orig_p for (_, _, new_p), (_, _, orig_p) in zip(new_probs, orig_probs)):
                                best_probs = new_probs
                                best_layer = merged_layer
                                best_ratio = ratio
                                layer_changed = True
                        elif strategy == "quantitive":
                            improved_phrases = 0
                            regressed_phrases = 0
                            total_phrases = len(new_probs)  # Total number of phrases
                        
                            for (_, _, new_prob), (_, _, orig_prob) in zip(new_probs, orig_probs):
                                if new_prob < orig_prob:
                                    improved_phrases += 1
                                elif new_prob > orig_prob:
                                    regressed_phrases += 1
                        
                            # Decision Criteria
                            improvement_ratio = improved_phrases / total_phrases
                            if improvement_ratio >= strategy_threshold:
                                # Accept the merge
                                best_probs = new_probs
                                best_layer = merged_layer
                                best_ratio = ratio
                                layer_changed = True
                        
                    # Update/retain the model state dictionary with the best performing layer, using our clean dict
                    model1.model.load_state_dict(model1dict)
                    model1.model.layers[i].load_state_dict(best_layer)
    
                    #print(torch.cuda.memory_summary())
    
                    if layer_changed == True:
                        layer_origins[i].append([best_ratio, model2name])
                        layer_changed_label = 'CHANGED'
                    else:
                        layer_changed_label = 'RETAINED'
    
                    del model1dict
                    del best_layer
                    torch.cuda.empty_cache()
                    gc.collect()
    
                    best_prob = sum(prob for _, _, prob in best_probs)
                    orig_prob = sum(prob for _, _, prob in orig_probs)
    
                    print(layer_origins[i])
    
                    if layer_changed_label == 'CHANGED':
                        print(f"{datetime.now().strftime('%H:%M:%S')} - Layer {i+1}/{layerCount} - {layer_changed_label} - {(orig_prob):.5f} > {(best_prob):.5f} - {abs(((best_prob - orig_prob) / orig_prob * 100)):.1f}%")
                    else:
                        print(f"{datetime.now().strftime('%H:%M:%S')} - Layer {i+1}/{layerCount} - {layer_changed_label} - {(best_prob):.5f}")

            # -------------------------------------------------------------------------------------------------------
            # START OF HEADER OPTIMIZATION LOOP
            # -------------------------------------------------------------------------------------------------------

            if merge_headers == True:
                # As befores, save a copy of the unchanged dict at start, otherwise probabilities get messed up
                model1dict = copy.deepcopy(model1.model.state_dict())

                orig_probs = calculate_word_probabilities(model1, tokenizer, bad_phrases, good_phrases, device)
                best_probs = orig_probs
                best_header = model1.state_dict()['lm_head.weight']
                best_vocab = model1.state_dict()['model.embed_tokens.weight']
                best_ratio = 1.0
                header_changed = False
    
                # We go along the scale of ratios and test each possibility
                for ratio in tqdm(merge_ratios, desc="Optimizing Header"):
                    # Restore our original dict copy, otherwise probabilities get messed up - Very expensive in terms of efficiency, but necessary
                    model1.model.load_state_dict(model1dict)
                    
                    current_header = merge_header_tensors(model1, model2, merge_method, model1.state_dict()['lm_head.weight'], model2.state_dict()['lm_head.weight'], ratio)
                    current_vocab = merge_header_tensors(model1, model2, merge_method, model1.state_dict()['model.embed_tokens.weight'], model2.state_dict()['model.embed_tokens.weight'], ratio)
    
                    # Directly modify the weights of the model
                    model1.lm_head.weight.data = current_header
                    model1.model.embed_tokens.weight.data = current_vocab
    
                    new_probs = calculate_word_probabilities(model1, tokenizer, bad_phrases, good_phrases, device)

                    if strategy == "cumulative":
                        if sum(p for _, _, p in new_probs) < sum(p for _, _, p in best_probs):
                            best_probs = new_probs
                            best_header = current_header
                            best_vocab = current_vocab
                            best_ratio = ratio
                            header_changed = True
                    elif strategy == "all_phrases":
                        if all(new_p <= orig_p for (_, _, new_p), (_, _, orig_p) in zip(new_probs, orig_probs)):
                            best_probs = new_probs
                            best_header = current_header
                            best_vocab = current_vocab
                            best_ratio = ratio
                            header_changed = True
                    elif strategy == "quantitive":
                        improved_phrases = 0
                        regressed_phrases = 0
                        total_phrases = len(new_probs)  # Total number of phrases
                    
                        for (_, _, new_prob), (_, _, orig_prob) in zip(new_probs, orig_probs):
                            if new_prob < orig_prob:
                                improved_phrases += 1
                            elif new_prob > orig_prob:
                                regressed_phrases += 1
                    
                        # Decision Criteria
                        improvement_ratio = improved_phrases / total_phrases
                        if improvement_ratio >= strategy_threshold:
                            # Accept the merge
                            best_probs = new_probs
                            best_header = current_header
                            best_vocab = current_vocab
                            best_ratio = ratio
                            header_changed = True

                if header_changed == True:
                    layer_origins[999].append([best_ratio, model2name])
                    header_changed_label = 'CHANGED'
                else:
                    header_changed_label = 'RETAINED'

                best_prob = sum(prob for _, _, prob in best_probs)
                orig_prob = sum(prob for _, _, prob in orig_probs)

                print(layer_origins[999])

                if header_changed_label == 'CHANGED':
                    print(f"{datetime.now().strftime('%H:%M:%S')} - Header - {header_changed_label} - {(orig_prob):.5f} > {(best_prob):.5f} - {(abs((best_prob - orig_prob) / orig_prob * 100)):.1f}%") 
                else:
                    print(f"{datetime.now().strftime('%H:%M:%S')} - Header - {header_changed_label} - {(best_prob):.5f}") 
                    
                # Update/retain the model state dictionary with the best performing headers, using our clean dict
                model1.model.load_state_dict(model1dict)
                model1.lm_head.weight.data = best_header
                model1.model.embed_tokens.weight.data = best_vocab
    
                del best_header
                del best_vocab
                del model1dict
                torch.cuda.empty_cache()
                gc.collect()
        
            # -------------------------------------------------------------------------------------------------------
            # END OF HEADER OPTIMIZATION LOOP
            # -------------------------------------------------------------------------------------------------------

            del model2
            torch.cuda.empty_cache()
            gc.collect()

            # Calculate final composition for each layer
            final_layer_composition = calculate_final_composition(layer_origins)
            
            # Aggregate the composition across all layers
            aggregated_composition = aggregate_composition(final_layer_composition)

            print_phrase_probabilities(model1, tokenizer, bad_phrases, good_phrases, device)
            
            # Display the aggregated composition
            print("-------- MERGE COMPOSITION ---------")
            for model_name, ratio in aggregated_composition.items():
                print(f"{model_name}: {ratio:.2f}")
            print("")

        # Save final model
        save_model(output_directory, model_path1, model1)

        # Save log
        sys.stdout = original_stdout  # Restore the original stdout
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create 'logs' subdirectory if it doesn't exist
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Define the log file path
        log_file_path = os.path.join(logs_dir, f"merge-monster-{timestamp}.txt")
        
        # Write the log contents to the file in the 'logs' subdirectory
        with open(log_file_path, "w") as file:
            file.write(logger.contents)

def main():
    parser = argparse.ArgumentParser(description="Gryphe's Mythical Merge Monster")
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to the config YAML file')
    args = parser.parse_args()

    merge_monster(args.config)

if __name__ == "__main__":
    main()