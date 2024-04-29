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
from modules.mapping import map_contexts
from modules.models import load_model, NoInit
from modules.probability import print_phrase_probabilities, convert_to_new_phrase_format

def monster_mapper(config_path):
    # We save everything that gets printed to the screen
    original_stdout = sys.stdout
    logger = PrintAndStoreLogger(original_stdout)
    sys.stdout = logger  # Redirect stdout to the logger instance
    
    config = load_config(config_path)

    if 'device' in config: device = config['device']
    else: device = ['cpu']

    model_path1 = config['directories']['model_path1']

    # Phrase config
    if 'bad_phrases' in config: bad_phrases = config['bad_phrases']
    else: bad_phrases = []

    if 'good_phrases' in config: good_phrases = config['good_phrases']
    else: good_phrases = []

    # Seed config
    if 'random_seed' in config: random_seed = config['random_seed']
    else: random_seed = 512

    # Mapper specific options
    if 'prob_min' in config['mapper']: prob_min = config['mapper']['prob_min']
    else: prob_min = 10
    if 'top_k' in config['mapper']: top_k = config['mapper']['top_k']
    else: top_k = 3
    if 'max_depth' in config['mapper']: max_depth = config['mapper']['max_depth']
    else: max_depth = 10
    if 'additional_length' in config['mapper']: additional_length = config['mapper']['additional_length']
    else: additional_length = 20

    # Actual start of script
    print_ascii_art("modules/logo.ascii")
    print(f"{datetime.now().strftime('%H:%M:%S')} - MONSTER CONTEXT MAPPER")
    print("------------------------------------")
    print(f"Device                 : {device}")
    print(f"Random seed            : {random_seed}")
    print(f"Model to map           : {model_path1}")
    print(f"Phrases loaded         : {len(bad_phrases)+len(good_phrases)}")
    print("------------------------------------")
    print(f"Minimum branching prob : {prob_min}%")
    print(f"Top # per branch       : {top_k}")
    print(f"Max branch depth       : {max_depth}")
    print(f"Extra tokens generated : {additional_length}")

    with torch.no_grad():
        if device == "cpu": torch.set_default_dtype(torch.float32)
        else: torch.set_default_dtype(torch.float16)
            
        torch.set_default_device(device)

        # Setting all the seeds
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Testing the output model
        # model_path1 = output_directory

        # Load the base model + tokenizer
        model1 = load_model(model_path1, device)
        model1name = model_path1.split('/')[-1]
        header_chosen = [1.0, model1name]
        
        tokenizer = AutoTokenizer.from_pretrained(model_path1)
        tokenizer.padding_side = 'left'

        # Convert to new internal phrase format
        bad_phrases = convert_to_new_phrase_format(bad_phrases)
        good_phrases = convert_to_new_phrase_format(good_phrases)

        if device != "cuda":
            model1 = model1.to('cuda')

        # Mapping time!
        map_contexts(model1, tokenizer, bad_phrases, good_phrases, prob_min, top_k, max_depth, additional_length)

        # Save log
        sys.stdout = original_stdout  # Restore the original stdout
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create 'logs' subdirectory if it doesn't exist
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Define the log file path
        log_file_path = os.path.join(logs_dir, f"monster-mapper-{timestamp}.txt")
        
        # Write the log contents to the file in the 'logs' subdirectory
        with open(log_file_path, "w") as file:
            file.write(logger.contents)

def main():
    parser = argparse.ArgumentParser(description="Gryphe's Mythical Monster Mapper")
    parser.add_argument('--config', type=str, default='default.yaml', help='Path to the config YAML file')
    args = parser.parse_args()

    monster_mapper(args.config)

if __name__ == "__main__":
    main()