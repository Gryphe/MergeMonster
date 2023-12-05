import colorama

import math
import re
import torch

from colorama import Fore, Style
from tqdm import tqdm

colorama.init(autoreset=True)

def map_contexts(model, tokenizer, bad_phrases, good_phrases, prob_min, top_k, max_depth, additional_length):
    unique_contexts = set()
    
    # Collect unique contexts from bad_phrases
    for entry in bad_phrases:
        for context_entry in entry['contexts']:
            if len(context_entry['context']) > 0:
                unique_contexts.add(context_entry['context'])
    
    # Collect unique contexts from good_phrases
    for entry in good_phrases:
        for context_entry in entry['contexts']:
            if len(context_entry['context']) > 0:
                unique_contexts.add(context_entry['context'])

    # Generate token probabilities for each unique context
    for context in unique_contexts:
        context_cleaned = re.sub(' +', ' ', context.replace('\n', ' ').strip())
        
        print("\n------------------------------------")
        print(f"CONTEXT: {context_cleaned}")
        print("------------------------------------")
        
        generate_token_probabilities(model, tokenizer, context, top_k=top_k, max_depth=max_depth, additional_length=additional_length, prob_min=prob_min)

def recursive_explore_and_generate(model, tokenizer, input_ids, prob_min, top_k, depth, max_depth, additional_length, device, path=[], pbar=None):
    # Initialize tqdm progress bar at the top level of recursion
    if pbar is None and depth == 0:
        # Initial total can be less than the maximum possible count
        initial_total = sum(math.pow(top_k, i) for i in range(depth + 1))
        pbar = tqdm(total=initial_total, desc="Mapping tokens", leave=False)
    
    logits = model(input_ids).logits
    log_probabilities = torch.log_softmax(logits[0, -1], dim=0)
    top_k_tokens = torch.topk(log_probabilities, top_k).indices

    extended_token_probabilities = []
    branches_explored = 0

    for token in top_k_tokens:
        token_prob = math.exp(log_probabilities[token].item())

        # Update the tqdm progress bar
        pbar.update(1)

        # Only proceed if the token probability is greater than or equal to 10%
        if token_prob * 100 >= prob_min:
            branch_input_ids = torch.cat((input_ids, token.unsqueeze(0).unsqueeze(0)), dim=1)
            new_path = path + [(token.item(), token_prob)]
            branches_explored += 1

            if depth < max_depth:
                deeper_token_probs = recursive_explore_and_generate(model, tokenizer, branch_input_ids, prob_min, top_k, depth + 1, max_depth, additional_length, device, new_path, pbar)
                extended_token_probabilities.extend(deeper_token_probs)
            else:
                for _ in range(additional_length):
                    logits = model(branch_input_ids).logits
                    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                    branch_input_ids = torch.cat((branch_input_ids, next_token_id), dim=1)

                sequence = tokenizer.decode(branch_input_ids[0], skip_special_tokens=True)
                sequence_cleaned = re.sub(' +', ' ', sequence.replace('\n', ' ').strip())
                extended_token_probabilities.append((new_path, sequence_cleaned))
        elif depth == max_depth - 1:
            pbar.update(1)

    # Adjust the total count of pbar based on actual branches explored
    if depth == 0 and branches_explored < top_k:
        pbar.total -= (top_k - branches_explored) * math.pow(top_k, max_depth - depth - 1)
        pbar.refresh()

    # Close the tqdm progress bar when the top-level recursion is finished
    if depth == 0:
        pbar.close()

    return extended_token_probabilities

def colorize_probability(prob):
    """Colorize the probability percentage with a green-yellow-orange-red scale."""
    if prob >= 80.0:
        color = Fore.LIGHTGREEN_EX
    elif prob >= 60.0:
        color = Fore.GREEN
    elif prob >= 40.0:
        color = Fore.LIGHTYELLOW_EX
    elif prob >= 20.0:
        color = Fore.YELLOW
    elif prob >= 10.0:
        color = Fore.LIGHTRED_EX
    else:
        color = Fore.RED
    return f'{color}{prob:.2f}%'

def display_token_probabilities(token_probabilities, tokenizer):
    for path, sequence in token_probabilities:
        # Construct the probability path string with tokens
        prob_path_str = " > ".join([
            f'{colorize_probability(token_prob * 100)}% [{tokenizer.decode([token], skip_special_tokens=True)}]'
            for token, token_prob in path
        ])
        prob_path_str = prob_path_str.replace('\n', '\\n')
        print(f"Sequence: {sequence}\nTrace: {prob_path_str}{Fore.RESET}\n---")

def generate_token_probabilities(model, tokenizer, context="", prob_min=10, top_k=3, max_depth=5, additional_length=20, device='cuda'):
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    explored_and_extended_contexts = recursive_explore_and_generate(model, tokenizer, input_ids, prob_min, top_k, 0, max_depth, additional_length, device, [])
    display_token_probabilities(explored_and_extended_contexts, tokenizer)


