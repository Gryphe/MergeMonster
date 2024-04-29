import copy
import gc
import math
import torch

from modules.utils import format_context

# Global dictionary to store initial probabilities
initial_phrase_probabilities = {}

def convert_to_new_phrase_format(phrase_list):
    new_format_list = []
    for entry in phrase_list:
        phrase = entry['phrase']
        weight = entry.get('weight', 1)  # Default weight to 1 if not present
        contexts = entry['contexts']

        if isinstance(contexts[0], dict):  # If already in the new format, copy as is
            new_format_list.append(entry)
        else:  # Convert to the new format
            new_contexts = [{"context": context, "weight": weight} for context in contexts]
            new_format_list.append({"phrase": phrase, "contexts": new_contexts})

    return new_format_list

def auto_adjust_weights(model, tokenizer, bad_phrases, good_phrases, device):
    if device != "cuda":
        model_copy = copy.deepcopy(model).to('cuda')
    else:
        model_copy = model

    def adjust_phrase_weights(phrase_list):
        for entry in phrase_list:
            phrase = entry['phrase']
            for context_entry in entry['contexts']:
                context = context_entry['context']
                # Calculate unweighted joint probability
                joint_log_prob = calculate_joint_log_probability(model_copy, tokenizer, context, phrase)
                joint_prob = math.exp(joint_log_prob)

                # Adjust the weight: aiming for each phrase-context pair to have an equal contribution
                # Avoid division by zero; if joint_prob is 0, we can keep the weight unchanged
                if joint_prob > 0:
                    context_entry['weight'] = 1.0 / joint_prob

    # Adjust weights for both bad and good phrases
    adjust_phrase_weights(bad_phrases)
    adjust_phrase_weights(good_phrases)

    if device != "cuda":
        del model_copy
        torch.cuda.empty_cache()

    return bad_phrases, good_phrases

def calculate_joint_log_probability(model, tokenizer, context, phrase):
    sequence = context + phrase
    sequence_input_ids = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(sequence_input_ids)
        logits = outputs.logits

    phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
    joint_log_prob = 0.0
    context_len = len(sequence_input_ids[0]) - len(phrase_tokens)

    for i, token_id in enumerate(phrase_tokens):
        word_log_prob = torch.log_softmax(logits[0, context_len + i - 1], dim=0)[token_id].item()
        joint_log_prob += word_log_prob

    return joint_log_prob

def print_phrase_probabilities(model, tokenizer, bad_phrases, good_phrases, device):
    global initial_phrase_probabilities

    if device != "cuda":
        model_copy = copy.deepcopy(model).to('cuda')
    else:
        model_copy = model

    print("\n-----------------------------------------------------------------------------------------------------")
    print("| Type | Phrase             | Context                  | Raw Prob*    | Used Prob**  | Change       |")
    print("-----------------------------------------------------------------------------------------------------")

    # Initialize sums for good and bad phrases separately
    sums = {
        "BAD": {"real": 0, "weighted": 0, "change": 0},
        "GOOD": {"real": 0, "weighted": 0, "change": 0}
    }
    
    for phrase_type, phrase_list in [("BAD", bad_phrases), ("GOOD", good_phrases)]:
        for entry in phrase_list:
            phrase = entry['phrase']
            for context_entry in entry['contexts']:
                context = context_entry['context']
                weight = context_entry['weight']
                joint_log_prob = calculate_joint_log_probability(model_copy, tokenizer, context, phrase)
                joint_prob = math.exp(joint_log_prob)
                weighted_prob = joint_prob * weight

                # Update the sums
                sums[phrase_type]["real"] += joint_prob
                sums[phrase_type]["weighted"] += weighted_prob

                real_prob_str = f"{joint_prob * 100:.5f}%".ljust(12)

                if weighted_prob < 999999: prob_str = f"{weighted_prob * 100:.2f}%".ljust(12)
                else: prob_str = '###'.ljust(12)

                formatted_context = format_context(context.replace('\n',' '), 24)
                formatted_phrase = format_context(phrase.replace('\n',' '), 18)
                phrase_context_key = (phrase, context)

                if phrase_context_key not in initial_phrase_probabilities:
                    initial_phrase_probabilities[phrase_context_key] = joint_prob
                    print(f"| {phrase_type.ljust(4)} | {formatted_phrase} | {formatted_context} | {real_prob_str} | {prob_str} | {'N/A'.ljust(12)} |")
                else:
                    initial_prob = initial_phrase_probabilities[phrase_context_key]
                    change = ((joint_prob - initial_prob) * 100) * weight
                    sums[phrase_type]["change"] += change

                    if change < 999999: change_str = f"{change:+.2f}%".ljust(12)
                    else: change_str = '###'.ljust(12)
                    
                    print(f"| {phrase_type.ljust(4)} | {formatted_phrase} | {formatted_context} | {real_prob_str} | {prob_str} | {change_str} |")

    # Calculate the net sums and print them
    net_real = sums["GOOD"]["real"] + sums["BAD"]["real"]
    net_weighted = sums["GOOD"]["weighted"] + sums["BAD"]["weighted"]
    net_change = sums["GOOD"]["change"] + sums["BAD"]["change"]

    net_real_str = f"{net_real * 100:.2f}%".ljust(12)
    
    if net_weighted < 999999: net_weighted_str = f"{net_weighted * 100:.2f}%".ljust(12)
    else: net_weighted_str = '###'.ljust(12)
    
    if net_change < 999999: net_change_str = f"{net_change:.2f}%".ljust(12)
    else: net_change_str = '###'.ljust(12)

    print("------------------------------------------------------------------------------------------------------")
    print(f"| {'Totals'.ljust(52)} | {net_real_str} | {net_weighted_str} | {net_change_str} |")
    print("------------------------------------------------------------------------------------------------------")
    print("* = Unweighted, raw probability - ** = Probability after weight adjustments\n")

    if device != "cuda":
        del model_copy
        torch.cuda.empty_cache()
        gc.collect()


def calculate_word_probabilities(model, tokenizer, bad_phrases, good_phrases, device):
    if device != "cuda":
        model_copy = copy.deepcopy(model).to('cuda')
    else:
        model_copy = model

    phrase_probs = []

    for phrase_list, sign in [(bad_phrases, 1), (good_phrases, -1)]:
        for entry in phrase_list:
            phrase = entry['phrase']
            for context_entry in entry['contexts']:
                context = context_entry['context']
                weight = context_entry['weight']
                joint_log_prob = calculate_joint_log_probability(model_copy, tokenizer, context, phrase)
                joint_prob = math.exp(joint_log_prob)
                weighted_prob = joint_prob * weight * sign
                phrase_probs.append((phrase, context, weighted_prob))

    if device != "cuda":
        del model_copy
        torch.cuda.empty_cache()
        gc.collect()

    return phrase_probs