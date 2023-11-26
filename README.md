# Merge Monster
An unsupervised merging algorithm for Transformers-based language models, using a list of phrases (both good and bad) and a fully automated strategy that strives to decrease (or increase) the probability of these phrases occuring in the final merge.

**Refer to the default.yaml example configuration for an explanation of all potential configuration options.**

## How It Works

1. The algorithm loads a base model of your own choosing (model 1), along with a directory (or repo list) containing multiple models of the same architecture and size.
2. Each model from the directory is loaded one by one and merged with model 1 on a layer-by-layer basis.
3. For each layer merge, the algorithm evaluates whether the merge improves the base model, iterating along a customizable list of merging ratios. (See YAML)
   - If the merge is beneficial (it lowers the cumulative probability), it is permanently applied to model 1.
   - If not, model 1 retains its current structure, and the algorithm proceeds to the next layer.
4. Upon completing the merge process with one model, the algorithm proceeds to the next model in the list, repeating the cycle.
5. After all models have been processed, the algorithm saves the final merged model and generates a complete log of the entire process.

At its very core Merge Monster is nothing more but a relentless number chaser - It tries to decrease the probability of unwanted completions. Wanted completions also subtract from that same number (the monster only cares about lowering the total number), which is why the number displayed during processing might go negative.

## Merge Strategies

Merge strategies have been added and can be defined in the YAML configuration.

- **"cumulative"** - Default strategy. If there's a chance of reducing the combined probability, accept the merge.
- **"all_phrases"** - Only accept the merge if all phrases show an improvement. (Warning: This rarely happens)
- **"quantitive"** - Ignores probabilities completely. Only looks at how many phrases show an improvement, as defined by the **strategy_threshold** variable.

## Why This Might Be A Big Deal

We can now set clear goals for an algorithm to pursue based on the things that truly matter - The actual output of a model. While the included example may be more focused on reducing GPTisms the possibilities are potentially endless. Anything can be used as a phrase and the merge monster will happily chase it down, as it is truly relentless.

## Requirements

This script, when configured in "cpu" mode, requires the presence of a CUDA-capable card with the capacity to store **at least a single model** with float16 precision in its memory.

When configured in "cuda" mode it requires enough VRAM to store **3 copies of a fp16 model**.

For Mistral 7B v0.1 this translates itself to either a 15 GB VRAM (1x) or a 45 GB VRAM (3x) requirement.

## Usage

1. **Prepare your configuration file (or modify the included one)**: Create a YAML file with your model paths, device settings, and other configurations.
2. **Run the script**: Use the following command to start the merging process:

    ```bash
    python merge_monster.py --config your_config_file.yaml
    ```
If no --config argument is given, it will instead revert to loading default.yaml.

3. **Evaluate and save**: The script will automatically evaluate, log, and save the best-performing merged model. A copy of the log will be saved in the same folder as the script.
