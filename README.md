# MergeMonster
An unsupervised merging algorithm for Transformers-based language models, using a list of phrases (both good and bad) and a fully automated strategy that strives to decrease (or increase) the probability of these phrases occurring in the final merge.

**Refer to the default.yaml example configuration for an explanation of all potential configuration options.**

## NEW: MonsterMapper
MonsterMapper is a companion tool to MergeMonster that reuses its configuration files to check the first model's probabilities for the defined contexts. At its core it basically just lists the most probable auto-completions for the contexts you entered in your phrase dictionaries so that you can make the necessary changes to increase the algorithm's efficiency. 

Perhaps the phrase you originally defined isn't as common as you thought, but there's a much more common auto-completion that matches your intended goal. MonsterMapper will help you discover this.

A seperate config (default-mapper.yaml) has been made available as this tool has its own set of exclusive options to configure as needed. 

## How It Works

1. The algorithm loads a base model of your own choosing (model 1), along with a directory (or repo list) containing multiple models of the same architecture and size.
2. Each model from the directory is loaded one by one and merged with model 1 on a layer-by-layer basis.
3. For each layer merge, the algorithm evaluates whether the merge improves the base model, iterating along a customizable list of merging ratios. (See YAML)
   - If the merge is beneficial (it lowers the cumulative probability), it is permanently applied to model 1.
   - If not, model 1 retains its current structure, and the algorithm proceeds to the next layer.
4. Upon completing the merge process with one model, the algorithm proceeds to the next model in the list, repeating the cycle.
5. After all models have been processed, the algorithm saves the final merged model and generates a complete log of the entire process.

At its very core Merge Monster is nothing more but a relentless number chaser - It tries to decrease the probability of unwanted completions. Wanted completions also subtract from that same number (the monster only cares about lowering the total number), which is why the number displayed during processing might go negative.

## Progress bar layout
```
[[1.0, 'Mistral-7B-v0.1'], [0.5, 'Nous-Capybara-7B-V1.9'], [0.25, 'SynthIA-7B-v1.3'], [0.1, 'zephyr-7b-beta']]
09:19:57 - Layer 3/32 - CHANGED - 0.38319 > 0.38261 - 0.2%

[List of merges applied to this layer, with the first being model 1]
Current time - Layer progress - CHANGED/RETAINED - Old total probability > New total probability - Global change in percentage
```
## Merge Methods

New merging methods have been added (slice/cyclic) to help target specific parts of tensors. These are highly experimental but verified to be fully functional.

**Multiple merging methods**: It is possible to combine multiple merging methods by supplying a list to the `merge_method` parameter, such as `["lerp","cyclic"]`. The loop will look like `LAYER > LERP > CYCLIC > NEXT LAYER`.

**The following methods are currently available:**

### "lerp"

Default method. Linear Interpolation, your basic merging method.

### "slerp"

- Spherical Linear Interpolation, which better aligns the weights inside the two tensors.
- Full credit to [Charles Goddard's mergekit](https://github.com/cg123/mergekit) for the Slerp function.

### "slice"

- Highly experimental. A shift from one model to another, with a smooth 10% transition in-between.
- Ratio defines the starting point of the transition.

![Slice](images/slice.png?raw=true "Slice")

### "cyclic"

- Highly experimental. A shift from one model to the other, then back to the original model.
- Especially useful for when you wish to target specific parts of model 1's tensors as model 2 only has a 15% contribution in the resulting output tensor.
- Merge ratios are ignored and a predefined scale is used that covers the entire spectrum during the optimization process.

![Cyclic](images/cyclic.png?raw=true "Cyclic")

### "gradient"

- Highly experimental, but has displayed some remarkable effectiveness so far during testing.
- Ratio defines a 90% opacity peak from which model 2 forms a gradient to model 1 on either side of the spectrum.
- Roughly results in a 45% blend with model 1.

![Gradient](images/gradient.png?raw=true "Gradient")

## Merge Strategies

The following merge strategies are available:

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

**Note**: MonsterMapper works exactly the same.

1. **Prepare your configuration file (or modify the included one)**: Create a YAML file with your model paths, device settings, and other configurations.
2. **Run the script**: Use the following command to start the merging process:

    ```bash
    python merge_monster.py --config your_config_file.yaml
    ```
If no --config argument is given, it will instead revert to loading default.yaml.

3. **Evaluate and save**: The script will automatically evaluate, log, and save the best-performing merged model. A copy of the log will be saved in a subfolder called "logs".
