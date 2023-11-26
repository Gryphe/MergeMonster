import os
import shutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True

def load_model(model_path, device):
    with NoInit():
        print("------------------------------------")
        print(f"{datetime.now().strftime('%H:%M:%S')} - Loading model ({model_path})...")
        tf_model = AutoModelForCausalLM.from_pretrained(model_path)
        tf_model.half()
        tf_model = tf_model.to(device)
        tf_model.eval()
        print(f"{datetime.now().strftime('%H:%M:%S')} -  Model loaded. Dtype: {tf_model.dtype}")
        print("------------------------------------")

    return tf_model

def save_model(output_directory, source_directory, model):
    print(f"{datetime.now().strftime('%H:%M:%S')} - Saving model to {output_directory}...")
    model.save_pretrained(output_directory)

    # Check if additional files need to be copied
    if output_directory != source_directory:
        print(f"{datetime.now().strftime('%H:%M:%S')} - Copying tokenizer files to {output_directory}...")
        files_to_copy = ["added_tokens.json", "tokenizer.model", "special_tokens_map.json", 
                         "tokenizer_config.json", "vocab.json", "merges.txt"]
        
        for filename in files_to_copy:
            src_path = os.path.join(source_directory, filename)
            dst_path = os.path.join(output_directory, filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {filename}")
            else:
                print(f"Skipped {filename} (not found)")

    print(f"{datetime.now().strftime('%H:%M:%S')} - Model and tokenizer files saved successfully.")