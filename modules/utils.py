import os
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class PrintAndStoreLogger:
    def __init__(self, original_stdout):
        self.contents = ''
        self.original_stdout = original_stdout

    def write(self, text):
        self.contents += text
        self.original_stdout.write(text)  # Print to the console as well

    def flush(self):
        pass  # This might be needed depending on the environment

def print_ascii_art(file_path):
    try:
        with open(file_path, 'r') as file:
            ascii_art = file.read()
            print(ascii_art)
    except FileNotFoundError:
        print("ASCII art file not found.")

def format_context(context, length=30):
    return (context[:length-2] + '..') if len(context) > length else context.ljust(length)