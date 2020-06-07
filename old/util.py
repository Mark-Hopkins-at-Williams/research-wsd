import torch
import os
import json

class Logger:
    def __init__(self, verbose):
        self.verbose = verbose
    
    def __call__(self, msg):
        if self.verbose:
            print(msg)


def cudaify(x):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            cuda = torch.device('cuda:2')
        else:
            cuda = torch.device('cuda:0')
        return x.cuda(cuda)
    else: 
        return x


def generate_sense_to_pofs_dict():
    with open("data/googledata.json", "r") as f:
        data = json.load(f)
    d = {}
    for doc in data:
        print("processing", doc["docname"], "...")
        for word in doc["doc"]:
            if "lemma" in word.keys() and "sense" in word.keys() and "pos" in word.keys():
                d[word["sense"]] = word["pos"]
    with open("data/sense_to_pofs_dict.json", "w") as f:
        json.dump(d, f, indent=4)

class Cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.join(os.getcwd(), newPath)

    def __enter__(self):
        if not os.path.exists(self.newPath):
            os.mkdir(self.newPath)
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        
def delete_last_line(file):
    with open(file, "r+", encoding = "utf-8") as file:

        # Move the pointer (similar to a cursor in a text editor) to the end of the file
        file.seek(0, os.SEEK_END)

        # This code means the following code skips the very last character in the file -
        # i.e. in the case the last line is null we delete the last line
        # and the penultimate one
        pos = file.tell() - 1

        # Read each character in the file one at a time from the penultimate
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)

        # So long as we're not at the start of the file, delete all the characters ahead
        # of this position
        if pos > 0:
            file.seek(pos, os.SEEK_SET)
            file.truncate()
