from experiment import train_finetune, train_cross_lemmas, neighbors_test
import json
def run_main():
    train_finetune(21,21, 10, 20)

if __name__ == "__main__":
    train_finetune(21,21, 10, 2000)
