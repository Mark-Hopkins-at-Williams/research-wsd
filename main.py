from experiment import train_finetune
import json
if __name__ == "__main__":
    result = train_finetune(20,1000000000,10,1000)
    with open("finetune_result.txt", "w") as f:
        json.dump(result, f)

