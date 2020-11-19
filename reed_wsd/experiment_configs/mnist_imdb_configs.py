import json
import sys

tasks = ["mnist"]
architectures = ["simple", "abstaining"]
alpha = 0.7
criterions = ["nll", "conf1", "pairwise"]
warmup_epochs = 3
confuse = "all"
styles = ["single", "pairwise"]
confidences = ["max_prob", "max_non_abs", "inv_abs", "abs"]
dev_corpus = None
bsz = 64
n_epochs = 100
trustscore = False

architecture_to_confs = {"simple": ["max_prob"],
                         "abstaining": ["max_non_abs", "inv_abs", "abs"]}
conf_to_criterion = {"max_prob": "nll",
                     "max_non_abs": "conf1",
                     "inv_abs": "conf1",
                     "abs": "pairwise"}
def criterion_to_style(criterion):
    if criterion == "pairwise":
        return "pairwise"
    else:
        return "single"

def configs(task):
    l = []
    for arch in architectures:
        confs = architecture_to_confs[arch]
        for conf in confs:
            criterion = conf_to_criterion[conf]
            config = {}
            config["task"] = task
            config["architecture"] = arch
            config["confidence"] = conf
            config["criterion"] = {"name": criterion,
                                   "alpha": alpha,
                                   "warmup_epochs": warmup_epochs}
            config["confuse"] = confuse
            config["style"] = criterion_to_style(criterion)
            config["dev_corpus"] = dev_corpus
            config["bsz"] = bsz
            config["n_epochs"] = n_epochs
            config["trustscore"] = trustscore
            l.append(config)
    return l
                
if __name__ == "__main__":
    task = sys.argv[1]
    configs_list = configs(task)
    with open("{}_configs.json".format(task), "w") as f:
        json.dump(configs_list, f)
