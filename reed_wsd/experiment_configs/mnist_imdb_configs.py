import json

tasks = ["mnist", "imdb"]
architectures = ["simple", "abstaining"]
alpha = 0.5
criterions = ["nll", "conf1", "pairwise"]
warmup_epochs = 5
confuse = "all"
styles = ["single", "pairwise"]
confidences = ["max_prob", "max_non_abs", "inv_abs", "abs"]
dev_corpus = None
bsz = 64
n_epochs = 40
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

def configs():
    l = []
    for task in tasks:
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
    configs_list = configs()
    with open("mnist_imdb_configs.json", "w") as f:
        json.dump(configs_list, f)
