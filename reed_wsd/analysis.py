import json

def read_valid_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data

class Prediction:
    def __init__(self, pred):
        self.pred = pred
        self.convertors = dict()

    def __getitem__(self, key):
        return self.convertors[key](self.pred[key])

class Predictions:

    def __init__(self, data, pred_cls):
        self.predictions = _init_predictions(data[0][1][0]) #[experiment#][the predictions][# of rep]

    def _init_predictions(self, preds):
        return [pred_cls(pred) for pred in preds]

    def __getitem__(self, i):
        return self.predictions[i]
