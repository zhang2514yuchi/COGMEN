import pickle
import os
import argparse
import torch
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from tqdm import tqdm

import cogmen

log = cogmen.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def main(args):
    data = load_pkl(f"data/{args.dataset}/data_{args.dataset}.pkl")#改成自己的

    model_dict = torch.load(
        "model_checkpoints/"
        + str(args.dataset)
        + "_best_dev_f1_model_"
        + str(args.modalities)
        + ".pt",
    )
    stored_args = model_dict["args"]
    model = model_dict["state_dict"]
    testset = cogmen.Dataset(data, stored_args)

    test = True
    with torch.no_grad():
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            # golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            y_hat = model(data)
            print(y_hat)            #显示结果
            # preds.append(y_hat.detach().to("cpu"))

        
        # golds = torch.cat(golds, dim=-1).numpy()
        # preds = torch.cat(preds, dim=-1).numpy()
        # f1 = metrics.f1_score(golds, preds, average="weighted")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eval0.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="at",
        # required=True,
        choices=["a", "at", "atv"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    args = parser.parse_args()
    main(args)
