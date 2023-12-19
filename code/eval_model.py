import torch
from torch_geometric.data import DataLoader
import time
import json
import numpy as np
from collections import OrderedDict
from wordsegment import load, segment

import conf
import data
import demod_data_collector
import gen_kw
import gen_token
import gen_token_vec
import graph2seq as g2s
import graph2multi_label as g2ml
from gen_token_vec import callback


MODEL_SAVE_PATH = "./model_save/model_parm.pt2023-11-22T21:01:45.671180"


def eval_model():
    _, _, test_data_loader, test_ml_dict = g2ml.load_data()
    encoder_model = g2s.Encoder(conf)
    decoder_model = g2ml.Decoder(conf)
    pred_model = g2ml.Graph2ML(encoder_model, decoder_model).to(g2s.device)
    pred_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    s_time = time.time()
    _, _, _, precision2, recall2, f1_score2 = g2ml.evaluate(pred_model, test_data_loader, test_ml_dict, "test")
    e_time = time.time()
    print("train precision: {0}, recall: {1}, f1_score: {2}".format(precision2, recall2, f1_score2))
    n_len = conf.other_batch_size * (len(test_ml_dict) // conf.other_batch_size)
    print(e_time - s_time, n_len, (e_time - s_time) / n_len)


if __name__ == "__main__":
    g2s.set_seed(conf.seed)
    eval_model()

