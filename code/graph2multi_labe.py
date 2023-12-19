import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geometric_nn
from torch_geometric.data import DataLoader
import time
import datetime
import os

import conf
import data
import gen_token_vec
from gen_token_vec import callback
import demod_data_collector
import graph2seq
import data_handler


class Decoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.in_dim = conf.rnn_hidden
        self.hidden_dim = conf.mlp_hidden
        self.out_dim = conf.vocab_size
        self.mlp = geometric_nn.MLP([self.in_dim, self.hidden_dim, self.out_dim], dropout=conf.dropout)

    def forward(self, x):
        pred = self.mlp(x)
        return pred


class Graph2ML(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        print(self.encoder)
        self.decoder = decoder
        print(self.decoder)

    def forward(self, data, mode):
        node_embed, graph_embed = self.encoder(data.x, data.x_api_feat, data.x_root_feat, data.x_root_id, data.edge_index, data.bw_edge_index, data.batch)
        if mode == "train" or mode == "validation":
            output = self.decoder(graph_embed)
        elif mode == "test":
            output = self.decoder(graph_embed)
        return output


def load_data():
    _, feat_token_embed_dict = gen_token_vec.gen_token_embedding()
    graph_data_list, des_label_list = demod_data_collector.read_data(
        demod_data_collector.GRAPH_DATA_SAVE_PATH, demod_data_collector.DES_LABEL_SAVE_PATH)
    train_graph_data, train_des_label, validation_graph_data, validation_des_label, test_graph_data, test_des_label = demod_data_collector.split_data(graph_data_list, des_label_list)
    print("train data num: {0}, validation data num: {1}, test data num: {2}".format(len(train_des_label),
                                                                                     len(validation_des_label),
                                                                                     len(test_des_label)))

    train_data, _ = data.gen_torch_ml_data(train_graph_data, train_des_label, feat_token_embed_dict)
    validation_data, _ = data.gen_torch_ml_data(validation_graph_data, validation_des_label, feat_token_embed_dict)
    test_data, test_ml_dict = data.gen_torch_ml_data(test_graph_data, test_des_label, feat_token_embed_dict)

    train_data_loader = DataLoader(train_data[: conf.train_batch_size * (len(train_data) // conf.train_batch_size)], batch_size=conf.train_batch_size)
    validation_data_loader = DataLoader(validation_data[: conf.other_batch_size * (len(validation_data) // conf.other_batch_size)], batch_size=conf.other_batch_size)
    test_data_loader = DataLoader(test_data[: conf.other_batch_size * (len(test_data) // conf.other_batch_size)], batch_size=conf.other_batch_size)
    print("train data num: {0}, validation data num: {1}, test data num: {2}".format(len(train_data[: conf.train_batch_size * (len(train_data) // conf.train_batch_size)]),
                                                                                     len(validation_data[: conf.other_batch_size * (len(validation_data) // conf.other_batch_size)]),
                                                                                     len(test_data[: conf.other_batch_size * (len(test_data) // conf.other_batch_size)])))
    return train_data_loader, validation_data_loader, test_data_loader, test_ml_dict


def train(model, dataloader, optimizer, criterion):
    model.train()
    sum_loss = 0
    for data in dataloader:
        data = data.to(graph2seq.device)
        optimizer.zero_grad()
        pred = model(data=data, mode="train").to(graph2seq.device)
        data.y = torch.reshape(data.y, (-1, conf.vocab_size)).to(graph2seq.device)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss / len(dataloader)


def cal_acc(pred, y_seq, x_id, test_ml_dict, mode="validation"):
    seq_num = pred.shape[0]
    pred_seq = []
    label_token_idx_dict = gen_token_vec.load_label_token_dict()
    rlabel_token_idx_dict = dict(zip(label_token_idx_dict.values(), label_token_idx_dict.keys()))
    x_id = x_id.tolist()

    for seq_idx in range(0, seq_num):
        top5_pred = torch.topk(pred[seq_idx], 5)[1].tolist()
        top5_pred_list = [rlabel_token_idx_dict[tid] for tid in top5_pred]
        if mode == "test":
            for token in test_ml_dict[x_id[seq_idx]][0]:
                if token in top5_pred_list:
                    pred_idx = top5_pred[top5_pred_list.index(token)]
                    pred[seq_idx][pred_idx] += conf.thr
        cur_pred_seq = [pred_idx for pred_idx in top5_pred if pred[seq_idx][pred_idx] > conf.thr]
        if len(cur_pred_seq) == 0:
            cur_pred_seq = top5_pred[: 3]
        pred_seq.append(cur_pred_seq)

    n_cor = 0
    len_pred = 0
    len_lab = 0
    n_src = 0
    sum_prec = 0
    sum_rec = 0

    for cur_pred, cur_y_seq, cur_x_id in zip(pred_seq, y_seq, x_id):
        n_cor += len(set(cur_pred) & set(cur_y_seq))
        len_pred += len(cur_pred)
        len_lab += len(cur_y_seq)
        n_src += 1
        sum_prec += len(set(cur_pred) & set(cur_y_seq)) / len(cur_pred)
        sum_rec += len(set(cur_pred) & set(cur_y_seq)) / len(cur_y_seq)
    return n_cor, len_pred, len_lab, n_src, sum_prec, sum_rec


def evaluate(model, dataloader, test_ml_dict, mode="validation"):
    model.eval()
    ttl_cor = 0
    ttl_len_pred = 0
    ttl_len_lab = 0
    ttl_src = 0
    ttl_prec = 0
    ttl_rec = 0
    for data in dataloader:
        data = data.to(graph2seq.device)
        pred = model(data=data, mode="test").to(graph2seq.device)
        pred = torch.sigmoid(pred)
        n_cor, len_pred, len_lab, n_src, sum_prec, sum_rec = cal_acc(pred, data.y_seq, data.x_id, test_ml_dict, mode)
        ttl_cor += n_cor
        ttl_len_pred += len_pred
        ttl_len_lab += len_lab
        ttl_src += n_src
        ttl_prec += sum_prec
        ttl_rec += sum_rec

    precision1 = ttl_cor / ttl_len_pred
    recall1 = ttl_cor / ttl_len_lab
    f1_score1 = 2 * precision1 * recall1 / (precision1 + recall1)
    precision2 = ttl_prec / ttl_src
    recall2 = ttl_rec / ttl_src
    f1_score2 = 2 * precision2 * recall2 / (precision2 + recall2)
    return precision1, recall1, f1_score1, precision2, recall2, f1_score2


if __name__ == "__main__":
    graph2seq.set_seed(conf.seed)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_data_loader, validation_data_loader, test_data_loader, test_ml_dict = load_data()

    encoder_model = graph2seq.Encoder(conf)
    decoder_model = Decoder(conf)
    graph2ml = Graph2ML(encoder_model, decoder_model).to(graph2seq.device)

    optimizer = optim.Adam(graph2ml.parameters(), lr=conf.lr)
    for epoch_idx in range(conf.epochs):
        loss = train(graph2ml, train_data_loader, optimizer, criterion)
        time_str = datetime.datetime.now().isoformat()
        print("epoch id: {0}, train loss: {1}, time: {2}".format(epoch_idx, loss, time_str))
        s_time = time.time()
        precision1, recall1, f1_score1, precision2, recall2, f1_score2 = evaluate(graph2ml, test_data_loader, test_ml_dict, "test")
        e_time = time.time()
        print("[ori] test precision: {0}, recall: {1}, f1_score: {2}".format(precision1, recall1, f1_score1))
        print("train precision: {0}, recall: {1}, f1_score: {2}".format(precision2, recall2, f1_score2))
        n_len = 2112
        print(e_time - s_time, n_len, (e_time - s_time) / n_len, "\n")

