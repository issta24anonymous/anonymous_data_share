import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geometric_nn
from torch_geometric.data import DataLoader
import copy
import datetime
import random
import time
import numpy as np

import conf
import data
import gen_token_vec
from gen_token_vec import callback
import demod_data_collector
import feature_fusion
import data_handler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.input_dim = conf.x_dim
        self.gnn_layer_num = conf.gnn_layers
        self.gnn_hidden_dim = conf.gnn_hidden
        if conf.gnn_type == "ggnn":
            self.ggnn_model = geometric_nn.GatedGraphConv(out_channels=conf.rnn_hidden, num_layers=conf.gnn_layers)
        elif conf.gnn_type == "gcn":
            self.gcn_model = geometric_nn.GCN(in_channels=self.input_dim, hidden_channels=self.gnn_hidden_dim, num_layers=1)
            self.fw_gnn_conv = nn.ModuleList([copy.deepcopy(self.gcn_model) for _ in range(self.gnn_layer_num)])
        elif conf.gnn_type == "graphsage":
            self.fw_graphsage_model = geometric_nn.GraphSAGE(in_channels=self.input_dim, hidden_channels=self.gnn_hidden_dim, num_layers=self.gnn_layer_num)
            self.bw_graphsage_model = geometric_nn.GraphSAGE(in_channels=self.input_dim, hidden_channels=self.gnn_hidden_dim, num_layers=self.gnn_layer_num)
        else:
            print("model type error.")

    def forward(self, x, api_feat, root_feat, root_id, fw_edge_idx, bw_edge_idx, batch):
        if conf.gnn_type == "ggnn":
            x = self.ggnn_model(x, fw_edge_idx)
        elif conf.gnn_type == "gcn":
            for fw_conv in self.fw_gnn_conv:
                fw_x = fw_conv(x, fw_edge_idx).relu()
            x = fw_x
        elif conf.gnn_type == "graphsage":
            fw_x = self.fw_graphsage_model(x, fw_edge_idx)
            bw_x = self.bw_graphsage_model(x, bw_edge_idx)
            x = torch.cat((fw_x, bw_x), dim=1)
        # graph_embed = geometric_nn.global_add_pool(x, batch)
        # graph_embed = geometric_nn.global_mean_pool(x, batch)
        graph_embed = geometric_nn.global_max_pool(x, batch)
        return x, graph_embed


class Decoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.rnn_layer_num = conf.rnn_layers
        self.rnn_hidden_dim = conf.rnn_hidden
        self.dec_output_dim = conf.vocab_size
        if conf.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.dec_output_dim, hidden_size=self.rnn_hidden_dim, num_layers=self.rnn_layer_num, batch_first=False)
        elif conf.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.dec_output_dim, hidden_size=self.rnn_hidden_dim, num_layers=self.rnn_layer_num, batch_first=False)
        print(self.rnn)
        self.linear = nn.Linear(self.rnn_hidden_dim, self.dec_output_dim)
        self.dropout = nn.Dropout(conf.dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def cal_lstm_ot(self):
        w_io = self.rnn.weight_ih_l0[self.rnn_hidden_dim * 3:]
        w_ho = self.rnn.weight_hh_l0[self.rnn_hidden_dim * 3:]
        b_io = self.rnn.bias_ih_l0[self.rnn_hidden_dim * 3:]
        b_ho = self.rnn.bias_hh_l0[self.rnn_hidden_dim * 3:]
        x = b_io + b_ho
        ot = torch.sigmoid(x)
        return ot

    def forward(self, y, hidden, cell=None):
        # [seq_len: 1, batch_size, output_size]
        y = y.unsqueeze(0)
        if conf.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(y, (hidden, cell))
        elif conf.rnn_type == "gru":
            output, hidden = self.rnn(y, hidden)
        line_out = self.linear(output)
        line_out = self.dropout(line_out)
        prediction = self.softmax(line_out)
        if conf.rnn_type == "lstm":
            return prediction, hidden, cell
        elif conf.rnn_type == "gru":
            return prediction, hidden


class Attention(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.rnn_hidden_dim = conf.rnn_hidden
        self.weight = nn.Parameter(torch.Tensor(self.rnn_hidden_dim, self.rnn_hidden_dim))
        self.linear = nn.Linear(self.rnn_hidden_dim + self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.softmax = nn.Softmax(dim=2)
        self.activation = nn.ReLU()
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, node_embed, hidden):
        attn_score = hidden.matmul(self.weight).matmul(node_embed.t())
        attn_weight = self.softmax(attn_score)
        attn_node = attn_weight.matmul(node_embed)
        hidden = self.linear(torch.cat((hidden, attn_node), dim=2))
        return hidden


class Graph2Seq(nn.Module):
    def __init__(self, encoder, decoder, conf, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn = Attention(conf)

    def forward(self, data, y_input, mode):
        if mode == "train":
            batch_size = conf.train_batch_size
        elif mode == "validation":
            batch_size = conf.other_batch_size
        elif mode == "test":
            batch_size = conf.other_batch_size

        seq_len = conf.max_seq_len + 1
        outputs = torch.zeros(y_input.shape)
        node_embed, hidden = self.encoder(data.x, data.x_api_feat, data.x_root_feat, data.x_root_id, data.edge_index, data.bw_edge_index, data.batch)
        # [seq_len: 1, batch_size, hidden_size]
        hidden = hidden.unsqueeze(0)
        hidden = self.attn(node_embed, hidden)
        if conf.rnn_type == "lstm":
            cell = hidden
            hidden = torch.mul(torch.tanh(cell), self.decoder.cal_lstm_ot())
        if mode == "train" or mode == "validation":
            for i in range(seq_len):
                if i == 0:
                    decoder_input = y_input[i * batch_size: (i + 1) * batch_size].to(device)
                else:
                    if random.random() < conf.teacher_forcing_ratio:
                        decoder_input = y_input[i * batch_size: (i + 1) * batch_size].to(device)
                    else:
                        decoder_input = outputs[(i - 1) * batch_size: i * batch_size].to(device)
                if conf.rnn_type == "lstm":
                    output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                elif conf.rnn_type == "gru":
                    output, hidden = self.decoder(decoder_input, hidden)
                outputs[i * batch_size: (i + 1) * batch_size] = output
        elif mode == "test":
            for i in range(seq_len):
                if i == 0:
                    decoder_input = y_input[i * batch_size: (i + 1) * batch_size].to(device)
                else:
                    decoder_input = outputs[(i - 1) * batch_size: i * batch_size].to(device)
                if conf.rnn_type == "lstm":
                    output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                elif conf.rnn_type == "gru":
                    output, hidden = self.decoder(decoder_input, hidden)
                outputs[i * batch_size: (i + 1) * batch_size] = output
        return outputs


def load_data():
    _, feat_token_embed_dict = gen_token_vec.gen_token_embedding()
    graph_data_list, des_label_list = demod_data_collector.read_data(
        demod_data_collector.GRAPH_DATA_SAVE_PATH, demod_data_collector.DES_LABEL_SAVE_PATH)
    train_graph_data, train_des_label, validation_graph_data, validation_des_label, test_graph_data, test_des_label = demod_data_collector.split_data(graph_data_list, des_label_list)
    print("train data num: {0}, validation data num: {1}, test data num: {2}".format(len(train_des_label),
                                                                                     len(validation_des_label),
                                                                                     len(test_des_label)))

    train_data = data.gen_torch_pyg_data(train_graph_data, train_des_label, feat_token_embed_dict)
    validation_data = data.gen_torch_pyg_data(validation_graph_data, validation_des_label, feat_token_embed_dict)
    test_data = data.gen_torch_pyg_data(test_graph_data, test_des_label, feat_token_embed_dict)
    train_data_loader = DataLoader(train_data[: conf.train_batch_size * (len(train_data) // conf.train_batch_size)], batch_size=conf.train_batch_size)
    validation_data_loader = DataLoader(validation_data[: conf.other_batch_size * (len(validation_data) // conf.other_batch_size)], batch_size=conf.other_batch_size)
    test_data_loader = DataLoader(test_data[: conf.other_batch_size * (len(test_data) // conf.other_batch_size)], batch_size=conf.other_batch_size)
    print("train data num: {0}, validation data num: {1}, test data num: {2}".format(len(train_data[: conf.train_batch_size * (len(train_data) // conf.train_batch_size)]),
                                                                                     len(validation_data[: conf.other_batch_size * (len(validation_data) // conf.other_batch_size)]),
                                                                                     len(test_data[: conf.other_batch_size * (len(test_data) // conf.other_batch_size)])))
    return train_data_loader, validation_data_loader, test_data_loader


def batch_label(data_y, seq_len):
    idx = 0
    label = torch.zeros((data_y.shape), dtype=torch.float32)
    for pos in range(0, seq_len):
       for offset in range(0, data_y.shape[0], seq_len):
           label[idx] = data_y[offset + pos]
           idx += 1
    return label


def batch_loss(loss_weights):
    seq_num = len(loss_weights)
    seq_len = len(loss_weights[0])
    mask_seq = []
    for pos in range(0, seq_len):
        for seq_idx in range(0, seq_num):
            mask_seq.append(loss_weights[seq_idx][pos])
    return mask_seq


def mark_pad_token(pred, mask_seq):
    pad_token_vec = torch.zeros((1, pred.shape[1]), dtype=torch.int)
    pad_token_vec[0][0] = 1
    for idx, mask in enumerate(mask_seq):
        if mask == 0:
            pred[idx] = pad_token_vec
    return pred


def cal_acc(pred, y_seq, mode="validation"):
    seq_len = conf.max_seq_len + 1
    seq_num = int(pred.shape[0] / seq_len)
    pred_seq = []

    for seq_idx in range(0, seq_num):
        cur_pred_seq = []
        for pos_idx in range(0, seq_len):
            token_idx = torch.argmax(pred[seq_idx + pos_idx * seq_num]).item()
            if token_idx == conf.vocab_size - 1 or token_idx == 0:
                break
            cur_pred_seq.append(token_idx)
        pred_seq.append(cur_pred_seq)

    n_cor = 0
    len_pred = 0
    len_lab = 0
    n_src = 0
    sum_prec = 0
    sum_rec = 0

    label_token_idx_dict = gen_token_vec.load_label_token_dict()
    rlabel_token_idx_dict = dict(zip(label_token_idx_dict.values(), label_token_idx_dict.keys()))

    for cur_pred, cur_y_seq in zip(pred_seq, y_seq):
        cur_pred = sorted(set(cur_pred), key=cur_pred.index)
        n_cor += len(set(cur_pred) & set(cur_y_seq))
        len_pred += len(cur_pred)
        len_lab += len(cur_y_seq)
        n_src += 1
        if len(cur_pred) == 0:
            sum_prec += 0
        else:
            sum_prec += len(set(cur_pred) & set(cur_y_seq)) / len(cur_pred)
        sum_rec += len(set(cur_pred) & set(cur_y_seq)) / len(cur_y_seq)

        if mode == "test":
            pred_list = []
            y_seq_list = []
            for tid in cur_pred:
                pred_list.append(rlabel_token_idx_dict[tid])
            for tid in cur_y_seq:
                y_seq_list.append(rlabel_token_idx_dict[tid])

    return n_cor, len_pred, len_lab, n_src, sum_prec, sum_rec


def glo_root_id(root_id):
    root_id = root_id.tolist()
    g_root_id = []
    for idx, rid in enumerate(root_id):
        rid = idx * gen_token_vec.MAX_GRAPH_NODE_NUM + rid
        g_root_id.append(int(rid))
    g_root_id = torch.tensor(g_root_id, dtype=torch.long).to(device)
    return g_root_id


def train(model, dataloader, optimizer, criterion):
    model.train()
    sum_loss = 0
    for data in dataloader:
        data = data.to(device)
        data.x_root_id = glo_root_id(data.x_root_id)
        y_input = batch_label(data.y_input, conf.max_seq_len + 1).to(device)
        y_target = batch_label(data.y_target, conf.max_seq_len + 1).to(device)
        mask_seq = batch_loss(data.y_loss_weights)
        optimizer.zero_grad()
        pred = model(data=data, y_input=y_input, mode="train").to(device)
        loss = criterion(pred, y_target)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss / len(dataloader)


def evaluate(model, dataloader, mode="validation"):
    model.eval()
    ttl_cor = 0
    ttl_len_pred = 0
    ttl_len_lab = 0
    ttl_src = 0
    ttl_prec = 0
    ttl_rec = 0

    for data in dataloader:
        data = data.to(device)
        y_input = batch_label(data.y_input, conf.max_seq_len + 1).to(device)
        if mode == "train":
            pred = model(data=data, y_input=y_input, mode="train")
        elif mode == "validation":
            pred = model(data=data, y_input=y_input, mode="validation")
        elif mode == "test":
            pred = model(data=data, y_input=y_input, mode="test")
        n_cor, len_pred, len_lab, n_src, sum_prec, sum_rec = cal_acc(pred, data.y_seq, mode)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed(conf.seed)

    criterion = nn.CrossEntropyLoss()
    train_data_loader, validation_data_loader, test_data_loader = load_data()

    encoder_model = Encoder(conf)
    decoder_model = Decoder(conf)
    graph2seq = Graph2Seq(encoder_model, decoder_model, conf, device).to(device)
    optimizer = optim.Adam(graph2seq.parameters(), lr=conf.lr)

    for epoch_idx in range(conf.epochs):
        loss = train(graph2seq, train_data_loader, optimizer, criterion)
        time_str = datetime.datetime.now().isoformat()
        print("epoch id: {0}, train loss: {1}, time: {2}".format(epoch_idx, loss, time_str))
        s_time = time.time()
        precision1, recall1, f1_score1, precision2, recall2, f1_score2 = evaluate(graph2seq, test_data_loader, "test")
        e_time = time.time()
        print("[ori] test precision: {0}, recall: {1}, f1_score: {2}".format(precision1, recall1, f1_score1))
        print("train precision: {0}, recall: {1}, f1_score: {2}".format(precision2, recall2, f1_score2))
        n_len = 2112
        print(e_time - s_time, n_len, (e_time - s_time) / n_len, "\n")
