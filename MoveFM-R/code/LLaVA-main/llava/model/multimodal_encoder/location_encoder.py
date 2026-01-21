import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.weights[i](x)
            x = x0 * xw + self.biases[i] + x
        return x

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class DCN(nn.Module):
    def __init__(self, config, num_cross_layers):
        super().__init__()
        self.cross_network = CrossNetwork(config.n_embd, num_cross_layers)
        self.deep_network = DeepNetwork(config.n_embd, 4 * config.n_embd, config.n_embd)
        self.combined_output = nn.Linear(config.n_embd * 2, config.n_embd)

    def forward(self, x):
        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        output = self.combined_output(combined)
        return output

#todo Change to the sum of codebook indices
class Vocab_emb(nn.Module):
    def __init__(self, config, num_cross_layers=2):
        super().__init__()
        self.lon_lat_embedding = nn.Linear(2, config.n_embd // 2)
        self.poi_feature_embedding = nn.Linear(34, config.n_embd // 4)
        self.flow_rank_embedding = nn.Embedding(9, config.n_embd // 4)
        self.dcn = DCN(config, num_cross_layers)

    def forward(self, vocab):
        vocab_poi = vocab[:, :34]
        vocab_lon_lat = vocab[:, 68:70]
        vocab_rank = vocab[:, -1].to(torch.long)
        
        vocab_poi_embedding = self.poi_feature_embedding(vocab_poi)  # Shape: [batch_size, n_embd // 4]
        vocab_lon_lat_emb = self.lon_lat_embedding(vocab_lon_lat)  # Shape: [batch_size, n_embd // 2]
        vocab_rank_emb = self.flow_rank_embedding(vocab_rank)  # Shape: [batch_size, n_embd // 4]
        
        vocab_embedding0 = torch.cat((vocab_lon_lat_emb, vocab_rank_emb,vocab_poi_embedding), dim=-1) 
        vocab_embedding = self.dcn(vocab_embedding0)
        return vocab_embedding


# text_emb-final
class DCN(nn.Module):
    def __init__(self, config, num_cross_layers):
        super().__init__()
        self.cross_network = CrossNetwork(int(config.n_embd*1.5), num_cross_layers)
        self.deep_network = DeepNetwork(int(config.n_embd*1.5), 4 * config.n_embd, config.n_embd)
        self.combined_output = nn.Linear(int(config.n_embd*2.5), config.n_embd)

    def forward(self, x):
        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        output = self.combined_output(combined)
        return output

#todo Change to the sum of codebook indices
class Vocab_emb(nn.Module):
    def __init__(self, config, num_cross_layers=2,vocab_size=512):
        super().__init__()
        self.lon_lat_embedding = nn.Linear(2, config.n_embd // 2)
        self.poi_feature_embedding = nn.Linear(34, config.n_embd // 4)
        self.flow_rank_embedding = nn.Embedding(9, config.n_embd // 4)
        self.dcn = DCN(config, num_cross_layers)


        # Add text codebook embedding layers
        self.text_embedding_1 = nn.Embedding(vocab_size, config.n_embd // 2)
        self.text_embedding_2 = nn.Embedding(vocab_size, config.n_embd // 2)
        self.text_embedding_3 = nn.Embedding(vocab_size, config.n_embd // 2)
        self.text_embedding_4 = nn.Embedding(vocab_size, config.n_embd // 2)
        

    def forward(self, vocab):
        vocab_poi = vocab[:, :34]
        vocab_lon_lat = vocab[:, 68:70]
        # vocab_rank = vocab[:, -1].to(torch.long)

        vocab_rank = vocab[:, 70].to(torch.long)

        vocab_text1 = vocab[:, 71].to(torch.long)
        vocab_text2 = vocab[:, 72].to(torch.long)
        vocab_text3 = vocab[:, 73].to(torch.long)
        vocab_text4 = vocab[:, 74].to(torch.long)

        # print(vocab[:, 71:74])
        
        vocab_poi_embedding = self.poi_feature_embedding(vocab_poi)  # Shape: [batch_size, n_embd // 4]
        vocab_lon_lat_emb = self.lon_lat_embedding(vocab_lon_lat)  # Shape: [batch_size, n_embd // 2]
        vocab_rank_emb = self.flow_rank_embedding(vocab_rank)  # Shape: [batch_size, n_embd // 4]

        text_emb1 = self.text_embedding_1(vocab_text1)  # Shape: [batch_size, n_embd // 2]
        text_emb2 = self.text_embedding_2(vocab_text2)  # Shape: [batch_size, n_embd // 2]
        text_emb3 = self.text_embedding_3(vocab_text3)  # Shape: [batch_size, n_embd // 2]
        text_emb4 = self.text_embedding_4(vocab_text4)  # Shape: [batch_size, n_embd // 2]

        # test_emb = (text_emb1 + text_emb2 + text_emb3 )  # Shape: [batch_size, n_embd // 2]
        test_emb = (text_emb1 + text_emb2 + text_emb3 + text_emb4)  # Shape: [batch_size, n_embd // 2]

        vocab_embedding0 = torch.cat((vocab_lon_lat_emb, vocab_rank_emb,vocab_poi_embedding,test_emb), dim=-1) 
        vocab_embedding = self.dcn(vocab_embedding0)
        return vocab_embedding
    
