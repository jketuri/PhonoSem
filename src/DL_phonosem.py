#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
   Introduction to Deep Learning (LDA-T3114)
   Code structure bases to the homework exercise tutorials
   Dev and test data are set empty during final run, therefore dev acc is always 0%
"""


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[1]:


from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data import CONTEXT_SIZE, get_subtokens, read_datasets
from paths import data_dir


# In[2]:


torch.set_num_threads(10)


# In[3]:


#--- hyperparameters ---
N_EPOCHS = 20
LEARNING_RATE = 0.1
REPORT_EVERY = 1
VERBOSE = False
EMBEDDING_DIM = 300
HIDDEN_DIM = 500
N_LAYERS = 3
BATCH_SIZE = 1000


# In[4]:


#--- model ---
class LanguageModel(nn.Module):
    def __init__(
            self, 
            embedding_dim,
            context_size,
            hidden_dim, 
            n_layers,
            vocab,
            tokenmap):
        super(LanguageModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.tokenmap = tokenmap
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.hidden = []
        for n_layer in range(1, n_layers):
            self.hidden.append(nn.Linear(embedding_dim if n_layer == 1 else hidden_dim, hidden_dim))
        self.linear = nn.Linear(embedding_dim if n_layers == 1 else hidden_dim, len(vocab))

    def add_all(self, embeds):
        result = torch.zeros(self.embedding_dim)
        for embed in embeds:
            result = result.add(embed)
        return result.tolist()
    
    def embed_tokens(self, an_input):
        embeddings = []
        for token_index in an_input[0]:
            token = self.vocab[token_index]
            if token.startswith('#') and token.endswith('#'):
                subtokens = get_subtokens(token.strip('#'))
            else:
                subtokens = [token]
            embedding = torch.zeros(self.embedding_dim)
            for subtoken in subtokens:
                embedding = embedding.add(self.embed(torch.tensor(self.tokenmap[subtoken])))
            embeddings.append(embedding.tolist())
        result = torch.tensor([embeddings])
        return result

    def forward(self, inputs):
        outputs = torch.cat([self.embed_tokens(an_input) for an_input in inputs], dim=0).view(
            -1, self.embedding_dim)
        outputs = torch.tensor([self.add_all(outputs.narrow(0, index, self.context_size - 1))
                                for index in range(0, len(outputs), self.context_size - 1)])
        for hidden in self.hidden:
            outputs = F.relu(hidden(outputs))
        return F.log_softmax(self.linear(outputs), dim=1)


# In[5]:


def evaluate(loader, model):
    correct = 0
    count = 0
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        count += len(source)
        log_probs = model(source)
        _, predicted = torch.max(log_probs, 1)
        correct += torch.eq(predicted, target).sum().item()
    return (correct * 100.0 / count) if count else 0


# In[6]:


#--- initialization ---
data, tokens, vocab, tokenmap = read_datasets('pg', data_dir, embeds_only=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def make_model():
    return LanguageModel(
        EMBEDDING_DIM,
        CONTEXT_SIZE,
        HIDDEN_DIM,
        N_LAYERS,
        vocab,
        tokenmap)

model = make_model()

def make_optimizer(model):
    return optim.Adam(model.parameters(), lr=LEARNING_RATE)

optimizer = make_optimizer(model)


# In[9]:


# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=data['training'], batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=data['test'], batch_size=BATCH_SIZE, shuffle=False)
dev_loader = torch.utils.data.DataLoader(
    dataset=data['dev'], batch_size=BATCH_SIZE, shuffle=False)

loss_function = nn.NLLLoss()


# In[10]:


#--- training ---
for epoch in range(N_EPOCHS):
    total_loss = 0
    for source, target in train_loader:
        source, target = source.to(device), target.to(device)
        output = model(source)
        loss = loss_function(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if ((epoch + 1) % REPORT_EVERY) == 0:
        train_acc = evaluate(train_loader, model)
        dev_acc = evaluate(dev_loader, model)
        print(
            'epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
            (epoch + 1, total_loss, train_acc, dev_acc))


# In[11]:


torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model.pt')


# In[12]:


#--- test ---
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    batch_count = 0
    for source, target in test_loader:
        source, target = source.to(device), target.to(device)
        output = model(source)
        loss = loss_function(output, target)
        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        test_correct += (predicted == target).sum().item()
        batch_count += 1
    if batch_count:
        print('Test accuracy: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % (test_loss / batch_count, 100. * test_correct / total, test_correct, total))


# In[10]:


PER_CLUSTER = 10

def get_subset_and_plot(x, dist_cluster_index, dist_cluster_ids_x, title):
    x_subset = torch.index_select(
        x, 0,
        torch.tensor([index for index in range(len(x)) if dist_cluster_ids_x[index] == dist_cluster_index]))
    if len(x_subset) < 2:
        return
    x_subset = x_subset.detach()
    x_subset_np = x_subset.numpy()
    x_max = max(np.max(x_subset_np), np.abs(np.min(x_subset_np)))
    x_subset_np = np.divide(x_subset_np, x_max)
    nstd = np.std(x_subset_np)
    pca = PCA(n_components=2)
    x = pca.fit_transform(x_subset)
    print()
    print(f'{title} nstd = {nstd}')
    plt.figure(figsize=(4, 3), dpi = 160)
    plt.scatter(x[:, 0], x[:, 1])
    plt.title(title)
    plt.show()
    return nstd

def cluster_and_plot(test):
    test_title = 'test ' if test else ''
    model = make_model()
    optimizer = make_optimizer(model)
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    dist_weight = model.embed.weight
    indices = [index for index in range(len(dist_weight)) if vocab[index].startswith('#') and vocab[index].endswith('#')]
    dist_weight_subset = torch.index_select(dist_weight, 0, torch.tensor(indices))
    tokens_subset = torch.index_select(tokens, 0, torch.tensor(indices))
    if test:
        n_clusters = int(len(dist_weight_subset) / PER_CLUSTER)
        dist_cluster_ids_x = np.random.randint(0, n_clusters, len(dist_weight_subset)).tolist()
    else:
        dist_cluster_ids_x = KMeans(n_clusters=int(len(dist_weight_subset) / PER_CLUSTER)).fit_predict(
            dist_weight_subset.detach())
    phon_count = 0
    total_count = 0
    nstd_delta = 0
    nstd_sum_dist = 0
    nstd_sum_phon = 0
    for dist_cluster_index in range(0, max(dist_cluster_ids_x) + 1):
        vocab_subset = [vocab[indices[index]].strip('#') for index in range(len(dist_cluster_ids_x))
                        if dist_cluster_ids_x[index] == dist_cluster_index]
        print(vocab_subset)
        dist_weight = model.embed.weight
        nstd_dist = get_subset_and_plot(
            dist_weight_subset, dist_cluster_index, dist_cluster_ids_x, f'{test_title}dist cluster {dist_cluster_index + 1}')
        nstd_phon = get_subset_and_plot(
            tokens_subset, dist_cluster_index, dist_cluster_ids_x, f'{test_title}dist cluster {dist_cluster_index + 1} phon')
        if nstd_dist and nstd_phon:
            nstd_sum_dist += nstd_dist
            nstd_sum_phon += nstd_phon
            nstd_delta += nstd_phon - nstd_dist
            total_count += 1
        print()
        print()
    print(f'Nstd mean dist={nstd_sum_dist / total_count}')
    print(f'Nstd mean phon={nstd_sum_phon / total_count}')
    print(f'Nstd delta={nstd_delta}')

cluster_and_plot(False)

# cluster_and_plot(True)


# In[ ]:




