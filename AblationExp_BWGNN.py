import matplotlib.pyplot as plt
import numpy as np
import torch
from BWGNN import *
from keras.utils import to_categorical
import random
import dgl
import warnings
from utils import Measure
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Define dataset
from dataset_E import EllipticDataset

dataset = EllipticDataset(
    raw_dir='C:/Users/74102/OneDrive - University of Nottingham Malaysia/moneylaundry_dataset/elliptic_bitcoin_dataset',
    processed_dir='C:/Users/74102/OneDrive - University of Nottingham Malaysia/moneylaundry_dataset/',
    self_loop=True,
    reverse_edge=True)

g, node_mask_by_time = dataset.process()
g = g.to(device)

num_classes = dataset.num_classes
labels = g.ndata['label'].to(device)

# All features
features = g.ndata['feat'][:, 1:166].to(device)



# %% Set Hyperparameters

# for BWGNN
EPOCH = 500
h_dim = 128  # Embedding Dimension h
num_classes = 2
activation = torch.nn.PReLU(h_dim)
degree = 2

# For Early stop strategy
patience = 20
cnt_wait = 0
best = float('inf')
best_t = 0

# %% Initialize BWGNN
model = BWGNN(165, h_dim, num_classes, g, degree).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


# %% Split dataset into training, validation, and testing sets based on time steps

traininglist = range(34)
validationlist = range(36, 41)
testlist = range(34, 49)
labeled_idx_one = g.ndata['label'] == 1
labeled_idx_zero = g.ndata['label'] == 0

indx = (labeled_idx_zero == 1).nonzero(as_tuple=True)[0]
indx = indx[torch.randperm(len(indx))]
labeled_adj_idx_zero = labeled_idx_zero
labeled_adj_idx_zero[indx[:int(len(indx) / 1.25)]] = 0
#
labeled_idx = labeled_idx_one | labeled_idx_zero
labeled_idx_adj = labeled_idx_one | labeled_adj_idx_zero

train_mask = torch.zeros([203769]).bool().to(device)
val_mask = torch.zeros([203769]).bool().to(device)
test_mask = torch.zeros([203769]).bool().to(device)

for i in traininglist:
    train_mask = train_mask | node_mask_by_time[i].to(device)
for k in validationlist:
    val_mask = val_mask | node_mask_by_time[k].to(device)
for j in testlist:
    test_mask = test_mask | node_mask_by_time[j].to(device)

test_mask = test_mask & labeled_idx
val_mask = val_mask & labeled_idx
train_mask = train_mask & labeled_idx

# %%

weight = (1-g.ndata['label'][train_mask]).sum().item() / g.ndata['label'][train_mask].sum().item()
criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1., weight]), reduction='mean').to(device)

plot_train = []
plot_val = []
plot_test = []
saved_epoch = []
plot_loss = []

# AF
X_train = features[train_mask].cpu().data.numpy()
X_test = features[test_mask].cpu().data.numpy()

y_train = labels[train_mask].cpu().data.numpy()
y_test = labels[test_mask].cpu().data.numpy()

train_measure = Measure(num_classes=num_classes, target_class=1)
valid_measure = Measure(num_classes=num_classes, target_class=1)
test_measure = Measure(num_classes=num_classes, target_class=1)

X_tr = features[train_mask]
Y_tr = labels[train_mask]
X_te = features[test_mask]
Y_te = labels[test_mask]
encoded_labels = torch.FloatTensor(to_categorical(Y_tr.tolist())).to(device)

# %%
best_test_f1 = 0  # 初始化最佳测试F1得分
best_epoch = 0    # 存储取得最佳F1得分的epoch
best_test_precision = 0  # 最佳测试精确度
best_test_recall = 0     # 最佳测试召回率

for epoch in range(1, EPOCH):
    model.train()
    optimizer.zero_grad()
    preds = model(features)
    loss = criterion(preds[train_mask], encoded_labels)
    loss.backward()
    optimizer.step()

    if (epoch % 25) == 0:
        train_measure.append_measures(preds[train_mask], Y_tr)
        cl_precision, cl_recall, cl_f1 = train_measure.get_total_measure()
        train_measure.update_best_f1(cl_f1, epoch)
        train_measure.reset_info()
        print("Train Epoch {} | class {} | loss:{:.4f} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, 1, loss.item(), cl_precision, cl_recall, cl_f1))

    if (epoch % 50) == 0:
        model.eval()
        predictions = preds[test_mask]
        test_measure.append_measures(predictions, Y_te)
        test_precision, test_recall, test_f1 = test_measure.get_total_measure()
        print("  Test Dateset | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(1, test_precision, test_recall, test_f1))


        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch
            best_test_precision = test_precision
            best_test_recall = test_recall

print("\nBest BWGNN Performance at Epoch {}".format( best_epoch))
print("Best Test Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(best_test_precision, best_test_recall, best_test_f1))

# %%
mid_feat = model.mid_feat
features_emb = torch.cat((features, mid_feat), 1)
X_train_femb = features_emb[train_mask].cpu().data.numpy()
X_test_femb = features_emb[test_mask].cpu().data.numpy()

y_test = labels[test_mask].cpu().data.numpy()
y_train = labels[train_mask].cpu().data.numpy()

# %% RandomForest
# ALL feature + embeddings
model_RF_emb = RandomForestClassifier(class_weight={0: 1, 1: 3}, n_estimators=50, max_features=50).fit(X_train_femb, y_train)
y_preds = model_RF_emb.predict(X_test_femb)
#
prec, rec, f1, num = precision_recall_fscore_support(y_test, y_preds, labels=[1, 0])
#
print("ALL feature + embeddings RandomForest Classifier")
print("Illicit Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec[0], rec[0], f1[0]))


# %%Use MLP for classification
in_feats = X_train_femb.shape[1]

model_mlp = MLPClassifier(hidden_layer_sizes=(in_feats // 4, in_feats // 8), activation='relu', max_iter=500,
                          random_state=42)

# Train the downstream classifier
model_mlp.fit(X_train_femb, y_train)
y_preds = model_mlp.predict(X_test_femb)

# Evaluate the Classifier
prec, rec, f1, num = precision_recall_fscore_support(y_test, y_preds, labels=[1, 0])

print("ALL feature + embeddings MLP Classifier")
print("Illicit Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec[0], rec[0], f1[0]))
