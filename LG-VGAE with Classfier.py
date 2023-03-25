# %%
# import packages
import warnings

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from LG_VGAE import LG_VGAE

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Define dataset
from dataset_E import EllipticDataset

dataset = EllipticDataset(
    raw_dir='I:/OneDrive - University of Nottingham Malaysia/moneylaundry_dataset/elliptic_bitcoin_dataset',
    processed_dir='I:/OneDrive - University of Nottingham Malaysia/moneylaundry_dataset/',
    self_loop=True,
    reverse_edge=True)

g, node_mask_by_time = dataset.process()
g = g.to(device)

num_classes = dataset.num_classes
labels = g.ndata['label'].to(device)

# All features
features = g.ndata['feat'][:, 1:166].to(device)

# Local features
# features = g.ndata['feat'][:, 1:95].to(device)

# %% Set Hyperparameters

# for LG-VGAE
Epoch = 1500
h_dim = 80  # Embedding Dimension h
z_dim = 40  # VAE Computing Dimension z
d = 2  # Polynomial Degree for Beta Wavelet Filter d
b = 0.50  # Balance constant b

# For Early stop strategy
patience = 50
cnt_wait = 0
best = float('inf')
best_t = 0

# %% Initialize LG-VGAE
model = LG_VGAE(g, 165, h_dim, z_dim, d, b).to(device)

LG_VGAE_optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# %%
# train LG-VGAE using early stop strategy

model.train()
for epoch in range(1, Epoch):
    loss = model(features)
    LG_VGAE_optimizer.zero_grad()
    loss.backward()
    LG_VGAE_optimizer.step()
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'I://best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break
    if (epoch % 20) == 0:
        print("Epoch {:05d} | Loss {:.4f} | ".format(epoch, loss.item()))

# %% Obtain the hidden representation h from the encoder

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('I://best_dgi.pkl'))

embeds = model.encoder(features)
embeds = embeds.detach()

# %% Split dataset into training, validation, and testing sets based on time steps

traininglist = range(34)
validationlist = range(36, 41)
testlist = range(34, 49)
labeled_idx_one = g.ndata['label'] == 1
labeled_idx_zero = g.ndata['label'] == 0
labeled_idx = labeled_idx_one | labeled_idx_zero

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

# %% Concatenate h and X to obtain combined feature matrix X'

# AF + NE
features_emb = torch.cat((features, embeds), 1)
X_train_hx = features_emb[train_mask].cpu().data.numpy()
X_test_hx = features_emb[test_mask].cpu().data.numpy()

# NE
X_train_h = features_emb[train_mask].cpu().data.numpy()
X_test_h = features_emb[test_mask].cpu().data.numpy()

# AF
X_train = features[train_mask].cpu().data.numpy()
X_test = features[test_mask].cpu().data.numpy()

y_train = labels[train_mask].cpu().data.numpy()
y_test = labels[test_mask].cpu().data.numpy()

# %% train classifier RandomForest
# ALL feature( all 165 features)
model_RF = RandomForestClassifier(class_weight={0: 1, 1: 3}, n_estimators=50, max_features=50).fit(X_train, y_train)
y_preds = model_RF.predict(X_test)
#
prec, rec, f1, num = precision_recall_fscore_support(y_test, y_preds, labels=[1, 0])
#
print("AF RandomForest Classifier")
print("Illicit Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec[0], rec[0], f1[0]))

# %%
# ALL feature + embeddings
# random_state = 781

model_RF_hx = RandomForestClassifier(criterion='entropy', bootstrap=False, class_weight={0: 1, 1: 3}, n_estimators=140,
                                     max_features=20, random_state=252).fit(X_train_hx, y_train)
y_preds = model_RF_hx.predict(X_test_hx)
#
prec, rec, f1, num = precision_recall_fscore_support(y_test, y_preds, labels=[1, 0])
#
print("ALL feature + embeddings RandomForest Classifier")
print("Illicit Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec[0], rec[0], f1[0]))

# %% Node embeddings
model_NB_RF = RandomForestClassifier(class_weight={0: 1, 1: 3}, n_estimators=50, max_features=50).fit(X_train_h,
                                                                                                      y_train)
y_preds = model_NB_RF.predict(X_test_h)

prec, rec, f1, num = precision_recall_fscore_support(y_test, y_preds, labels=[1, 0])

print("Node embeddings RandomForest Classifier")
print("Illicit Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec[0], rec[0], f1[0]))
