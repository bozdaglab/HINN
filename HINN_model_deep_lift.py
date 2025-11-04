# ---------------------------
# SECTION: Library Imports
# ---------------------------

import os
import sys
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from keras import regularizers
from keras.utils import plot_model

# Set backend for Keras to use PyTorch
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

import captum
from captum.attr import DeepLift
import plotly.express as px
import plotly.graph_objects as go

import plotly.colors as pc


# ---------------------------
# SECTION: Data Loading
# ---------------------------

def load_and_process_data():
    def preprocess(file_path, suffix):
        df = pd.read_csv(file_path)
        df.index = df.iloc[:, 0]
        df = df.drop(df.columns[0], axis=1)
        df.columns = [f"{col}_{suffix}" for col in df.columns]
        return df

    expression = preprocess("~/gene_data.csv", "expression")
    methy = preprocess("~/methyl_data.csv", "methy")
    snp = preprocess("~/snp_data.csv", "snp")
    demograph = pd.read_csv("~/demo_label_data.csv", usecols=range(7))
    demograph.index = demograph.iloc[:, 0]
    demograph = demograph.drop(demograph.columns[0], axis=1)
    demograph.columns = [f"{col}_demograph" for col in demograph.columns]

    label = pd.read_csv("~/demo_label_data.csv", usecols=[0, 8])
    label.index = label.iloc[:, 0]
    label = label.drop(label.columns[0], axis=1)
    label.columns = [f"{col}_label" for col in label.columns]

    # --- Inner join all datasets on their indices ---
    data = snp.join(expression, how="inner") \
              .join(methy, how="inner") \
              .join(demograph, how="inner") \
              .join(label, how="inner")
    return data

# ---------------------------
# SECTION: Custom Keras Layers
# ---------------------------

class PrimaryInputLayer(nn.Module):
    def __init__(self, units, output_dim, activation="sigmoid", mask=None):
        super().__init__()
        self.units = units
        self.output_dim = output_dim

        # activation
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # trainable weight and bias
        self.w = nn.Parameter(torch.empty(units, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

        nn.init.xavier_normal_(self.w)

        # non-trainable mask (same shape as w)
        if mask is None:
            raise ValueError("mask tensor is required")
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        # x: (batch, units)
        masked_w = self.w * self.mask
        out = x @ masked_w + self.b
        return self.activation(out)


class SecondaryInputLayer(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units

        # diagonal mask
        self.register_buffer("mask", torch.eye(units))

        self.w = nn.Parameter(torch.empty(units, units))
        nn.init.xavier_normal_(self.w)

    def forward(self, x):
        # x: (batch, units)
        masked_w = self.w * self.mask
        return x @ masked_w


class MultiplicationInputLayer(nn.Module):
    def __init__(self, units, activation="sigmoid"):
        super().__init__()
        self.units = units

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.b = nn.Parameter(torch.zeros(units))
        nn.init.xavier_normal_(self.b.unsqueeze(0))

    def forward(self, x):
        return self.activation(x + self.b)

# ---------------------------
# SECTION: Dataset for PyTorch
# ---------------------------

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return [input[idx] for input in self.inputs], self.targets[idx]
        
# ---------------------------
# SECTION: Early Stopping
# ---------------------------

class EarlyStopping:
def __init__(self, patience=50, delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        if val_loss < self.best_loss - self.delta:
            # Improvement: reset counter and save best state
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            # No improvement
            self.counter += 1

        # Check patience
        if self.counter >= self.patience:
            print(f"⏹ Early stopping triggered. Best val_loss = {self.best_loss:.4f}")
            if self.restore_best_weights and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
            return True

        return False

# ---------------------------
# SECTION: Model Training Pipeline
# ---------------------------

def train_model_torch(model, train_loader, val_loader, device="cpu",
                      lr=1e-3, epochs=1000, patience=500):
    criterion = torch.nn.L1Loss()  # MAE
    # criterion = torch.nn.MSELoss()  # MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=patience, delta=0.0, restore_best_weights=True)

    model.to(device)

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = [x.to(device).float() for x in inputs]
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(*inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- VAL -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device).float() for x in inputs]
                targets = targets.to(device).float()
                outputs = model(*inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # ----- EARLY STOPPING -----
        if early_stopper(val_loss, model):
            print(f"Stopping at epoch {epoch+1}")
            break

    return model

# ---------------------------
# SECTION: Evaluation Function
# ---------------------------

def evaluate_model_torch(model, test_loader, device="cpu"):
    model.eval()
    model.to(device)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = [x.to(device).float() for x in inputs]
            targets = targets.to(device).float().unsqueeze(1)
            preds = model(*inputs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0).squeeze()
    y_pred = np.concatenate(all_preds, axis=0).squeeze()

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    return {"mse": mse, "mae": mae, "y_true": y_true, "y_pred": y_pred}

 ---------------------------
# SECTION: Interpretation
# ---------------------------

def interpret_model(model, test_inputs, baselines, device="cpu"):

    model.eval()
    model.to(device)

    test_inputs = tuple(t.to(device) for t in test_inputs)
    baselines = tuple(b.to(device) for b in baselines)

    explainer = DeepLift(model)
    attributions = explainer.attribute(
        test_inputs,
        baselines=baselines,
        return_convergence_delta=False,
    )
    return attributions


# ---------------------------
# SECTION: Attribution Export
# ---------------------------

def export_attributions(attributions, feature_names, save_path_prefix):
    for i, name in enumerate(['snp', 'methy', 'gene', 'demo']):
        df = pd.DataFrame(attributions[i].detach().numpy(), columns=feature_names[i])
        df.to_csv(f"{save_path_prefix}_{name}.csv", index=False)


# ---------------------------
# SECTION: Matrix Filtering and Sankey Plotting
# ---------------------------

def filter_matrices_by_top_features(snp_list, methy_list, gene_list, 
                                     sparse_methy, sparse_gene, sparse_pathway):
    subset_methy_matrix = sparse_methy.loc[snp_list, methy_list]
    subset_gene_matrix = sparse_gene.loc[methy_list, gene_list]
    subset_pathway_matrix = sparse_pathway.loc[gene_list, :]

    subset_methy_matrix = subset_methy_matrix.loc[subset_methy_matrix.any(axis=1) == 1, subset_methy_matrix.any(axis=0)]
    subset_gene_matrix = subset_gene_matrix.loc[subset_gene_matrix.any(axis=1) == 1, subset_gene_matrix.any(axis=0)]
    subset_pathway_matrix = subset_pathway_matrix.loc[subset_pathway_matrix.index.isin(subset_gene_matrix.columns)]
    subset_pathway_matrix = subset_pathway_matrix.loc[subset_pathway_matrix.any(axis=1) == 1, subset_pathway_matrix.any(axis=0)]

    return subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix


def summarize_connections(*matrices):
    connection_counts = [int(matrix.sum().sum()) for matrix in matrices]
    labels = ["SNP-Methylation", "Methylation-Gene", "Gene-Pathway"]
    for label, count in zip(labels, connection_counts):
        print(f"Total connections ({label}): {count}")

def build_edge_list(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix):
    # SNP → Methylation edges
    edges_snp_methy = (
        subset_methy_matrix[subset_methy_matrix == 1]
        .stack()
        .reset_index()
    )
    edges_snp_methy.columns = ["source", "target", "value"]
    edges_snp_methy["layer"] = "snp_methy"

    # Methylation → Gene edges
    edges_methy_gene = (
        subset_gene_matrix[subset_gene_matrix == 1]
        .stack()
        .reset_index()
    )
    edges_methy_gene.columns = ["source", "target", "value"]
    edges_methy_gene["layer"] = "methy_gene"

    # Gene → GO term edges
    edges_gene_go = (
        subset_pathway_matrix[subset_pathway_matrix == 1]
        .stack()
        .reset_index()
    )
    edges_gene_go.columns = ["source", "target", "value"]
    edges_gene_go["layer"] = "gene_go"

    # Combine all edges
    edges_all = pd.concat(
        [edges_snp_methy, edges_methy_gene, edges_gene_go],
        ignore_index=True,
    )

    edges_all["value"] = 1

    return edges_all

def plot_sankey_from_edges(edges_all):
    edges_all_filtered = edges_all.copy()

    # Get all unique nodes
    nodes = pd.unique(edges_all_filtered[["source", "target"]].values.ravel())

    # Assign node categories
    snps = [
        node for node in nodes
        if ((node.startswith("rs") or ":" in node) and not node.startswith("GO"))
    ]
    methylation = [node for node in nodes if node.startswith("cg")]
    genes = [node for node in nodes if "_at" in node]
    go_terms = [node for node in nodes if node.startswith("GO:")]

    # Define node order (left → right)
    ordered_nodes = snps + methylation + genes + go_terms

    # Map node name → index
    node_indices = {name: i for i, name in enumerate(ordered_nodes)}

    # Keep only edges where both nodes are in ordered_nodes
    edges_all_filtered = edges_all_filtered[
        edges_all_filtered["source"].isin(ordered_nodes)
        & edges_all_filtered["target"].isin(ordered_nodes)
    ].copy()

    edges_all_filtered["source_index"] = edges_all_filtered["source"].map(node_indices)
    edges_all_filtered["target_index"] = edges_all_filtered["target"].map(node_indices)

    # x positions by category (normalized 0–1)
    node_positions_x = [
        0.0 if node in snps
        else 0.33 if node in methylation
        else 0.66 if node in genes
        else 0.99
        for node in ordered_nodes
    ]

    # Colors for nodes
    unique_colors = pc.qualitative.Dark24
    repeated_colors = (unique_colors * ((len(ordered_nodes) // len(unique_colors)) + 1))[:len(ordered_nodes)]
    node_colors = repeated_colors

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=ordered_nodes,
            color=node_colors,
            x=node_positions_x,
        ),
        link=dict(
            source=edges_all_filtered["source_index"],
            target=edges_all_filtered["target_index"],
            value=edges_all_filtered["value"],
        ),
    ))

    fig.update_layout(
        font_size=14,
        height=1500,
        width=2000,
        # title_text="SNP → Methylation → Gene → GO Term Associations",
    )
    fig.show()

# --------------------------
# SECTION: Torch HINN Model
# --------------------------

class HINN(nn.Module):
    def __init__(
        self,
        snp_dim,
        methy_dim,
        exp_dim,
        demo_dim,
        sparse_methy_tensor,
        sparse_gene_tensor,
        sparse_pathway_tensor,
        dense_nodes_1=128,
        drop_rate=0.7,
        activation_function="sigmoid",  # we will treat this as "only sigmoid"
    ):
        super().__init__()

        # --- First block: SNP -> Methy ---
        self.primary1 = PrimaryInputLayer(
            units=snp_dim,
            output_dim=methy_dim,
            activation=activation_function,  # "sigmoid"
            mask=sparse_methy_tensor,
        )
        self.secondary1 = SecondaryInputLayer(units=methy_dim)
        self.mult1 = MultiplicationInputLayer(
            units=methy_dim,
            activation=activation_function,  # "sigmoid"
        )

        self.snp_fc = nn.Linear(snp_dim, 20)  # con_cat_layer_first

        # --- Second block: Methy -> Gene (expression) ---
        self.primary2 = PrimaryInputLayer(
            units=methy_dim,
            output_dim=exp_dim,
            activation=activation_function,  # "sigmoid"
            mask=sparse_gene_tensor,
        )
        self.secondary2 = SecondaryInputLayer(units=exp_dim)
        self.mult2 = MultiplicationInputLayer(
            units=exp_dim,
            activation=activation_function,  # "sigmoid"
        )

        self.mid_fc = nn.Linear(methy_dim + 20, 20)  # con_cat_layer_sec

        # --- Third block: Gene -> Pathway ---
        pathway_dim = sparse_pathway_tensor.shape[1]
        self.primary3 = PrimaryInputLayer(
            units=exp_dim,
            output_dim=pathway_dim,
            activation=activation_function,  # "sigmoid"
            mask=sparse_pathway_tensor,
        )
        self.mid_fc2 = nn.Linear(exp_dim + 20, 20)  # con_cat_layer_third

        # --- Dense "custom_layers" stack ---
        custom_input_dim = pathway_dim + 20  # fourth_output + con_cat_layer_third

        self.bn1 = nn.BatchNorm1d(custom_input_dim)
        self.fc1 = nn.Linear(custom_input_dim, dense_nodes_1)
        self.drop1 = nn.Dropout(drop_rate)

        self.bn2 = nn.BatchNorm1d(dense_nodes_1)
        self.fc2 = nn.Linear(dense_nodes_1, dense_nodes_1)
        self.drop2 = nn.Dropout(drop_rate)

        self.bn3 = nn.BatchNorm1d(dense_nodes_1)
        self.fc3 = nn.Linear(dense_nodes_1, dense_nodes_1)
        self.drop3 = nn.Dropout(drop_rate)

        self.bn4 = nn.BatchNorm1d(dense_nodes_1)
        self.fc4 = nn.Linear(dense_nodes_1, dense_nodes_1)
        self.drop4 = nn.Dropout(drop_rate)

        self.dense_fourth = nn.Linear(dense_nodes_1, 20)  # dense_fourth
        self.bn_demo = nn.BatchNorm1d(20 + demo_dim)
        self.fc_demo = nn.Linear(20 + demo_dim, dense_nodes_1)
        self.drop_demo = nn.Dropout(drop_rate)

        self.out = nn.Linear(dense_nodes_1, 1)  # final output

        # store activation choice, but we'll only use sigmoid currently
        self.activation_function = activation_function

    def _nonlin(self, x):
        return torch.sigmoid(x)

    def forward(self, snp, methy, exp, demo):
        # --- First block ---
        primary1 = self.primary1(snp)
        secondary1 = self.secondary1(methy)
        mult_res1 = primary1 * secondary1
        mult1 = self.mult1(mult_res1)

        snp_fc = self._nonlin(self.snp_fc(snp))  # sigmoid(snp_fc)
        out2 = torch.cat([mult1, snp_fc], dim=1)

        # --- Second block ---
        primary2 = self.primary2(mult1)
        secondary2 = self.secondary2(exp)

        eps = 1e-6
        denom = primary2.clone()
        denom = torch.where(denom.abs() < eps, eps * torch.ones_like(denom), denom)

        div_res1 = secondary2 / denom
        div_res1 = torch.clamp(div_res1, -1e6, 1e6)

        mult2 = self.mult2(div_res1)

        mid_fc = self._nonlin(self.mid_fc(out2))
        out3 = torch.cat([mult2, mid_fc], dim=1)

        # --- Third block ---
        primary3 = self.primary3(mult2)
        mid_fc2 = self._nonlin(self.mid_fc2(out3))
        out4 = torch.cat([primary3, mid_fc2], dim=1)

        # --- Dense stack similar to custom_layers ---
        x = self.bn1(out4)
        x = self._nonlin(self.fc1(x))
        x = self.drop1(x)

        x = self.bn2(x)
        x = self._nonlin(self.fc2(x))
        x = self.drop2(x)

        x = self.bn3(x)
        x = self._nonlin(self.fc3(x))
        x = self.drop3(x)

        x = self.bn4(x)
        x = self._nonlin(self.fc4(x))
        x = self.drop4(x)

        dense_fourth = self._nonlin(self.dense_fourth(x))
        demo_concat = torch.cat([dense_fourth, demo], dim=1)

        x = self.bn_demo(demo_concat)
        x = self._nonlin(self.fc_demo(x))
        x = self.drop_demo(x)

        out = self.out(x)
        return out

# ---------------------------
# SECTION: Execution Pipeline
# ---------------------------

def main():
    device = "cpu"

    # ---------------------------
    # Data & splits
    # ---------------------------
    data = load_and_process_data()
    y = data["MMSE_label"]
    X = data.drop(columns=[c for c in data.columns if c.endswith("MMSE_label")])

    # train / test split
    X_train_int, X_test_df, y_train_int, y_test = train_test_split(
        X, y, test_size=0.3
    )
    # train / val split
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_int, y_train_int, test_size=0.2
    )

    # TRAIN
    X_train_snp = X_train_df.filter(like="_snp").values
    X_train_methy = X_train_df.filter(like="_methy").values
    X_train_exp = X_train_df.filter(like="_expression").values
    X_train_demo = X_train_df.filter(like="_demograph").values

    X_train_list = [
        torch.tensor(X_train_snp, dtype=torch.float32),
        torch.tensor(X_train_methy, dtype=torch.float32),
        torch.tensor(X_train_exp, dtype=torch.float32),
        torch.tensor(X_train_demo, dtype=torch.float32),
    ]
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

    # VAL
    X_val_snp = X_val_df.filter(like="_snp").values
    X_val_methy = X_val_df.filter(like="_methy").values
    X_val_exp = X_val_df.filter(like="_expression").values
    X_val_demo = X_val_df.filter(like="_demograph").values

    X_val_list = [
        torch.tensor(X_val_snp, dtype=torch.float32),
        torch.tensor(X_val_methy, dtype=torch.float32),
        torch.tensor(X_val_exp, dtype=torch.float32),
        torch.tensor(X_val_demo, dtype=torch.float32),
    ]
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    # TEST
    X_test_snp = X_test_df.filter(like="_snp").values
    X_test_methy = X_test_df.filter(like="_methy").values
    X_test_exp = X_test_df.filter(like="_expression").values
    X_test_demo = X_test_df.filter(like="_demograph").values

    X_test_list = [
        torch.tensor(X_test_snp, dtype=torch.float32),
        torch.tensor(X_test_methy, dtype=torch.float32),
        torch.tensor(X_test_exp, dtype=torch.float32),
        torch.tensor(X_test_demo, dtype=torch.float32),
    ]
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    # ---------------------------
    # Datasets & loaders
    # ---------------------------
    train_dataset = CustomDataset(X_train_list, y_train_t)
    val_dataset = CustomDataset(X_val_list, y_val_t)
    test_dataset = CustomDataset(X_test_list, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---------------------------
    # Sparse matrices -> tensors
    # ---------------------------
    sparse_methy = pd.read_csv("~/snp_methyl_matrix.csv", index_col=0)
    sparse_gene = pd.read_csv("~/methyl_gene_matrix.csv.zip", compression='zip', index_col=0)
    sparse_pathway = pd.read_csv("~/gene_pathway_matrix.csv", index_col=0)

    sparse_methy_tensor = torch.tensor(sparse_methy.values, dtype=torch.float32)
    sparse_gene_tensor = torch.tensor(sparse_gene.values, dtype=torch.float32)
    sparse_pathway_tensor = torch.tensor(sparse_pathway.values, dtype=torch.float32)

    # ---------------------------
    # Model init
    # ---------------------------
    snp_dim = X_train_snp.shape[1]
    methy_dim = X_train_methy.shape[1]
    exp_dim = X_train_exp.shape[1]
    demo_dim = X_train_demo.shape[1]

    activation_function = "sigmoid"
    dense_nodes_1 = 128
    drop_rate = 0.7

    model = HINN(
        snp_dim=snp_dim,
        methy_dim=methy_dim,
        exp_dim=exp_dim,
        demo_dim=demo_dim,
        sparse_methy_tensor=sparse_methy_tensor,
        sparse_gene_tensor=sparse_gene_tensor,
        sparse_pathway_tensor=sparse_pathway_tensor,
        dense_nodes_1=dense_nodes_1,
        drop_rate=drop_rate,
        activation_function=activation_function,
    )

    # ---------------------------
    # Train
    # ---------------------------
    model = train_model_torch(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=1e-3,
        epochs=1000,
        patience=50,
    )

    # ---------------------------
    # Evaluate on TRUE test set
    # ---------------------------
    eval_results = evaluate_model_torch(model, test_loader, device=device)
    print("MAE (Test):", eval_results["mae"])
    print("MSE (Test):", eval_results["mse"])

    # ---------------------------
    # Captum: DeepLift attributions
    # ---------------------------

    test_inputs = tuple(
        torch.tensor(arr, dtype=torch.float32, requires_grad=True).to(device)
        for arr in [X_test_snp, X_test_methy, X_test_exp, X_test_demo]
    )

    baselines = tuple(
        torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        .unsqueeze(0)
        .expand_as(torch.tensor(arr, dtype=torch.float32))
        .to(device)
        for arr in [X_test_snp, X_test_methy, X_test_exp, X_test_demo]
    )

    attributions = interpret_model(model, test_inputs, baselines, device=device)
    attr_snp, attr_methy, attr_gene, attr_demo = attributions

    # Mean absolute DeepLift attribution per feature
    snp_importance   = attr_snp.abs().mean(dim=0).detach().cpu().numpy()
    methy_importance = attr_methy.abs().mean(dim=0).detach().cpu().numpy()
    gene_importance  = attr_gene.abs().mean(dim=0).detach().cpu().numpy()

    # ---------------------------
    # Attribution export
    # ---------------------------
    feature_names = [
        X_train_df.filter(like=s).columns.tolist()
        for s in ["_snp", "_methy", "_expression", "_demograph"]
    ]
    export_attributions(attributions, feature_names, "~/MMSE")

    snp_feature_names   = feature_names[0]  # _snp
    methy_feature_names = feature_names[1]  # _methy
    gene_feature_names  = feature_names[2]  # _expression

    # Top-k per modality
    TOP_SNP   = 20
    TOP_METHY = 100
    TOP_GENE  = 50

    top_snp_idx   = np.argsort(-snp_importance)[:TOP_SNP]
    top_methy_idx = np.argsort(-methy_importance)[:TOP_METHY]
    top_gene_idx  = np.argsort(-gene_importance)[:TOP_GENE]

    snp_list = [
        snp_feature_names[i].replace("_snp", "")
        for i in top_snp_idx
    ]
    methy_list = [
        methy_feature_names[i].replace("_methy", "")
        for i in top_methy_idx
    ]
    gene_list = [
        gene_feature_names[i].replace("_expression", "")
        for i in top_gene_idx
    ]

    subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix = filter_matrices_by_top_features(
    snp_list, methy_list, gene_list, sparse_methy, sparse_gene, sparse_pathway
    )

    summarize_connections(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix)

    # Build edge list and plot old-style Sankey
    edges_all = build_edge_list(
        subset_methy_matrix,
        subset_gene_matrix,
        subset_pathway_matrix,
    )

    plot_sankey_from_edges(edges_all)

if __name__ == "__main__":
    main()
