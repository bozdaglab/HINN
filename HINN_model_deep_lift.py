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

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import plot_model

from captum.attr import DeepLift
import plotly.express as px
import plotly.graph_objects as go

# Set backend for Keras to use PyTorch
os.environ["KERAS_BACKEND"] = "torch"


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
    
    # Read the demographic and label file
    demo_label_df = pd.read_csv("~/demo_label_data.csv")
    demo_label_df.index = demo_label_df.iloc[:, 0]
    demo_label_df = demo_label_df.drop(demo_label_df.columns[0], axis=1)
    
    # Split into demographic (first 6 columns) and label (7th column which is index 6)
    demograph = demo_label_df.iloc[:, :6].copy()
    demograph.columns = [f"{col}_demograph" for col in demograph.columns]
    
    label = demo_label_df.iloc[:, 6:7].copy()
    label.columns = ['MMSE_label']

    data = pd.concat([snp, expression, methy, demograph, label], axis=1)
    return data


# ---------------------------
# SECTION: Custom Keras Layers
# ---------------------------

class PrimaryInputLayer(keras.layers.Layer):
    def __init__(self, units=50, output_dim=32, activation='sigmoid', mask=None):
        super().__init__()
        self.units = units
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.mask = self.add_weight(shape=mask.shape, initializer="ones", trainable=False) * mask
        self.w = self.add_weight(shape=(units, output_dim), initializer="glorot_normal", trainable=True)
        self.b = self.add_weight(shape=(output_dim,), initializer="glorot_normal", trainable=True)

    def call(self, inputs):
        masked_weights = keras.ops.multiply(self.w, self.mask)
        return self.activation(keras.ops.matmul(inputs, masked_weights) + self.b)


class SecondaryInputLayer(keras.layers.Layer):
    def __init__(self, units=50):
        super().__init__()
        self.units = units
        self.mask = torch.eye(units, dtype=torch.float32, requires_grad=False)
        self.w = self.add_weight(shape=(units, units), initializer="glorot_normal", trainable=True)

    def call(self, inputs):
        masked_weights = keras.ops.multiply(self.w, self.mask)
        return keras.ops.matmul(inputs, masked_weights)


class MultiplicationInputLayer(keras.layers.Layer):
    def __init__(self, units=32, activation='sigmoid'):
        super().__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.b = self.add_weight(shape=(units,), initializer="glorot_normal", trainable=True)

    def call(self, inputs):
        return self.activation(inputs + self.b)


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
    def __init__(self, patience=50, delta=0, restore_best_weights=True):
        self.patience = patience
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_model_state)
            return True
        return False


# ---------------------------
# SECTION: Model Training Pipeline
# ---------------------------

def train_model(X_train_list, y_train, X_val_list, y_val, model):
    history = model.fit(
        x=X_train_list,
        y=y_train,
        batch_size=100,
        epochs=1000,
        shuffle=True,
        validation_data=(X_val_list, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=50,
                mode="auto",
                restore_best_weights=True
            )
        ]
    )
    return history


# ---------------------------
# SECTION: Evaluation Function
# ---------------------------

def evaluate_model(model, X_test_list, y_test):
    results = model.evaluate(x=X_test_list, y=y_test, verbose=2)
    predictions = model.predict(X_test_list)
    mse = np.mean((y_test - predictions.squeeze()) ** 2)
    mae = np.mean(np.abs(y_test - predictions.squeeze()))
    return {
        "results": results,
        "mse": mse,
        "mae": mae,
        "predictions": predictions
    }


# ---------------------------
# SECTION: Interpretation
# ---------------------------

def interpret_model(model, test_inputs, baselines):
    wrapped_model = torch.nn.Module()
    wrapped_model.forward = lambda *args: model(list(args))
    explainer = DeepLift(wrapped_model)
    attributions, delta = explainer.attribute(test_inputs, baselines=baselines, return_convergence_delta=True)
    return attributions, delta


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


def plot_sankey(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix):
    snps = subset_methy_matrix.index.tolist()
    methys = subset_methy_matrix.columns.tolist()
    genes = subset_gene_matrix.columns.tolist()
    pathways = subset_pathway_matrix.columns.tolist()

    all_nodes = snps + methys + genes + pathways
    node_labels = [str(n) for n in all_nodes]
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    def create_links(matrix, source_list, target_list):
        sources, targets, values = [], [], []
        for src in source_list:
            for tgt in target_list:
                if matrix.loc[src, tgt] == 1:
                    sources.append(node_indices[src])
                    targets.append(node_indices[tgt])
                    values.append(1)
        return sources, targets, values

    s1, t1, v1 = create_links(subset_methy_matrix, snps, methys)
    s2, t2, v2 = create_links(subset_gene_matrix, methys, genes)
    s3, t3, v3 = create_links(subset_pathway_matrix, genes, pathways)

    sources = s1 + s2 + s3
    targets = t1 + t2 + t3
    values = v1 + v2 + v3

    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="blue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )
    ])

    fig.update_layout(title_text="Multi-Omic Feature Flow in HINN Model", font_size=10)
    fig.show()


# ---------------------------
# SECTION: Execution Pipeline
# ---------------------------

def main():
    data = load_and_process_data()
    X = data.drop(columns=[col for col in data.columns if col.endswith('_label')])
    y = data['MMSE_label']
    
    # Split data into train, validation, and test sets
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val_df, X_test_df, y_val, y_test = train_test_split(X_temp_df, y_temp, test_size=0.5, random_state=42)

    # Extract features for each modality - TRAIN
    X_train_snp = X_train_df.filter(like='_snp').values.astype(np.float32)
    X_train_methy = X_train_df.filter(like='_methy').values.astype(np.float32)
    X_train_exp = X_train_df.filter(like='_expression').values.astype(np.float32)
    X_train_demo = X_train_df.filter(like='_demograph').values.astype(np.float32)

    # Extract features for each modality - VALIDATION
    X_val_snp = X_val_df.filter(like='_snp').values.astype(np.float32)
    X_val_methy = X_val_df.filter(like='_methy').values.astype(np.float32)
    X_val_exp = X_val_df.filter(like='_expression').values.astype(np.float32)
    X_val_demo = X_val_df.filter(like='_demograph').values.astype(np.float32)

    # Extract features for each modality - TEST
    X_test_snp = X_test_df.filter(like='_snp').values.astype(np.float32)
    X_test_methy = X_test_df.filter(like='_methy').values.astype(np.float32)
    X_test_exp = X_test_df.filter(like='_expression').values.astype(np.float32)
    X_test_demo = X_test_df.filter(like='_demograph').values.astype(np.float32)

    # Create lists for Keras
    X_train_list = [X_train_snp, X_train_methy, X_train_exp, X_train_demo]
    X_val_list = [X_val_snp, X_val_methy, X_val_exp, X_val_demo]
    X_test_list = [X_test_snp, X_test_methy, X_test_exp, X_test_demo]

    # Convert y to numpy arrays
    y_train = y_train.values.astype(np.float32)
    y_val = y_val.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    # Load sparse matrices
    sparse_methy = pd.read_csv("~/snp_methyl_matrix.csv", index_col=0)
    sparse_gene = pd.read_csv("~/methyl_gene_matrix.csv.zip", compression='zip', index_col=0)
    sparse_pathway = pd.read_csv("~/gene_pathway_matrix.csv", index_col=0)
    
    sparse_methy_tensor = torch.tensor(sparse_methy.values, dtype=torch.float32)
    sparse_gene_tensor = torch.tensor(sparse_gene.values, dtype=torch.float32)
    sparse_pathway_tensor = torch.tensor(sparse_pathway.values, dtype=torch.float32)

    # Define input layers for SNP, methylation, expression, and demographic data
    input_first_layer = Input(shape=(X_train_snp.shape[1],))
    input_second_layer = Input(shape=(X_train_methy.shape[1],))
    input_third_layer = Input(shape=(X_train_exp.shape[1],))
    input_fourth_layer = Input(shape=(X_train_demo.shape[1],))

    # Define missing config parameters
    activation_function = 'sigmoid'
    fully_activation_function = 'relu'
    kernel_initializer = 'glorot_normal'
    l2_reg = 0.01
    dense_nodes_1 = 64
    dense_nodes = 32
    drop_rate = 0.3
    
    # Define custom layers instances
    primary_output = PrimaryInputLayer(units=X_train_snp.shape[1],
                                       output_dim=X_train_methy.shape[1],
                                       activation=activation_function,
                                       mask=sparse_methy_tensor)(input_first_layer)

    secondary_output = SecondaryInputLayer(units=X_train_methy.shape[1])(input_second_layer)
    multiplication_result_1 = keras.ops.multiply(primary_output, secondary_output)
    multiplication_output = MultiplicationInputLayer(units=X_train_methy.shape[1], activation=activation_function)(multiplication_result_1)

    con_cat_layer_first = Dense(units=20, bias_initializer='zeros', activation=activation_function)(input_first_layer)
    output_2 = Concatenate()([multiplication_output, con_cat_layer_first])

    second_output = PrimaryInputLayer(units=X_train_methy.shape[1], output_dim=X_train_exp.shape[1], activation=activation_function, mask=sparse_gene_tensor)(multiplication_output)
    third_output = SecondaryInputLayer(units=X_train_exp.shape[1])(input_third_layer)
    division_result_1 = keras.ops.divide(third_output, second_output + 1e-7)
    division_output = MultiplicationInputLayer(units=X_train_exp.shape[1], activation=activation_function)(division_result_1)

    con_cat_layer_sec = Dense(units=20, bias_initializer='zeros', activation=activation_function)(output_2)
    output_3 = Concatenate()([division_output, con_cat_layer_sec])

    fourth_output = PrimaryInputLayer(units=X_train_exp.shape[1], output_dim=sparse_pathway.shape[1], activation=activation_function, mask=sparse_pathway_tensor)(division_output)
    con_cat_layer_third = Dense(units=20, bias_initializer='zeros', activation=activation_function)(output_3)
    output_4 = Concatenate()([fourth_output, con_cat_layer_third])

    # Define custom layers pipeline
    def batch_norm_layer(x):
        return BatchNormalization(axis=-1, momentum=0.9, epsilon=0.005, center=True, scale=True)(x)

    def dense_layer(x, units, activation, kernel_initializer, kernel_regularizer):
        return Dense(units=units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)

    def dropout_layer(x, rate):
        return Dropout(rate)(x)

    def custom_layers(x):
        for i in range(1, 4):
            x = batch_norm_layer(x)
            x = dense_layer(x, units=dense_nodes_1, activation=fully_activation_function, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(l2_reg))
            x = dropout_layer(x, drop_rate)

        x = batch_norm_layer(x)
        x = dense_layer(x, units=dense_nodes, activation=fully_activation_function, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(l2_reg))
        x = dropout_layer(x, drop_rate)

        dense_fourth = dense_layer(x, units=20, activation=activation_function, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(l2_reg))
        demo_complete_layer = Concatenate()([dense_fourth, input_fourth_layer])

        x = batch_norm_layer(demo_complete_layer)
        x = dense_layer(x, units=dense_nodes_1, activation=fully_activation_function, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(l2_reg))
        final_complete_layer = dropout_layer(x, drop_rate)

        return final_complete_layer

    # Define output layer
    outputs = Dense(units=1, activation='linear', kernel_initializer=kernel_initializer)(custom_layers(output_4))

    # Define and compile model
    model = Model(inputs=[input_first_layer, input_second_layer, input_third_layer, input_fourth_layer], outputs=outputs, name="HINN")
    model.compile(loss='mae', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

    train_model(X_train_list, y_train, X_val_list, y_val, model)
    results = evaluate_model(model, X_test_list, y_test)
    print("MAE (Test):", results['mae'])
    print("MSE (Test):", results['mse'])

    baselines = tuple(torch.tensor(arr.mean(axis=0), dtype=torch.float32).unsqueeze(0).expand_as(torch.tensor(arr, dtype=torch.float32)) for arr in [X_test_snp, X_test_methy, X_test_exp, X_test_demo])
    test_inputs = tuple(torch.tensor(arr, dtype=torch.float32, requires_grad=True) for arr in [X_test_snp, X_test_methy, X_test_exp, X_test_demo])
    attributions, delta = interpret_model(model, test_inputs, baselines)

    feature_names = [X_train_df.filter(like=s).columns.tolist() for s in ['_snp', '_methy', '_expression', '_demograph']]
    export_attributions(attributions, feature_names, "~/MMSE")

    snp_list = [name.replace('_snp', '') for name in feature_names[0][:20]]
    methy_list = [name.replace('_methy', '') for name in feature_names[1][:100]]
    gene_list = [name.replace('_expression', '') for name in feature_names[2][:50]]

    subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix = filter_matrices_by_top_features(
        snp_list, methy_list, gene_list, sparse_methy, sparse_gene, sparse_pathway
    )

    summarize_connections(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix)
    plot_sankey(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix)


if __name__ == "__main__":
    main()
