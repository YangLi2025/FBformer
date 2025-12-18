"""Helper function for high-throughput GNN trainings."""
"""Implementation based on the template of ALIGNN."""

import os
import time
from typing import Optional, Sequence, Dict, Any

import matplotlib.pyplot as plt
from pydantic import ValidationError

from config import TrainingConfig
from train import train_dgl

plt.switch_backend("agg")


def train_prop_model(
    prop="",  
    dataset="dft_3d",  
    write_predictions=True,
    name="pygatt",
    save_dataloader=False,
    train_ratio=None,  
    classification_threshold=None,
    val_ratio=None,  
    test_ratio=None,  
    learning_rate=0.001,  
    batch_size=None,  
    scheduler=None,  
    n_epochs=None,  
    id_tag=None,
    num_workers=None,
    weight_decay=None,
    edge_input_features=None,
    triplet_input_features=None,  
    embedding_features=None,
    hidden_features=None,
    output_features=None,  
    random_seed=None,
    n_early_stopping=None,
    cutoff=None,
    max_neighbors=None,
    matrix_input=False,
    pyg_input=False,  
    use_lattice=True,  
    use_angle=True, 

    four_body=False,  

    output_dir=None,
    neighbor_strategy="k-nearest",  
    test_only=False,  
    use_save=True,
    mp_id_list=None,  
    file_name=None, 
    atom_features="cgcnn",
) -> Dict[str, Any]:

    if scheduler is None:
        scheduler = "onecycle"
    if batch_size is None:
        batch_size = 64
    if n_epochs is None:
        n_epochs = 500
    if num_workers is None:
        num_workers = 10

    model_name = name
    model_edge_input_features = edge_input_features
    model_hidden_features = hidden_features
    model_embedding_features = embedding_features
    model_output_features = output_features

    if four_body and model_edge_input_features is None:
        model_edge_input_features = 3


    config: Dict[str, Any] = {
        "dataset": dataset,
        "target": prop,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "weight_decay": 1e-05 if weight_decay is None else weight_decay,
        "learning_rate": learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": scheduler,
        "save_dataloader": save_dataloader,
        "pin_memory": False,
        "write_predictions": write_predictions,
        "num_workers": num_workers,
        "classification_threshold": classification_threshold,
        "atom_features": atom_features,
        "matrix_input": matrix_input,
        "pyg_input": pyg_input,
        "use_lattice": use_lattice,
        "use_angle": use_angle,
        "four_body": four_body,
        "neighbor_strategy": neighbor_strategy,
    }

    if output_dir is not None:
        config["output_dir"] = output_dir
    if id_tag is not None:
        config["id_tag"] = id_tag
    if random_seed is not None:
        config["random_seed"] = random_seed
    if file_name is not None:
        config["filename"] = file_name
    if n_early_stopping is not None:
        config["n_early_stopping"] = n_early_stopping
    if cutoff is not None:
        config["cutoff"] = cutoff
    if max_neighbors is not None:
        config["max_neighbors"] = max_neighbors

    if train_ratio is not None:
        config["train_ratio"] = train_ratio
        if val_ratio is None:
            raise ValueError("Enter val_ratio.")
        if test_ratio is None:
            raise ValueError("Enter test_ratio.")
        config["val_ratio"] = val_ratio
        config["test_ratio"] = test_ratio

    if dataset == "jv_3d":
        config["num_workers"] = 4
        config["pin_memory"] = False

    if dataset == "mp_3d_2020":
        config["id_tag"] = "id"
        config["num_workers"] = 0

    if dataset == "megnet2":
        config["id_tag"] = "id"
        config["num_workers"] = 0

    if dataset == "megnet":
        config["id_tag"] = "id"
        if prop in ("e_form", "gap pbe"):
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239
            config["num_workers"] = 8
        else:
            config["n_train"] = 4664
            config["n_val"] = 393
            config["n_test"] = 393

    if dataset == "oqmd_3d_no_cfid":
        config["id_tag"] = "_oqmd_entry_id"
        config["num_workers"] = 0

    if dataset == "hmof" and prop == "co2_absp":
        model_output_features = 5 

    if dataset == "edos_pdos":
        if prop == "edos_up":
            model_output_features = 300
        elif prop == "pdos_elast":
            model_output_features = 200
        else:
            raise ValueError("Target not available.")

    if dataset == "qm9_std_jctc":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 10829
        config["cutoff"] = 5.0
        config["standard_scalar_and_pca"] = False

    if dataset == "qm9_dgl":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 10831
        config["standard_scalar_and_pca"] = False
        config["batch_size"] = 64
        config["cutoff"] = 5.0
        if config["target"] == "all":
            model_output_features = 12

    if dataset == "hpov":
        config["id_tag"] = "id"

    if dataset == "qm9":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 13885
        config["batch_size"] = batch_size
        config["cutoff"] = 5.0
        config["max_neighbors"] = 9
        if prop in ["homo", "lumo", "gap", "zpve", "U0", "U", "H", "G"]:
            config["target_multiplication_factor"] = 27.211386024367243

    if test_only:
        t1 = time.time()

        try:
            cfg = TrainingConfig(**config)
        except ValidationError as e:
            print("error in converting to training config!")
            print(e)
            raise

        if getattr(cfg, "model", None) is not None:
            if hasattr(cfg.model, "name"):
                cfg.model.name = model_name
            if hasattr(cfg.model, "use_angle"):
                cfg.model.use_angle = bool(use_angle)
            if hasattr(cfg.model, "four_body"):
                cfg.model.four_body = bool(four_body)
            if model_edge_input_features is not None and hasattr(cfg.model, "edge_input_features"):
                cfg.model.edge_input_features = int(model_edge_input_features)
            if model_hidden_features is not None and hasattr(cfg.model, "hidden_features"):
                cfg.model.hidden_features = int(model_hidden_features)
            if model_embedding_features is not None and hasattr(cfg.model, "embedding_features"):
                cfg.model.embedding_features = int(model_embedding_features)
            if model_output_features is not None and hasattr(cfg.model, "output_features"):
                cfg.model.output_features = int(model_output_features)

        result = train_dgl(cfg, test_only=test_only, use_save=use_save, mp_id_list=mp_id_list)
        t2 = time.time()
        print("test mae=", result)
        print("Toal time:", t2 - t1)
        print("\n\n")
        return result

    try:
        cfg = TrainingConfig(**config)
    except ValidationError as e:
        print("error in converting to training config!")
        print(e)
        raise

    if getattr(cfg, "model", None) is not None:
        if hasattr(cfg.model, "name"):
            cfg.model.name = model_name
        if hasattr(cfg.model, "use_angle"):
            cfg.model.use_angle = bool(use_angle)
        if hasattr(cfg.model, "four_body"):
            cfg.model.four_body = bool(four_body)
        if model_edge_input_features is not None and hasattr(cfg.model, "edge_input_features"):
            cfg.model.edge_input_features = int(model_edge_input_features)
        if model_hidden_features is not None and hasattr(cfg.model, "hidden_features"):
            cfg.model.hidden_features = int(model_hidden_features)
        if model_embedding_features is not None and hasattr(cfg.model, "embedding_features"):
            cfg.model.embedding_features = int(model_embedding_features)
        if model_output_features is not None and hasattr(cfg.model, "output_features"):
            cfg.model.output_features = int(model_output_features)

    result = train_dgl(cfg, use_save=use_save, mp_id_list=mp_id_list)

    t2 = time.time()
    print("train=", result["train"])
    print("validation=", result["validation"])
    print("Toal time:", t2 - t1)
    print("\n\n")
    return result

