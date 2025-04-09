"""
Utility functions and data loading
"""
from typing import Tuple, Optional
import sys
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import urllib.request
from feature_engineering import feature_extraction


class MicrostructureImageDataset(Dataset):
    def __init__(self, file_path, group_name):
        self.file = h5py.File(file_path, 'r')
        self.images = self.file[group_name]["image_data"][...]
        self.features = self.file[group_name]["feature_vector"][...]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]).float().unsqueeze(0), torch.tensor(self.features[idx]).unsqueeze(0)


def get_param_fields(image, params):
    """
    Construct parameter fields for a two-phase microstructure described by image based on params.

    :param image:
    :param params:
    :return:
    """
    return params[0] * (1. - image) + params[1] * image


def get_node_coords(image, lx=1.0, ly=1.0):
    """

    :return:
    """
    nx, ny = image.shape[-2], image.shape[-1]
    x = torch.linspace(-lx / 2.0, lx / 2.0 - 1.0 / nx, nx, dtype=image.dtype, device=image.device)
    y = torch.linspace(-ly / 2.0, ly / 2.0 - 1.0 / ny, ny, dtype=image.dtype, device=image.device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    return X, Y


def get_macro_temp(
    temp: torch.Tensor, loading: Optional[torch.Tensor] = None, lx=1.0, ly=1.0, fluctuation_scaling=1.0
) -> torch.Tensor:
    """
    Compute corresponding macroscopic temperature based on microscopic temperature temp and a given loading

    :param temp:
    :param loading:
    :return:
    """
    if loading is None:
        loading = torch.eye(2, dtype=temp.dtype, device=temp.device)
    X, Y = get_node_coords(temp, lx=lx, ly=ly)
    coords = torch.stack([X, Y], dim=-3)
    macro_temp = torch.einsum("...ij,jyx->...ixy", loading, coords).unsqueeze(dim=-3) + fluctuation_scaling * temp
    return macro_temp


def homogenize(field: torch.Tensor) -> torch.Tensor:
    """
    Homogenization of a tensor field on a 2d grid by volume averaging

    :param field: tensor field with shape [..., n, n]
    :return:
    """
    return field.nanmean([-2, -1])


def darus_download(repo_id, file_id, file_path):
    darus_url = rf"https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/darus-{repo_id}/{file_id}"
    urllib.request.urlretrieve(darus_url, file_path)


def load_fnocg_model(problem="thermal", dim=2, bc="per", n_layers=15, device="cuda", dtype=torch.float64, compile_model=True):
    device_key = "cpu" if device == "cpu" else "cuda"
    dtype_key = "float32" if dtype == torch.float32 else "float64"
    model_name = f"fnocg_thermal_2d_per"

    if (not compile_model) or (device == "cpu"):
        print("Creating model in Python...", end=" ")
        with torch.inference_mode():
            model = create_fnocg_model(model_name=model_name, n_layers=n_layers, device=device, dtype=dtype)

        warmup_model(model, device=device, dtype=dtype)
        print("Successful", flush=True)
        return model
        
    try:
        # Try to load precompiled *.so file
        print("Trying to load precompiled model...", end=" ")
        so_path = os.path.join("models", f"{model_name}_{n_layers}l_{device_key}_{dtype_key}.so")
        with torch.inference_mode():
            model = torch._export.aot_load(so_path, device)
        
        warmup_model(model, device=device, dtype=dtype)
        print("Successful", flush=True)
        return model
    except:
        # Recreate model in PyTorch and then try to compile it
        print("Failed", flush=True)
        print("Creating model in Python...", end=" ")
        model = create_fnocg_model(model_name=model_name, n_layers=n_layers, device=device, dtype=dtype)
        warmup_model(model, device=device, dtype=dtype)
        print("Successful", flush=True)

        try:
            # First try to compile the recreated model to a *.so file and then load it
            print("Trying to compile to a C++ shared library...", end=" ")
            compile_options = {"max_autotune": True, "epilogue_fusion": True, "triton.cudagraphs": True}
            so_path = os.path.join("models", f"{model_name}_{n_layers}l_{device_key}_{dtype_key}.so")
            with torch.no_grad():
                param_field = torch.rand(1, 1, 400, 400, device=device, dtype=dtype)
                loading = torch.eye(2, device=device, dtype=dtype)
                so_path = torch._export.aot_compile(model, (param_field, loading,), options={**compile_options, "aot_inductor.output_path": so_path})
    
            with torch.inference_mode():
                model = torch._export.aot_load(so_path, device)
            warmup_model(model, device=device, dtype=dtype)
            print("Successful", flush=True)
            return model
        except:
            print("Failed", flush=True)
            print("Trying to compile in Python environment...", end=" ")
            try:
                # If this is not supported, compile the model in the Python environment
                with torch.inference_mode():
                    model = torch.compile(model, options=compile_options)
                warmup_model(model, device=device, dtype=dtype)
                return model
            except:
                print("Failed", flush=True)
                print("Returning uncompiled model")
                # If this is also not supported, just return the uncompiled model
                warmup_model(model, device=device, dtype=dtype)
                return model


def create_fnocg_model(model_name, n_layers, device, dtype):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fnocg_egg_path = os.path.abspath(os.path.join(dir_path, "models", f"{model_name}.egg"))
    sys.path.append(fnocg_egg_path)
    model = torch.load(os.path.join("models", f"{model_name}_{n_layers}l.pt"), weights_only=False, map_location=device)
    model = model.to(device=device, dtype=dtype)
    return model

def warmup_model(model, device, dtype, n_iter=3):
    param_field = torch.rand(1, 1, 400, 400, device=device, dtype=dtype)
    loading = torch.eye(2, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(n_iter):
            model(param_field, loading)
            if device != "cpu":
                torch.cuda.synchronize()

def get_surrogate_features(param_field):
    kappa0, kappa1 = param_field.min().item(), param_field.max().item()
    R = kappa0/kappa1       # Phase contrast ratio
    
    image = (param_field == kappa1).float().cpu().numpy().squeeze(0)
    features = feature_extraction.full_computation(image)
    return torch.tensor(np.append(features, [[1/R, R]], axis=1))

def get_sym_indices(dim):
    diag_idx = (torch.arange(dim), torch.arange(dim))    
    row, col = torch.tril_indices(dim, dim, -1)
    dof_idx = (torch.cat([diag_idx[0], row]), torch.cat([diag_idx[1], col]))
    return dof_idx

def pack_sym(symmetric_matrix, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(symmetric_matrix.device) for idx in dof_idx)
    return symmetric_matrix[(..., *dof_idx) if symmetric_matrix.dim() == 3 else dof_idx]

def unpack_sym(packed_values, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(packed_values.device) for idx in dof_idx)
    matrix = torch.zeros((*packed_values.shape[:-1], dim, dim), dtype=packed_values.dtype, device=packed_values.device)
    if packed_values.dim() == 2:
        matrix[:, dof_idx[0], dof_idx[1]] = packed_values
        return matrix + matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(matrix, dim1=1, dim2=2))
    matrix[dof_idx] = packed_values
    return matrix + matrix.T - torch.diag(torch.diag(matrix))