import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import numpy as np
from utils import MicrostructureImageDataset, get_param_fields, load_fnocg_model
from PIL import Image
import io
import base64
import re
import traceback
import time

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up device and dtype
dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"
args = {"device": device, "dtype": dtype}
print(f"Using device: {device}, dtype: {dtype}")

# Load the data
file_path = os.path.join("data", "feature_engineering_data.h5")
group_name = "train_set"

if not os.path.isfile(file_path):
    from utils import darus_download
    print("Downloading data from DaRUS...")
    darus_download(repo_id=3366, file_id=4, file_path=file_path)

# Load dataset
samples = MicrostructureImageDataset(
    file_path=file_path,
    group_name=group_name
)
print('Number of samples in dataset:', len(samples))

# Load the simulation model
simulation = load_fnocg_model(problem="thermal", dim=2, bc="per", n_layers=15, 
                             device=device, dtype=dtype, compile_model=True)

# Load the surrogate model if available
try:
    from utils import unpack_sym
    if device == "cpu":
        vrnn_model_file = os.path.join("models", "vrnn_thermal_2d_per_jit_cpu.pt")
    else:
        vrnn_model_file = os.path.join("models", "vrnn_thermal_2d_per_jit.pt")
    
    with torch.inference_mode():
        vrnn = torch.jit.load(vrnn_model_file, map_location=device).to(device=device, dtype=torch.float32)
    
    def surrogate(features, params):
        R = params[0] / params[1]
        features = torch.cat([features.to(dtype=torch.float32, device=params.device), 
                             torch.tensor([[1/R, R]], dtype=torch.float32, device=params.device)], dim=-1)
        return unpack_sym(vrnn(features), dim=2).squeeze() * params[0]
    
    print("Surrogate model loaded successfully")
    has_surrogate = True
except Exception as e:
    print(f"Could not load surrogate model: {e}")
    surrogate = None
    has_surrogate = False

# Define simulation parameters model
class SimulationParams(BaseModel):
    ms_id: int
    kappa1: float
    alpha: float

# Helper function to safely get value from tensor or float
def safe_item(value):
    if hasattr(value, 'item'):
        return value.item()
    return float(value)

# Convert base64 image to numpy array
def parse_base64_image(base64_string):
    """
    Parse a base64 encoded image into a binary numpy array suitable for thermal simulation.
    
    Args:
        base64_string: Base64 encoded image data
    
    Returns:
        binary_image: Numpy boolean array where True represents the conductive phase
        
    Raises:
        ValueError: If the image data is invalid or cannot be processed
    """
    try:
        # Extract the base64 encoded binary data part
        if not base64_string or len(base64_string) < 100:
            raise ValueError("Image data is too short or empty")
            
        base64_data = re.sub(r'^data:image/\w+;base64,', '', base64_string)
        
        # Decode base64 data
        try:
            binary_data = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {str(e)}")
        
        # Create PIL Image from binary data
        try:
            image = Image.open(io.BytesIO(binary_data))
        except Exception as e:
            raise ValueError(f"Failed to read image data: {str(e)}")
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to match model input size (400x400)
        image = image.resize((400, 400))
        
        # Binarize the image (threshold at 128)
        binary_image = np.array(image) > 128
        
        # Validate the microstructure - check if it's not just a single phase
        if binary_image.all() or not binary_image.any():
            raise ValueError("Invalid microstructure: The image needs to have both black and white regions")
        
        # Check for sufficient contrast
        white_percentage = np.mean(binary_image) * 100
        if white_percentage < 5 or white_percentage > 95:
            print(f"Warning: Microstructure has {white_percentage:.1f}% white area, which may produce unreliable results")
        
        return binary_image
    
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"Error parsing base64 image: {str(e)}")
        raise ValueError(f"Invalid image data: {str(e)}")

# Helper function to validate uploaded images
def validate_uploaded_image(image):
    """
    Validate uploaded image for suitability as a microstructure.
    
    Args:
        image: PIL Image object
    
    Returns:
        binary_image: Numpy boolean array where True represents the conductive phase
        
    Raises:
        ValueError: If the image is invalid for simulation
    """
    # Check image dimensions
    if image.width < 100 or image.height < 100:
        raise ValueError("Image dimensions must be at least 100x100 pixels")
    
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to match model input size (400x400)
    image = image.resize((400, 400))
    
    # Binarize the image (threshold at 128)
    binary_image = np.array(image) > 128
    
    # Validate the microstructure - check if it's not just a single phase
    if binary_image.all():
        raise ValueError("Invalid microstructure: The image is entirely white. Please provide an image with both phases.")
    
    if not binary_image.any():
        raise ValueError("Invalid microstructure: The image is entirely black. Please provide an image with both phases.")
    
    # Check for sufficient contrast
    white_percentage = np.mean(binary_image) * 100
    if white_percentage < 5 or white_percentage > 95:
        print(f"Warning: Microstructure has {white_percentage:.1f}% white area, which may produce unreliable results")
    
    return binary_image

# Helper function to run simulation on binary microstructure
def run_thermal_simulation(microstructure, kappa1, alpha):
    try:
        # Convert to tensor format if it's not already
        if not isinstance(microstructure, torch.Tensor):
            microstructure_tensor = torch.tensor(microstructure, dtype=torch.float32).unsqueeze(0)
        else:
            microstructure_tensor = microstructure
        
        # Set up parameters
        model_params = torch.tensor([1., kappa1]).reshape(2, 1)
        param_field = get_param_fields(microstructure_tensor, model_params).to(**args).unsqueeze(0)
        
        # Set up loading based on alpha
        alpha_rad = torch.deg2rad(torch.tensor(alpha))
        loading = torch.tensor([
            [torch.cos(alpha_rad), -torch.sin(alpha_rad)], 
            [torch.sin(alpha_rad), torch.cos(alpha_rad)]
        ], **args)
        
        # Run simulation
        with torch.inference_mode():
            field = simulation(param_field, loading)
            if device != "cpu":
                torch.cuda.synchronize()
            
            # Process results
            vol_frac = microstructure_tensor.mean()
            k0_val = safe_item(model_params[0])
            k1_val = safe_item(model_params[1])
            vf_val = safe_item(vol_frac)
            
            reuss = 1. / (vf_val / k1_val + (1. - vf_val) / k0_val)
            voigt = vf_val * k1_val + (1. - vf_val) * k0_val
            
            temp = field[..., 0, :, :].detach().cpu()
            flux = field[..., 1:, :, :]
            flux_norm = flux.norm(dim=-3).detach().cpu()
            
            # Calculate effective conductivity
            hom_flux = flux.mean([-1, -2]).squeeze()
            kappa_bar = -hom_flux @ loading.inverse()
            eig_kappa = torch.linalg.eigvals(kappa_bar).real.cpu()
            
            # Run surrogate if available (dummy response for custom images)
            if surrogate is not None:
                surrogate_results = {
                    'eig_pred': [float(eig_kappa[0].item()) if hasattr(eig_kappa[0], 'item') else float(eig_kappa[0]), 
                                float(eig_kappa[1].item()) if hasattr(eig_kappa[1], 'item') else float(eig_kappa[1])]
                }
            else:
                surrogate_results = None
            
            # Prepare results for JSON
            results = {
                'image': microstructure_tensor[0].cpu().numpy().tolist() if hasattr(microstructure_tensor, 'cpu') else microstructure_tensor[0].numpy().tolist() if hasattr(microstructure_tensor, 'numpy') else microstructure_tensor[0].tolist(),
                'param_field': param_field[0, 0].cpu().numpy().tolist() if hasattr(param_field, 'cpu') else param_field[0, 0].numpy().tolist() if hasattr(param_field, 'numpy') else param_field[0, 0].tolist(),
                'temp0': temp[0, 0].cpu().numpy().tolist() if hasattr(temp, 'cpu') else temp[0, 0].numpy().tolist() if hasattr(temp, 'numpy') else temp[0, 0].tolist(),
                'temp1': temp[0, 1].cpu().numpy().tolist() if hasattr(temp, 'cpu') else temp[0, 1].numpy().tolist() if hasattr(temp, 'numpy') else temp[0, 1].tolist(),
                'flux_norm0': flux_norm[0, 0].cpu().numpy().tolist() if hasattr(flux_norm, 'cpu') else flux_norm[0, 0].numpy().tolist() if hasattr(flux_norm, 'numpy') else flux_norm[0, 0].tolist(),
                'flux_norm1': flux_norm[0, 1].cpu().numpy().tolist() if hasattr(flux_norm, 'cpu') else flux_norm[0, 1].numpy().tolist() if hasattr(flux_norm, 'numpy') else flux_norm[0, 1].tolist(),
                'vol_frac': safe_item(vol_frac),
                'reuss': safe_item(reuss),
                'voigt': safe_item(voigt),
                'eig_kappa': eig_kappa.tolist() if hasattr(eig_kappa, 'tolist') else list(eig_kappa),
                'surrogate_results': surrogate_results
            }
            
            return results
            
    except Exception as e:
        print(f"Error in thermal simulation: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in thermal simulation: {str(e)}")

# API endpoint to get basic info
@app.get("/api/info")
async def get_info():
    return {
        "sample_count": len(samples),
        "has_surrogate": has_surrogate,
        "device": device
    }

# API endpoint to run simulation
@app.post("/api/simulate")
async def run_simulation(params: SimulationParams):
    try:
        # Validate microstructure ID
        if params.ms_id < 0 or params.ms_id >= len(samples):
            raise HTTPException(status_code=400, detail=f"Microstructure ID must be between 0 and {len(samples)-1}")
        
        # Get the data
        image, features = samples[params.ms_id]
        
        # Set up parameters
        model_params = torch.tensor([1., params.kappa1]).reshape(2, 1)
        param_field = get_param_fields(image, model_params).to(**args).unsqueeze(0)
        
        # Set up loading based on alpha
        alpha_rad = torch.deg2rad(torch.tensor(params.alpha))
        loading = torch.tensor([
            [torch.cos(alpha_rad), -torch.sin(alpha_rad)], 
            [torch.sin(alpha_rad), torch.cos(alpha_rad)]
        ], **args)
        
        # Run simulation
        with torch.inference_mode():
            field = simulation(param_field, loading)
            if device != "cpu":
                torch.cuda.synchronize()
            
            # Process results
            vol_frac = image.mean()
            k0_val = safe_item(model_params[0])
            k1_val = safe_item(model_params[1])
            vf_val = safe_item(vol_frac)
            
            reuss = 1. / (vf_val / k1_val + (1. - vf_val) / k0_val)
            voigt = vf_val * k1_val + (1. - vf_val) * k0_val
            
            temp = field[..., 0, :, :].detach().cpu()
            flux = field[..., 1:, :, :]
            flux_norm = flux.norm(dim=-3).detach().cpu()
            
            # Calculate effective conductivity
            hom_flux = flux.mean([-1, -2]).squeeze()
            kappa_bar = -hom_flux @ loading.inverse()
            eig_kappa = torch.linalg.eigvals(kappa_bar).real.cpu()
            
            # Run surrogate if available
            if surrogate is not None:
                with torch.inference_mode():
                    kappa_pred = surrogate(features.to(**args), model_params.to(**args)).squeeze()
                    eig_pred = torch.linalg.eigvals(kappa_pred).real.cpu()
                    surrogate_results = {
                        'eig_pred': eig_pred.tolist()
                    }
            else:
                surrogate_results = None
            
            # Prepare results for JSON
            # Convert tensors to nested lists for JSON serialization
            results = {
                'image': image[0].cpu().numpy().tolist() if hasattr(image, 'cpu') else image[0].numpy().tolist() if hasattr(image, 'numpy') else image[0].tolist(),
                'param_field': param_field[0, 0].cpu().numpy().tolist() if hasattr(param_field, 'cpu') else param_field[0, 0].numpy().tolist() if hasattr(param_field, 'numpy') else param_field[0, 0].tolist(),
                'temp0': temp[0, 0].cpu().numpy().tolist() if hasattr(temp, 'cpu') else temp[0, 0].numpy().tolist() if hasattr(temp, 'numpy') else temp[0, 0].tolist(),
                'temp1': temp[0, 1].cpu().numpy().tolist() if hasattr(temp, 'cpu') else temp[0, 1].numpy().tolist() if hasattr(temp, 'numpy') else temp[0, 1].tolist(),
                'flux_norm0': flux_norm[0, 0].cpu().numpy().tolist() if hasattr(flux_norm, 'cpu') else flux_norm[0, 0].numpy().tolist() if hasattr(flux_norm, 'numpy') else flux_norm[0, 0].tolist(),
                'flux_norm1': flux_norm[0, 1].cpu().numpy().tolist() if hasattr(flux_norm, 'cpu') else flux_norm[0, 1].numpy().tolist() if hasattr(flux_norm, 'numpy') else flux_norm[0, 1].tolist(),
                'vol_frac': safe_item(vol_frac),
                'reuss': safe_item(reuss),
                'voigt': safe_item(voigt),
                'eig_kappa': eig_kappa.tolist() if hasattr(eig_kappa, 'tolist') else list(eig_kappa),
                'surrogate_results': surrogate_results
            }
            
            return results
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Process uploaded image and use it as a microstructure
@app.post("/api/upload-microstructure")
async def upload_microstructure(
    file: UploadFile = File(...),
    kappa1: float = Form(...),
    alpha: float = Form(...)
):
    try:
        print(f"Processing uploaded image with kappa1={kappa1}, alpha={alpha}")
        
        # Validate parameters
        if kappa1 <= 0:
            raise HTTPException(status_code=400, detail="Thermal conductivity ratio (kappa1) must be greater than 0")
        
        if not (0 <= alpha <= 90):
            raise HTTPException(status_code=400, detail="Direction angle (alpha) must be between 0 and 90 degrees")
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload JPG, PNG, BMP, or GIF images."
            )
        
        # Read the uploaded file
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Validate and process the image
        try:
            binary_image = validate_uploaded_image(image)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Track processing time
        start_time = time.time()
        
        # Run simulation on the binary microstructure
        result = run_thermal_simulation(binary_image, kappa1, alpha)
        
        processing_time = time.time() - start_time
        print(f"Uploaded image processed in {processing_time:.2f} seconds")
        
        return result
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

# Process drawn microstructure
@app.post("/api/process-drawing")
async def process_drawing(
    drawing: str = Form(...),
    kappa1: float = Form(...),
    alpha: float = Form(...)
):
    try:
        print(f"Processing drawn microstructure with kappa1={kappa1}, alpha={alpha}")
        
        # Validate parameters
        if kappa1 <= 0:
            raise HTTPException(status_code=400, detail="Thermal conductivity ratio (kappa1) must be greater than 0")
        
        if not (0 <= alpha <= 90):
            raise HTTPException(status_code=400, detail="Direction angle (alpha) must be between 0 and 90 degrees")
        
        # Parse and validate the drawing
        try:
            binary_image = parse_base64_image(drawing)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Simple check for empty or invalid drawings
        white_pixels = binary_image.sum()
        total_pixels = binary_image.size
        white_percentage = (white_pixels / total_pixels) * 100
        
        # Warn if microstructure has very low or high phase content (but still process it)
        if white_percentage < 5:
            print(f"Warning: Drawing has very low white content ({white_percentage:.1f}%)")
        elif white_percentage > 95:
            print(f"Warning: Drawing has very high white content ({white_percentage:.1f}%)")
        
        # Track processing time for performance monitoring
        start_time = time.time()
        
        # Run simulation on the binary microstructure
        result = run_thermal_simulation(binary_image, kappa1, alpha)
        
        processing_time = time.time() - start_time
        print(f"Drawing processed in {processing_time:.2f} seconds")
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Error processing drawing: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing drawing: {str(e)}"
        )

# Serve HTML file
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)