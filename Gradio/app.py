# app.py
# Gradio App for UNet Pet Segmentation with 4 Model Variants

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import io

# ---------------------------------------
# MODEL ARCHITECTURE (Copy from training code)
# ---------------------------------------

class ContractingBlockMP(nn.Module):
    """MaxPool version"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        skip = x
        x = self.maxpool(x)
        return x, skip


class ContractingBlockStrided(nn.Module):
    """Strided Conv instead of pool"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        skip = x
        x = self.down(x)
        return x, skip


class ExpandingBlockTranspose(nn.Module):
    """Transposed Convolution upsampling"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class ExpandingBlockUpsample(nn.Module):
    """Nearest neighbor upsampling + convs"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, contracting, expanding, in_channels=3, out_channels=1):
        super().__init__()
        self.contract1 = contracting(in_channels, 64)
        self.contract2 = contracting(64, 128)
        self.contract3 = contracting(128, 256)

        self.expand1 = expanding(in_ch=256, skip_ch=256, out_ch=256)
        self.expand2 = expanding(in_ch=256, skip_ch=128, out_ch=128)
        self.expand3 = expanding(in_ch=128, skip_ch=64,  out_ch=64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)

        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        return torch.sigmoid(self.final_conv(x))


# ---------------------------------------
# MODEL LOADING
# ---------------------------------------

MODEL_CONFIGS = {
    "MaxPool + TransConv + BCE": {
        "contracting": ContractingBlockMP,
        "expanding": ExpandingBlockTranspose,
        "checkpoint": "MP_Tr_BCE.pth",
        "description": "Standard U-Net with MaxPooling and Transposed Convolutions"
    },
    "MaxPool + TransConv + Dice": {
        "contracting": ContractingBlockMP,
        "expanding": ExpandingBlockTranspose,
        "checkpoint": "MP_Tr_Dice.pth",
        "description": "MaxPooling + TransConv trained with Dice Loss"
    },
    "Strided Conv + TransConv + BCE": {
        "contracting": ContractingBlockStrided,
        "expanding": ExpandingBlockTranspose,
        "checkpoint": "StrConv_Tr_BCE.pth",
        "description": "Strided Convolutions for downsampling with BCE Loss"
    },
    "Strided Conv + Upsample + Dice": {
        "contracting": ContractingBlockStrided,
        "expanding": ExpandingBlockUpsample,
        "checkpoint": "StrConv_Ups_Dice.pth",
        "description": "Strided Convs + Nearest Neighbor Upsampling with Dice Loss"
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load all models
models = {}
for name, config in MODEL_CONFIGS.items():
    try:
        model = UNet(config["contracting"], config["expanding"]).to(device)
        model.load_state_dict(torch.load(config["checkpoint"], map_location=device))
        model.eval()
        models[name] = model
        print(f"✓ Loaded: {name}")
    except FileNotFoundError:
        print(f"⚠ Warning: {config['checkpoint']} not found. Please train this model first.")
    except Exception as e:
        print(f"⚠ Error loading {name}: {e}")


# ---------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------

def preprocess_image(image, size=128):
    """Preprocess uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def create_overlay(image_np, mask_np, alpha=0.5):
    """Create overlay visualization of mask on image"""
    # Create colored mask (green for pet)
    colored_mask = np.zeros((*mask_np.shape, 3))
    colored_mask[mask_np > 0.5] = [0, 1, 0]  # Green for pet
    
    # Blend with original image
    overlay = image_np * (1 - alpha) + colored_mask * alpha
    return np.clip(overlay, 0, 1)


def predict_segmentation(image, model_name, show_overlay, confidence_threshold):
    """Main prediction function"""
    if image is None:
        return None, None, None, "Please upload an image first!"
    
    if model_name not in models:
        return None, None, None, f"⚠ Model '{model_name}' not available. Please train it first."
    
    # Preprocess
    img_pil = Image.fromarray(image).convert('RGB')
    img_tensor = preprocess_image(img_pil).to(device)
    
    # Predict
    model = models[model_name]
    with torch.no_grad():
        pred = model(img_tensor).cpu().squeeze().numpy()
    
    # Apply confidence threshold
    pred_binary = (pred > confidence_threshold).astype(np.float32)
    
    # Prepare outputs
    img_np = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Create visualizations
    mask_vis = (pred * 255).astype(np.uint8)  # Raw probabilities
    binary_mask_vis = (pred_binary * 255).astype(np.uint8)  # Binary mask
    
    # Create overlay if requested
    overlay_vis = None
    if show_overlay:
        overlay = create_overlay(img_np, pred_binary, alpha=0.4)
        overlay_vis = (overlay * 255).astype(np.uint8)
    
    # Calculate metrics
    pet_percentage = (pred_binary.sum() / pred_binary.size) * 100
    confidence_avg = pred[pred_binary > 0.5].mean() if pred_binary.sum() > 0 else 0
    
    info_text = f"""
    ### 📊 Prediction Results
    
    **Model:** {model_name}
    
    **Metrics:**
    - Pet Coverage: {pet_percentage:.2f}% of image
    - Average Confidence: {confidence_avg:.3f}
    - Threshold Used: {confidence_threshold}
    
    **Model Info:**
    {MODEL_CONFIGS[model_name]['description']}
    """
    
    return binary_mask_vis, mask_vis, overlay_vis, info_text


def compare_all_models(image, confidence_threshold):
    """Compare predictions from all available models"""
    if image is None:
        return None, "Please upload an image first!"
    
    available_models = [name for name in MODEL_CONFIGS.keys() if name in models]
    
    if len(available_models) == 0:
        return None, "No models available. Please train models first."
    
    # Preprocess
    img_pil = Image.fromarray(image).convert('RGB')
    img_tensor = preprocess_image(img_pil).to(device)
    img_np = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Create comparison figure
    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4*(n_models+1), 4))
    
    # Plot original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Plot predictions from each model
    for idx, model_name in enumerate(available_models):
        model = models[model_name]
        with torch.no_grad():
            pred = model(img_tensor).cpu().squeeze().numpy()
        
        pred_binary = (pred > confidence_threshold).astype(np.float32)
        
        axes[idx + 1].imshow(pred_binary, cmap='gray')
        axes[idx + 1].set_title(model_name, fontsize=8, fontweight='bold')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    comparison_img = Image.open(buf)
    plt.close()
    
    info_text = f"Compared {len(available_models)} models at threshold {confidence_threshold}"
    
    return comparison_img, info_text


# ---------------------------------------
# GRADIO INTERFACE
# ---------------------------------------

def create_interface():
    with gr.Blocks(title="🐾 Pet Segmentation with U-Net") as demo:
        gr.Markdown(
            """
            # 🐾 Pet Segmentation using U-Net
            
            Upload a pet image and segment it using one of 4 trained U-Net variants!
            
            ### 📚 Available Models:
            1. **MaxPool + TransConv + BCE** - Standard U-Net architecture
            2. **MaxPool + TransConv + Dice** - Optimized with Dice loss
            3. **Strided Conv + TransConv + BCE** - Alternative downsampling
            4. **Strided Conv + Upsample + Dice** - Nearest neighbor upsampling
            """
        )
        
        with gr.Tab("🎯 Single Model Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload Pet Image", type="numpy")
                    
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value=list(MODEL_CONFIGS.keys())[0] if MODEL_CONFIGS else None,
                        label="Select Model",
                        info="Choose which U-Net variant to use"
                    )
                    
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Adjust sensitivity (higher = stricter)"
                    )
                    
                    show_overlay_check = gr.Checkbox(
                        value=True,
                        label="Show Overlay",
                        info="Overlay mask on original image"
                    )
                    
                    predict_btn = gr.Button("🚀 Predict Segmentation", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Row():
                        output_binary = gr.Image(label="Binary Mask", type="numpy")
                        output_prob = gr.Image(label="Probability Map", type="numpy")
                    
                    output_overlay = gr.Image(label="Overlay Visualization", type="numpy")
                    
                    output_info = gr.Markdown()
            
            predict_btn.click(
                fn=predict_segmentation,
                inputs=[input_image, model_dropdown, show_overlay_check, confidence_slider],
                outputs=[output_binary, output_prob, output_overlay, output_info]
            )
            
            # Examples section - only in first tab
            gr.Examples(
                examples=[
                    ["example_pet1.jpg", "MaxPool + TransConv + BCE", True, 0.5],
                    ["example_pet2.jpg", "Strided Conv + Upsample + Dice", True, 0.5],
                ],
                inputs=[input_image, model_dropdown, show_overlay_check, confidence_slider],
                outputs=[output_binary, output_prob, output_overlay, output_info],
                fn=predict_segmentation,
                cache_examples=False,
            )
        
        with gr.Tab("🔬 Compare All Models"):
            with gr.Row():
                with gr.Column(scale=1):
                    compare_input = gr.Image(label="Upload Pet Image", type="numpy")
                    
                    compare_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    
                    compare_btn = gr.Button("🔍 Compare All Models", variant="primary")
                
                with gr.Column(scale=2):
                    compare_output = gr.Image(label="Model Comparison")
                    compare_info = gr.Markdown()
            
            compare_btn.click(
                fn=compare_all_models,
                inputs=[compare_input, compare_threshold],
                outputs=[compare_output, compare_info]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About This App
                
                This app demonstrates **U-Net architecture** for semantic segmentation on the Oxford-IIIT Pet Dataset.
                
                ### 🏗️ Architecture Variants:
                
                **Downsampling Methods:**
                - **MaxPooling**: Traditional 2x2 max pooling
                - **Strided Convolution**: 3x3 conv with stride=2
                
                **Upsampling Methods:**
                - **Transposed Convolution**: Learnable upsampling
                - **Nearest Neighbor + Conv**: Fixed upsampling + convolution
                
                **Loss Functions:**
                - **BCE (Binary Cross-Entropy)**: Pixel-wise classification
                - **Dice Loss**: Optimizes overlap directly
                
                ### 📊 How to Use:
                1. Upload a pet image (cat or dog)
                2. Select a model variant
                3. Adjust confidence threshold if needed
                4. Click "Predict" to see the segmentation
                
                ### 🎨 Output Explanation:
                - **Binary Mask**: Clean black/white segmentation
                - **Probability Map**: Model confidence (darker = less confident)
                - **Overlay**: Green highlights show detected pet
                
                ---
                
                **Note:** Models must be trained first and `.pth` files should be in the same directory.
                """
            )
    
    return demo


# ---------------------------------------
# LAUNCH APP
# ---------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("🐾 PET SEGMENTATION APP")
    print("="*60)
    print(f"Device: {device}")
    print(f"Models loaded: {len(models)}/{len(MODEL_CONFIGS)}")
    print("="*60)
    
    demo = create_interface()
    demo.launch(
        share=False,         # Removed (not needed on HF Spaces)
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
