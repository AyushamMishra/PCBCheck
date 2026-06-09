# PCBCheck 🔍

**Production-grade PCB Defect Detection using YOLOv8**

A machine learning powered application for automated PCB defect detection using state-of-the-art computer vision. Detects 6 types of PCB defects with 98.1% mAP50 accuracy.

## Features

✨ **Real-time Detection** - Instantly identify PCB defects from uploaded images  
🎯 **High Accuracy** - 98.1% mAP50 on PCB defect dataset  
🚀 **Production Ready** - Deployed web interface with Streamlit  
🔧 **Multiple Defect Types** - Detects 6 different PCB defect categories  
⚡ **Fast Inference** - GPU-optimized with PyTorch  

## Supported Defect Types

The model detects the following PCB defects:

1. **Missing Hole** - Absent drilling/plating
2. **Mouse Bite** - Sharp corners in copper trace
3. **Open Circuit** - Broken copper connections
4. **Short** - Unintended electrical connections
5. **Spur** - Small copper extensions
6. **Spurious Copper** - Extra copper deposits

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AyushamMishra/PCBCheck.git
cd PCBCheck
```

2. Create a virtual environment (recommended):
```bash
python -m venv pcbcheck_env
source pcbcheck_env/bin/activate  # On Windows: pcbcheck_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit web interface:

```bash
streamlit run App/app.py
```

The application will open in your browser at `http://localhost:8501`

**Usage:**
1. Upload a PCB image (PNG, JPG, JPEG)
2. The model will automatically detect defects
3. View annotated results with defect count and confidence scores

## Project Structure

```
PCBCheck/
├── App/
│   └── app.py                 # Streamlit web application
├── model/
│   └── PCBCheck_best.pt      # Trained YOLOv8 model weights
├── docs/                      # Documentation files
├── predictions/               # Model prediction outputs
├── training_results/          # Training metrics and logs
├── pcb.yaml                   # Dataset configuration
├── requirements.txt           # Python dependencies
├── railway.json              # Deployment configuration
└── README.md                 # This file
```

## Dependencies

- **streamlit** (1.38.0) - Web UI framework
- **ultralytics** (8.4.21) - YOLOv8 framework
- **pillow** (10.4.0) - Image processing
- **opencv-python-headless** (4.10.0) - Computer vision
- **torch** (2.9.0) - PyTorch deep learning
- **torchvision** (0.20.0) - Vision utilities
- **numpy** (1.26.4) - Numerical computing
- **matplotlib** (3.8.0) - Visualization

## Dataset

The model is trained on the PCB Defect Dataset from Kaggle:
- **Source**: `/kaggle/input/datasets/norbertelter/pcb-defect-dataset/`
- **Train/Val/Test split**: Organized in separate image directories
- **Classes**: 6 defect types
- **Model Performance**: 98.1% mAP50

## Model Architecture

**Framework**: YOLOv8 (You Only Look Once v8)  
**Task**: Object Detection  
**Input**: PCB images (any standard resolution)  
**Output**: Bounding boxes with class labels and confidence scores  

## Performance Metrics

- **mAP50**: 98.1%
- **Inference Speed**: Real-time on GPU

## Deployment

The application is configured for deployment on Railway:
- See `railway.json` for deployment configuration
- Environment variables and build settings pre-configured

## Usage Examples

### Web Interface
Simply upload a PCB image through the Streamlit interface to get instant defect detection results with visual annotations.

### Custom Implementation
```python
from ultralytics import YOLO

# Load model
model = YOLO("model/PCBCheck_best.pt")

# Run inference
results = model("path/to/image.jpg", conf=0.5)

# Process results
for box in results[0].boxes:
    print(f"Defect: {box.cls}, Confidence: {box.conf:.2f}")
```

## Configuration

Edit `pcb.yaml` to modify dataset paths and class definitions:

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
nc: 6
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
```

## Development

### Training
To retrain the model:
```bash
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # Load pretrained model
results = model.train(data="pcb.yaml", epochs=100, imgsz=640)
```

### Results
Training results and predictions are saved to:
- `training_results/` - Training logs and metrics
- `predictions/` - Inference outputs

## Troubleshooting

**Model loading failed**: Ensure `model/PCBCheck_best.pt` exists  
**CUDA not available**: Install PyTorch with CUDA support or use CPU mode  
**Image upload fails**: Verify image format is PNG, JPG, or JPEG  

## Author

**Ayusham Mishra** - Production MLOps Demo  

## License

Please check the repository for license information.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests

## Acknowledgments

- YOLOv8 by Ultralytics
- PCB Defect Dataset from Kaggle
- Streamlit for the web framework

---

**Note**: This is a production MLOps demonstration project showcasing real-time PCB defect detection capabilities.
