# 👄 Lipread - Video to Text Conversion using Deep Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv)](https://opencv.org/)

A sophisticated deep learning application that performs **lip reading** - converting silent video footage of people speaking into text using advanced neural networks and computer vision techniques.

---

## 🌟 Features

- **3D Convolutional Neural Networks**: Utilizes 3D CNN layers to extract spatial and temporal features from video frames
- **Bidirectional LSTM Architecture**: Employs bidirectional LSTM layers for sequence-to-sequence learning
- **CTC Loss Function**: Implements Connectionist Temporal Classification for sequence alignment
- **Interactive GUI**: User-friendly Tkinter-based interface for easy video selection and processing
- **Real-time Processing**: Efficient video processing pipeline with frame extraction and normalization
- **Pre-trained Model**: Comes with trained model checkpoints ready for inference

---

## 🎯 How It Works

The system uses a multi-stage pipeline:

1. **Video Loading**: Extracts frames from input video files
2. **Preprocessing**: 
   - Converts frames to grayscale
   - Crops to lip region (190:236, 80:220)
   - Normalizes pixel values using mean and standard deviation
3. **Feature Extraction**: 3D CNN layers capture spatial-temporal patterns
4. **Sequence Learning**: Bidirectional LSTM layers model temporal dependencies
5. **Decoding**: CTC decoder converts predictions to readable text

---

## 🏗️ Architecture

```
Input Video (frames)
    ↓
Conv3D (128 filters) → ReLU → MaxPool3D
    ↓
Conv3D (256 filters) → ReLU → MaxPool3D
    ↓
Conv3D (75 filters) → ReLU → MaxPool3D
    ↓
TimeDistributed(Flatten)
    ↓
Bidirectional LSTM (128 units) → Dropout(0.5)
    ↓
Bidirectional LSTM (128 units) → Dropout(0.5)
    ↓
Dense (vocabulary_size + 1) → Softmax
    ↓
CTC Decoder → Output Text
```

---

## 📋 Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV
- NumPy
- Tkinter (usually comes with Python)
- Keras

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CSE439
   ```

2. **Install required dependencies**
   ```bash
   pip install tensorflow opencv-python numpy matplotlib keras
   ```

3. **Verify model checkpoint**
   Ensure the pre-trained model exists at:
   ```
   ./models/checkpoint
   ```

---

## 💻 Usage

### GUI Application

Run the graphical interface:

```bash
python gui.py
```

**Steps:**
1. Click **"Select Video File"** to choose a video (supports .mp4, .avi, .mkv, .wmv, .mpg)
2. Click **"Process Video"** to start lip reading
3. View the predicted text in the output window

### Jupyter Notebook

For experimentation and model training, use:

```bash
jupyter notebook main.ipynb
```

---

## 📁 Project Structure

```
CSE439/
├── gui.py              # Tkinter-based GUI application
├── main.ipynb          # Jupyter notebook for training/experimentation
├── models/
│   └── checkpoint      # Pre-trained model weights
└── README.md           # Project documentation
```

---

## 🔧 Configuration

### Vocabulary
The model supports the following character set:
```python
vocab = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
```

### Video Processing
- **Input dimensions**: Full frame
- **Lip region crop**: [190:236, 80:220]
- **Color space**: Grayscale
- **Normalization**: Z-score normalization (mean=0, std=1)

---

## 🧠 Model Details

### Hyperparameters
- **Optimizer**: Adam
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Dropout Rate**: 0.5
- **LSTM Units**: 128 (Bidirectional)
- **CNN Filters**: 128 → 256 → 75

### Training
The model is trained to predict text sequences from video frames using CTC loss, which allows for alignment-free sequence learning.

---

## 📊 Technical Specifications

| Component | Specification |
|-----------|--------------|
| Framework | TensorFlow/Keras |
| Model Type | 3D CNN + Bidirectional LSTM |
| Input Format | Video files (.mpg, .mp4, .avi, .mkv, .wmv) |
| Output Format | Plain text |
| Decoding | CTC Greedy Decoder |
| Sequence Length | 75 frames |

---

## 🎓 Use Cases

- **Accessibility**: Assist hearing-impaired individuals
- **Silent Communication**: Decode speech in noisy environments
- **Security**: Surveillance and forensic analysis
- **Media**: Automatic subtitling for silent footage
- **Research**: Speech recognition and computer vision studies

---

## ⚠️ Known Limitations

- Requires videos with clear lip movements
- Works best with frontal face views
- Fixed crop region may not suit all video formats
- Performance depends on video quality and lighting

---

## 🔮 Future Enhancements

- [ ] Multi-speaker support
- [ ] Real-time webcam processing
- [ ] Support for multiple languages
- [ ] Improved model architecture (Transformer-based)
- [ ] Data augmentation techniques
- [ ] Web-based interface
- [ ] API endpoint for integration

---

## 📚 References

- Deep learning for lip reading
- Connectionist Temporal Classification (CTC)
- 3D Convolutional Neural Networks
- Bidirectional LSTM for sequence modeling

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is part of CSE439 coursework.

---

## 🙏 Acknowledgments

Special thanks to the CSE439 course instructors and the open-source community for making this project possible.

---

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

