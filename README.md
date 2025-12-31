# Face Verification with Cancelable Biometrics & Fuzzy Commitment

This project implements a secure face verification system using the LFW dataset. It integrates advanced security layers to protect biometric templates.

## Features
- **Face Embedding**: Uses FaceNet (InceptionResnetV1) to generate 512-D embeddings.
- **Baseline Verification**: Standard cosine similarity matching.
- **Cancelable Biometrics**: BioHashing (Tokenized Random Mixing) for revocable templates.
- **Fuzzy Commitment**: Uses Reed-Solomon Error Correcting Codes to bind a cryptographic key to the biometric data.
- **Gradio UI**: Interactive interface for Enrollment, Authentication, and Evaluation.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/EllailiDaoud/LFW_Biometric_Project.git
cd LFW_Biometric_Project
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Create venv
python -m venv .venv

# Activate venv (Linux/Mac)
source .venv/bin/activate

# Activate venv (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages using the provided requirements file.

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook
Open the Jupyter Notebook to run the project.

```bash
jupyter notebook lfw_face_verification.ipynb
```

Run all cells to initialize the model, load the dataset, and launch the Gradio UI.

### Generating Full Notebook (Optional)
If you need to regenerate the notebook from the script:
```bash
python generate_full_notebook.py
```

## Dataset
The project is designed to work with the LFW (Labeled Faces in the Wild) dataset.
- The code automatically scans the `lfw_funneled` directory.
- If the dataset is not present, ensure `archive.zip` is in the root directory, or download/extract LFW into `lfw_funneled`.

## License
[MIT](LICENSE)
