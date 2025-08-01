# XAID: XAI-Assisted Intrusion Detection

---

## 🔧 Requirements

- **Python 3.12** <p>
- If you encounter issues, try using **Python 3.12.8.**

## 📌 Get Started

### Setup
1. Create virtual environment <p>
`python3.12 -m venv .env`
2. Activate environment <p>
`source .env/bin/activate`
3. Download required packages <p>
`pip install -r requirements.txt`
4. Select *.env* as Kernel in Jupyter Notebook

### Download Datasets
- Download CICIDS2017 Dataset:
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
- Download NSL-KDD Dataset:
https://www.kaggle.com/datasets/hassan06/nslkdd

### Store Dataset
Store the files from the downloaded Datasets in the folder `datasets/CICIDS2017/raw/` and `datasets/NSL-KDD/raw` respectively. It should look like that:
```
xai-assisted-intrusion-detection-system/
│── README.md
│...
│ ├── datasets/   
| │ ├── CICIDS2017/
| │ │ ├── raw/
| │ │ │ ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX
| │ │ │ ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX
| │ │ │ ├── Friday-WorkingHours-Morning.pcap_ISCX
| │ │ │ ├── ...
| │ ├── NSL-KDD/
| │ │ ├── raw/
| │ │ │ ├── KDDTest+.txt
| │ │ │ ├── KDDTrain+.txt
| │ │ │ ├── ...
```

## 📂 Organization & Workflow  

The source code of this project is organized into three main folders: **`functions/`**, **`notebooks/`**, and **`prototypes/`**, each serving a specific purpose.  
```
src/ 
│── functions/ # Core Python modules with reusable functions
│ ├── attack_generator.py
│ ├── data_preprocessing.py 
│ ├── detector.py
│ ├── explainer.py
│ ├── ...
│── notebooks/ # Jupyter notebooks for visualization and experimentation 
│ ├── attack-generator.ipynb
│ ├── data_preprocessing.ipynb  
│ ├── data_splitter.ipynb  
│ ├── ...
│── prototypes/ # Different prototype implementations using core functions 
│ ├── alpha.ipynb
│ ├── ...
```

This project follows a structured workflow to build a **xai-assisted intrusion detection system**. Below are the main steps:   

1. Preprocess CICIDS2017 data: *data-preprocessing.ipynb*.
2. Split Data into *train* and *test* sets: *data-splitter.ipynb*.
3. Build DNN Intrusion Detection System: *intrusion-detection-system.ipynb*
4. Generate Adversarial Attacks: *attack-generator.ipynb*
5. Generate Explanations: *explainer.ipynb*
6. Detect Adversarial Attacks: *detector.ipynb*
7. Visualize Findings: *visualizer.ipynb* <p>

**Important Note:** The same data splits must be used for both the Intrusion Detection System and adversarial attack generation to ensure consistency and comparability of results.

## 💡 Prototypes
Here are all available prototypes:
- [Prototype *alpha*](docs/Prototype%20-%20alpha.md): Simple binary adversarial detection for detecting *FGSM* adversarial attacks - *Success*
- [Prototype *beta*](docs/Prototype%20-%20beta.md): *C&W* attack detector - *Failed*
- [Prototype *delta*](docs/Prototype%20-%20delta.md): Misclassified samples from *C&W* attack detector - *Success*
- [Prototype *epsilon*](docs/Prototype%20-%20epsilon.md): Misclassified samples from *FGSM* attack detector - *Failed*
- [Prototype *iota*](docs/Prototype%20-%20iota.md): Misclassified samples detector on all White-Box attacks - *Success*
- [Prototype *kappa*](docs/Prototype%20-%20kappa.md): Adversarial Attack and misclassified samples detector on all White-Box attacks - *Success*
- [Prototype *my*](docs/Prototype%20-%20my.md): Use whole dataset for the evaluation of the previous detector - *Success*
- [Prototype *ny*](docs/Prototype%20-%20ny.md): Evaluation of detector on new *NSL-KDD* dataset - *Success*
- [Prototype *omikron*](docs/Prototype%20-%20omikron.md): Include Black-Box Attacks - *Success*
- [Prototype *pi*](docs/Prototype%20-%20pi.md): Robust Classification on White-Box Attacks - *Success*
