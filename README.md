# XAIDS: XAI-Assisted Intrusion Detection System

---

## Requirements

- **Python 3.12** <p>
- If you encounter issues, try using **Python 3.12.8.**

## Get Started

### Setup
1. Create virtual environment <p>
`python3.12 -m venv .env`
2. Activate environment <p>
`source .env/bin/activate`
3. Download required packages <p>
`pip install -r requirements.txt`
4. Select *.env* as Kernel in Jupyter Notebook

### Download Dataset
Download CICIDS2017 Dataset:
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

### Store Dataset
Store the files from the CICIDS2017 Dataset in the folder `CICIDS2017/raw/`. It should look like that:
```
xai-assisted-intrusion-detection-system
│   README.md
│   ...   
│
└───CICIDS2017
│   └───raw
│       |   Friday-WorkingHours-Afternoon-DDos.pcap_ISCX
|       |   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX
|       |   Friday-WorkingHours-Morning.pcap_ISCX
|       |   ...
```

## Workflow & Organization 

1. Preprocess CICIDS2017 data: *data-preprocessing.ipynb*.
2. Split Data into *train* and *test* sets: *data-splitter.ipynb*.
3. Build DNN Intrusion Detection System: *intrusion-detection-system.ipynb*
4. Generate Adversarial Attacks: *attack-generator.ipynb*
5. Generate Explanations: *explainer.ipynb*
6. Detect Adversarial Attacks: *detector.ipynb*

**Important Note:** The same data splits must be used for both the Intrusion Detection System and adversarial attack generation to ensure consistency and comparability of results.