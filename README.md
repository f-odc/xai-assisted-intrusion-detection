# XAIDS: XAI-Assisted Intrusion Detection System

---

## Requirements

Python 3.12.8

## Get Started

### Setup
1. Create virtual environment <p>
`python3.12.8 -m venv .env`
2. Activate environment <p>
`source .env/bin/activate`
3. Download required packages <p>
`pip install -r requirements.txt`
4. Select *.env* as Kernel in Jupyter Notebook

### Download Dataset
Download CICIDS2017 Dataset:
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

### Store Dataset
Store the CICIDS2017 Dataset in the folder `data/`. It should look like that:
```
xai-assisted-intrusion-detection-system
│   README.md
│   ...   
│
└───data
│   └───CICIDS2017
│       |   Friday-WorkingHours-Afternoon-DDos.pcap_ISCX
|       |   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX
|       |   Friday-WorkingHours-Morning.pcap_ISCX
|       |   ...
```

### Perform Preprocessing

Run file *data-preprocessing.ipynb* to preprocess the CICIDS2017 data.