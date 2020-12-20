# ecg12lead
Experiments on the **PhysioNet/CinC 2020 Challenge 12-lead ECG Classification  Dataset**.

## How to run
    git clone https://github.com/shehanmunasinghe/ecg12lead
    cd ecg12lead

    # install project 
    pip install -e .
    pip install -r requirements.txt

## Imports
This project is setup as a package which means any file can be imported into any other file like so:

    from ecg12lead.datasets import PhysioNet2020Dataset


## Downloading Data

### Download Links
   
* [CPSC](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz )

* [CPSC-Extra](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz)

* [StPetersburg](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz)

* [PTB](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz)

* [PTB-XL](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_PTB-XL.tar.gz)
* [Georgia](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz)

### When using Colab

    DATASET_DIR = '/content/Data'
    !mkdir "{DATASET_DIR}"
    !cd "{DATASET_DIR}"

    # Authentication
    from google.colab import auth
    auth.authenticate_user()

    # List avaialble datasets
    !gsutil ls "gs://physionet-challenge-2020-12-lead-ecg-public/"  

    # CPSC
    !gsutil cp "gs://physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz"           . 
    !tar -xzf PhysioNetChallenge2020_Training_CPSC.tar.gz
    !rm PhysioNetChallenge2020_Training_CPSC.tar.gz
    !mv Training_WFDB CPSC

    # CPSC-Extra
    !gsutil cp "gs://physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz"              . 
    !tar -xzf PhysioNetChallenge2020_Training_2.tar.gz
    !rm PhysioNetChallenge2020_Training_2.tar.gz
    !mv Training_2 CPSC-Extra

    # StPetersburg
    !gsutil cp "gs://physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz"   . 
    !tar -xzf PhysioNetChallenge2020_Training_StPetersburg.tar.gz
    !rm PhysioNetChallenge2020_Training_StPetersburg.tar.gz
    !mv WFDB StPetersburg

    # PTB
    !gsutil cp "gs://physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz"            . 
    !tar -xzf PhysioNetChallenge2020_Training_PTB.tar.gz
    !rm PhysioNetChallenge2020_Training_PTB.tar.gz
    !mv WFDB PTB

    # PTB-XL
    !gsutil cp "gs://physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB-XL.tar.gz"    . 
    !tar -xzf PhysioNetChallenge2020_Training_PTB-XL.tar.gz
    !rm PhysioNetChallenge2020_Training_PTB-XL.tar.gz
    !mv WFDB PTB-XL

    # Georgia
    !gsutil cp "gs://physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz"              . 
    !tar -xzf PhysioNetChallenge2020_Training_E.tar.gz
    !rm PhysioNetChallenge2020_Training_E.tar.gz
    !mv WFDB Georgia

  