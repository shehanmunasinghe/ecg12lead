# ecg12lead
An extension of https://github.com/shehanmunasinghe/diagno

## How to run
    git clone https://github.com/shehanmunasinghe/ecg12lead
    cd ecg12lead

    # install project 
    pip install -e .
    pip install -r requirements.txt

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

    from ecg12lead.datasets import PhysioNet2020Dataset