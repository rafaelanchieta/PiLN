## PiLN team at IberLEF 2021
This repository contains the trained models for the Irony Detection task in Portuguese at IberLEF 2021

## Requirements

- Python 3 (or later)
- `pip install -r requirements.txt`
- `sh download.sh`

## Usage
### Pre-trained embeddings news

`python irony_detection -m embeddings -c news -t <sentence>`

### Pre-trained wmbeddings tweets

`python irony_detection -m embeddings -c twitter -t <sentence>`

### Superficial features news

`python irony_detection -m superficial -c news -t <sentence>`

### Superficial features tweets

`python irony_detection -m superficial -c twitter -t <sentence>`

## Reference




