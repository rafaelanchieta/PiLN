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
```
@inproceedings{anchieta-et-al-21,
    author = "Anchi{\^e}ta, Rafael and Neto, Assis and Marinho, Jeziel and Moura, Raimundo",
    title = "{P}i{LN} {IDPT} 2021: {I}rony {D}etection in {P}ortuguese {T}exts with {S}uperficial {F}eatures and {E}mbeddings",
    booktitle = "Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2021)
co-located with the Conference of the Spanish Society for Natural Language Processing (SEPLN 2021)",
    month = sep,
    year = "2021",
    address = "Málaga, Spain",
    pages = "917--924",
}
```



