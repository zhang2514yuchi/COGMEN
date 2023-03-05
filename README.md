## COGMEN; Official Pytorch Implementation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cogmen-contextualized-gnn-based-multimodal/multimodal-emotion-recognition-on-iemocap)](https://paperswithcode.com/sota/multimodal-emotion-recognition-on-iemocap?p=cogmen-contextualized-gnn-based-multimodal)

**CO**ntextualized **G**NN based **M**ultimodal **E**motion recognitio**N**
![Teaser image](./COGMEN_architecture.png)


#### Used for my Graduation Project


## Requirements

- We use PyG (PyTorch Geometric) for the GNN component in our architecture. [RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv) and [TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)

- We use [comet](https://comet.ml) for logging all our experiments and its Bayesian optimizer for hyperparameter tuning. 

- For textual features we use [SBERT](https://www.sbert.net/).
### Installations
- [Install PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

- [Install Comet.ml](https://www.comet.ml/docs/python-sdk/advanced/)
- [Install SBERT](https://www.sbert.net/)


## Preparing datasets for training

        python preprocess.py --dataset="iemocap_4"

## Training networks 

        python train.py --dataset="iemocap_4" --modalities="atv" --from_begin --epochs=55

## Run Evaluation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1biIvonBdJWo2TiYyTiQkxZ_V88JEXa_d?usp=sharing)

        python eval.py --dataset="iemocap_4" --modalities="atv"
