# Efficient Global Content Recovery for Multi-View Data

Identifying shared content across multiple views using sparse Contrastive Representation Learning (CRL).

This project has been submitted in partial fulfillment of the requirements for COMP 588: Probabilistic Graphical Models at the University of McGill.
The final report can be found [here](./docs/report/main.pdf).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Numerical Experiment

Run the synthetic numerical experiment to verify causal factor independence:

```bash
python scripts/main_numerical.py
```

### Multimodal Experiment

To run the multimodal experiment, first prepare the [Multimodal3DIdent](https://github.com/imantdaunhawer/Multimodal3DIdent) dataset and set the path in the command below.

Run the multimodal (Image-Text) experiment:

```bash
python scripts/main_multimodal.py --data-root /path/to/multimodal_data
```

### Acknowledgements

This project is built upon the works of

- Daunhawer et al. (2023): <https://github.com/imantdaunhawer/multimodal-contrastive-learning>
- Yao et al. (2024): <https://github.com/CausalLearningAI/multiview-crl>
