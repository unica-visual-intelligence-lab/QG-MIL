# QGMIL: Qwen3-Inspired Gated MIL

Official PyTorch implementation of **QGMIL**, a transformer-based Multiple Instance Learning (MIL) architecture proposed in our MICCAI submission.

This repository contains the **model definitions only**.  
Training scripts and the full experimental pipeline will be released after paper acceptance.

---

## Requirements

- Python >= 3.9  
- PyTorch >= 2.0  

Install dependencies:

    pip install torch

---

## Available Models

The repository includes the following variants used in the paper:

- qgmil (default, full model)  
- qgmil_elementwise  
- qgmil_no_gate  
- qgmil_layernorm  
- qgmil_no_qknorm  
- qgmil_minimal  
- qgmil_deep  
- qgmil_light  

---

## Usage

### Import

    from qgmil import get_qgmil_model
    import torch

### Instantiate Model

    model = get_qgmil_model(
        model_type="qgmil",
        in_dim=1024,
        embed_dim=512,
        num_classes=1
    )

### Forward Pass

Input shape:

    (batch_size, num_instances, in_dim)

Example:

    x = torch.randn(2, 500, 1024)
    output = model(x)
    logits = output["logits"]

Optional mask:

    mask = torch.ones(2, 500)
    output = model(x, mask=mask)

---

## Notes

- The model expects pre-extracted instance features.  
- All ablations described in the paper are accessible via configuration presets.  
- Full training and evaluation code will be released after acceptance.
