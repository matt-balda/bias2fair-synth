import pandas as pd
import numpy as np
from utils.data_loader import load_compas
from sdv.metadata import SingleTableMetadata
from utils.diffusion_wrapper import TabDDPMWrapper
import torch

def test_diffusion():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    data = load_compas().head(500) # Small sample
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    wrapper = TabDDPMWrapper(metadata, device='cuda', steps=10, batch_size=256)
    print("Fitting TabDDPMWrapper...")
    wrapper.fit(data)
    
    print("Sampling from TabDDPMWrapper...")
    synth = wrapper.sample(10)
    print("Synthetic Data Sample:")
    print(synth.head())
    print("Test passed!")

if __name__ == "__main__":
    test_diffusion()
