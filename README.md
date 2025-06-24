## MoBLLM

This is the repository for the paper "A Foundational Individual Mobility Prediction Model based on Open-Source Large Language Models".

## Data Preparation
We provide processed datasets used in the experiments at the following link: 
https://drive.google.com/file/d/1TYH288GSErSIBQM8rR6qR6IBytVGznBV/view?usp=drive_link

Download and unzip the data in the main folder, then finish the data preparation.

## Model Training
This section includes the training of MoBLLM and the deep learning baseline models.
### MoBLLM Fine-Tuning
We provide the MoBLLM training realized by LoRA and its several advanced variants including OLoRA, EVA, PiSSA, LoftQ, LoRA+, rsLoRA and QLoRA.
For example, the MoBLLM is fine tuned by OLoRA method:
```python
python llmtrain_olora.py
```



