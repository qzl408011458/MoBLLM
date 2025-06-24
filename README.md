## MoBLLM

This is the repository for the paper "A Foundational Individual Mobility Prediction Model based on Open-Source Large Language Models".

## Data Preparation
We provide processed datasets used in the experiments at the following link: 
https://drive.google.com/file/d/1TYH288GSErSIBQM8rR6qR6IBytVGznBV/view?usp=drive_link

Download and unzip the data in the main folder, then finish the data preparation.

(note that the relationship between the abbr. in dataname and corresponding dataset, i.e.,

fsq: Foursquare New York; fsq_tky: Foursquare Tokyo; fsq_global: Foursquare global; geolife: Microsoft Geolife;

ori_hk, dest_hk: Hong Kong metro trip data of origin and destination; ori_hz, dest_hz: Hangzhou metro trip data of origin and destination;

data with network_change, event, incident or intervention involves in the names are the Hong Kong metro trip data of the four special scenarios.
)

## Model Training
This section includes the training of MoBLLM and the deep learning baseline models.
### MoBLLM Fine-Tuning
We provide the MoBLLM training realized by LoRA and its several advanced variants including OLoRA, EVA, PiSSA, LoftQ, LoRA+, rsLoRA and QLoRA.
For example, the MoBLLM fine tuned by OLoRA method can be executed by the following command:
```python
python llmtrain_olora.py
```

### Deep Learning Baseline Training
We provide the training of SOTA deep learning models DeepMove, MHSA, MobTCast in individual mobility prediciton. 
For example, training DeepMove on HK-ORI dataset (predicting next trip origin) can be executed by the following command:
```python
python train_dm.py -data hk -task ori
```

## Model Evaluation
This section includes the evaluation of MoBLLM, LLM-Mob and the deep learning baselines DeepMove, MHSA, MobTCast for individual mobility prediction under normal and special conditions.

The datasets with normal condition include fsq, fsq_tky, fsq_global, geolife, ori_hk, dest_hk, ori_hz and dest_hz. The others are with special condtions.

### Evaluation under Normal Condition
Evaluating the MoBLLM (fine tuned by oLoRA) on the fsq dataset is as follows:
```python
python run_mobllm-poi.py --data fsq --ft_path {fine-tuned model path} --ft_name olora
```

Evaluating the LLM-Mob on the HK-ORI dataset is as follows:
```python
python run_llm-mob-ori.py --data ori_hk
```

Evaluation of deep learning models under normal condition is also conducted in their trainig, see the Section of **Deep Learning Baseline Training**

### Evaluation under Special Condition
Evaluating the MoBLLM (fine tuned by oLoRA) on the ori_network_change dataset is as follows:
```python
python run_mobllm-ori.py --data ori_hk_network_change --ft_path {fine-tuned model path} --ft_name olora
```

Evaluating the LLM-Mob on the dest_network_change dataset is as follows:
```python
python run_llm-mob-dest.py --data dest_hk_network_change
```

Evaluation of deep learning models under special conditions needs well training on the HK metro data and then can be executed by the three .py files with _dleval_ (dleval_dm.py, dleval_mhsa.py, dleval_mob.py) 
with small modifications (e.g., the trained checkpoint path and the option of specific evaluation dataset)






