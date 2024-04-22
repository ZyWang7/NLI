# NLI
Train a model to perform Natural Language Inference task. Given a premise and a hypothesis, determine if the hypothesis is true based on the premise.


### Training Data
Randomly select 80% of the original data set ("train.csv") as the new training data set, and the remaining 20% as the test data set, and saved into new file: "train_data.csv" and "test_data.csv".

Data is available in: [Data](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/ziyi_wang-22_student_manchester_ac_uk/EuyPDqTDQjpImKZC6xDMAF4Bq3yP55MVQNVGDLKqQHD0UQ?e=zdH10J).


### Model
The model is fine-tuned using the pretrained roberta-base with LoRA.


### Files:
1. `nli_dataset.py`: the Dataset class used to load the data into PyTorch Dataset.
2. `lora_roberta.ipynb`: the code for set up the model, prepare the data and training.
3. `predict.ipynb`: the code for prediction and evaluation.



Trained weight is stored in the OneDrive: [roberta_lora_weight](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/ziyi_wang-22_student_manchester_ac_uk/ERKNvqve5pZEj2iPKG9XMykBKQg2ynxxJ_KyEaYklFA_Rg?e=yfb5MM).
