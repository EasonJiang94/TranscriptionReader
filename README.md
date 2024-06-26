# Transcription Reader
This project aims to extract key information, such as patients' age and what treatment they received, from a list of medical transcriptions. 

The dataset is downloaded on a [kaggle project](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download). 

## Envirorment
You can create the env via conda:
```
conda env create -f environment.yml
```

## Usage
First, go to src/
```
cd src/
```
```
python3 main.py -m ChatGPT -s 5
```
or 
```
python3 main.py -m llama2 -s 5
```
or 
```
python3 main.py -m basic -s 5
```
where you can choose models among ['basic', 'llama2', 'ChatGPT']

When you use basic model, it will go through the whole data anyway. 
I recommend the size for llama2 and ChatGPT not bigger than 20, it would either spending much time or costing a great amount of your money. 

You have to set your ChatGPT Key by following the instruction on the OpenAI's page. (https://platform.openai.com/docs/quickstart?context=python)
```
export OPENAI_API_KEY='your-api-key-here'
```

## Experiment
The experiment doc is [here](https://hackmd.io/@9fOPzjfpQ6-A61nMUwe76w/Hkws8dw1A)
