# BERT2BERT model for Korean legal case summarization

## Requirements

```
pip install -r requirements.txt
```

<br/>

## Usage


추가적인 fine-tuning을 위해 `mybert2bert.ipynb`를, 해당 모델을 이용한 test 문장 생성을 위해 `mybert2bert_test.ipynb`를 참조하세요.

<br/>


## Description

- datamodule: 모델의 input과 output으로 들어가는 데이터를 전처리합니다. 

    - input: Each case text
    - output: Issue of each case
    
    <br/>

- lightningbase: pytorch lightning의 lightningmodule을 상속해서 optimizer와 save_model을 정의합니다. 

- my_model: lightningbase를 상속해서 Bert2Bert의 특화된 부분을 정의합니다. 

- train: 실제로 실행하는 main 부분이고 hyper parameter들을 정의합니다. 