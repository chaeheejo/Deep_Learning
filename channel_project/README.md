# channel project

#### 상수관로 진동 센서 데이터로 누수 유형을 분류하는 문제

<br/>

### sparse_categorical.ipynb
> sparse categorical 손실 함수를 이용
> 평가지표인 f1 score을 측정할 수 없었음

<br/>

### onehot_categorical.ipynb
> f1 score을 측정하기 위해 one hot encoding을 따로 처리해준 뒤 categorical 손실 함수를 이용
> overfitting 문제 발견
> dropout, batch 정규화

<br/>

### convolution.ipynb
> minmax scaling, L2 정규화 
> 각 주파수별 진동의 특징을 알아내기 위해 convolution 연산

<br/>

### data_analyze.ipynb
> 데이터의 분포를 파악하기 위해 유형별 데이터셋의 모양, 전체 데이터셋의 모양을 분석

<br/>

### conv_concat.ipynb
> 분석한 결과를 바탕으로 특징적인 구간을 5개로 나눠 5개의 모델을 생성함
> 5개의 모델 각각 convolution 연산을 진행한 뒤 concat하여 하나의 출력값으로 만듦

<br/>

### dense_concat.ipynb
> 구간을 더 세분하게 나눠 
