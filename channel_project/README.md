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
> 구간을 더 세분하게 나눠 학습을 진행함
> 16개의 데이터 구간을 convolution 연산하고, dense layer를 추가해 구간의 특징만을 사용함

<br/>

### reduce_numOf_model.ipynb
> 8개의 구간으로 데이터 구간을 줄이고, dense layer의 노드 수를 64로 증가해 학습시킴
> 과적합은 일어나지 않으나 epoch이 300이어도 최종 loss 값이 0.5에 그침

<br/>

### dense_concat_eightAndEntire.ipynb 
> 최종 8개의 구간에 대해 convolution layer 대신 dense layer로 진행함
> 8개의 구간과 더불어 전체 데이터셋을 입력으로 한 모델을 추가시켰더니 과적합을 더 방지할 수 있었음
