# channel project

#### 상수관로 진동 센서 데이터로 누수 유형을 분류하는 문제

<br/>

### sparse_categorical.ipynb
> sparse categorical 손실 함수를 이용 <br/>
> 평가지표인 f1 score을 측정할 수 없었음 <br/>
> 최종 train loss 0.2890 validation loss 0.4517 epoch 500

<br/>

### onehot_categorical.ipynb
> f1 score을 측정하기 위해 one hot encoding을 따로 처리해준 뒤 categorical 손실 함수를 이용 <br/>
> overfitting 문제 발견 <br/>
> dropout, batch 정규화 <br/>
> 최종 train f1 score 98 validation f1 score 80 epoch 300

<br/>

### convolution.ipynb
> minmax scaling, L2 정규화  <br/>
> 각 주파수별 진동의 특징을 알아내기 위해 convolution 연산 <br/>
> 최종 train f1 score 45 validation f1 score 47 epoch 300

<br/>

### data_analyze.ipynb
> 데이터의 분포를 파악하기 위해 유형별 데이터셋의 모양, 전체 데이터셋의 모양을 분석

<br/>

### conv_concat.ipynb
> 분석한 결과를 바탕으로 특징적인 구간을 5개로 나눠 5개의 모델을 생성함 <br/>
> 5개의 모델 각각 convolution 연산을 진행한 뒤 concat하여 하나의 출력값으로 만듦 <br/>
> 최종 train f1 score 84 validation f1 score 81 epoch 100

<br/>

### dense_concat.ipynb
> 구간을 더 세분하게 나눠 학습을 진행함 <br/>
> 16개의 데이터 구간을 convolution 연산하고, dense layer를 추가해 구간의 특징만을 사용함  <br/>
> 최종 train f1 score 45 validation f1 score 39 epoch 100

<br/>

### reduce_numOf_model.ipynb
> 8개의 구간으로 데이터 구간을 줄이고, dense layer의 노드 수를 64로 증가해 학습시킴 <br/>
> 과적합은 일어나지 않으나 epoch이 300이어도 최종 loss 값이 0.5에 그침 <br/>
> 최종 train f1 score 55 validation f1 score 52 epoch 300

<br/>

### dense_concat_eightAndEntire.ipynb 
> 최종 8개의 구간에 대해 convolution layer 대신 dense layer로 진행함 <br/>
> 8개의 구간과 더불어 전체 데이터셋을 입력으로 한 모델을 추가시켰더니 과적합을 더 방지할 수 있었음 <br/>
> 최종 train f1 score 65 validation f1 score 63 epoch 100

<br/>

### conv_concat_eightAndEntire.ipynb
> 8개의 구간과 전체 데이터셋을 기반으로 모델을 생성해 학습을 진행 <br/>
> convolution layer를 추가할 때 처음 layer의 노드 개수를 64, 다음 layer를 128로 설정하는게 학습률이 더 좋음 <br/>
> concat 후 dense layer의 노드 개수는 2048로 촘촘하게 설정함 <br/>
> 최종 train f1 score 93 validation f1 score 87 epoch 200

<br/>

### set_numOf_convNode.ipynb
> 두개의 convolution layer 중 두번째 layer보다 첫번째 layer를 더 촘촘히 구성하는 것이 학습률이 좋았음 <br/>
> dropout 비율은 첫번째 layer에 대한 dropout을 0.8, 두번째 layer에 대한 dropout을 0.6으로 진행하는 것이 가장 성능이 좋았음 <br/>
> concat 후 dense layer은 첫번째 dense를 촘촘하게 구성한 뒤 dropout 비율을 0.8로 잡고, 두번째 dense를 널널하게 가져갔음 <br/>
> 최종 train f1 score 95 validation f1 score 84 epoch 200
