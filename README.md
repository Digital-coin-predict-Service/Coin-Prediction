# 가상화폐 예측 LSTM 모델
## 비트코인 및 알트코인 예측을 위한 모델
코로나 이후 사람들은 '월급 빼고 다 오른다', '월급 받아선 부자 못된다'라는 말을 많이 하곤 합니다.

이러한 말과 더불어 주식과 가상화폐에 대한 투자가 이전에 비해 더 활발해졌습니다.

개인 투자자들은 비트코인 투자를 그래프의 모양과 거래량을 보면서 눈치싸움을 하듯이 투자를 합니다.

이러한 과정을 사람이 하는 것이 아닌 인공지능이 예측을 할 수 있다면 불필요한 시간 낭비를 줄일 수 있습니다.

따라서 비트코인을 예측하고자 합니다.

앞으로 10분(10개)의 데이터를 예측합니다.

## 왜 LSTM?
![image](https://github.com/Digital-coin-predict-Service/Coin-Prediction/assets/112631585/87a8e12a-cbe9-4b05-a701-1a650fbb2002)
LSTM은 위 사진에 있는 C라는 상태가 있다.

이는 이전 셀로부터 전해지는 cell state라고 불린다.

이 cell state는 외부 입력이 직접 연산되지 않아 값이 잘 변하지 않는 성질이 있다.

이러한 이유로 LSTM은 장기 기억에 더 유리한 모델이다.

따라서 LSTM을 이용하여 가상화폐를 예측할 수 있다.
*******
## 결과
아래 그림은 test 한 결과이다.

본 validation과 test는 임의의 테스트 샘플에 대한 결과이다.

다른 샘플의 경우 더 큰 오류가 나타날 수 있다. 
![image](https://github.com/kwonsw812/SNR_Prediction_LSTM/assets/112631585/94dcb8bc-7e9a-42c4-af6e-b81ada152aa0)
![image](https://github.com/kwonsw812/SNR_Prediction_LSTM/assets/112631585/788d053b-1cee-4ab8-8457-f5ed053b5320)
*******
### 데이터 수집 및 전처리
1. 업비트 서버에서 1분봉 원화 데이터를 모두 받아온다.
2. 볼륨(거래량)로 가치(value)를 나누어 각 분의 가격을 구한다.
3. 비트코인의 경우, **가격이 매우 크므로 가격의 데이터를 10000으로 나눈다.** 만 원 단위 이하의 오차는 허용한다는 것이다.

   알트코인의 경우, 가격이 만 원을 넘지 않는 것도 많지만, 전처리 과정의 통일을 위해 10000으로 나누었다.
4. 가격에 **moving average를 적용한다.** 이 때 window size는 10으로 한다.

   10개의 가격을 평균을 내어 한 개의 가격을 도출한다. 이는 가격의 추세를 나타낸다.
6. 볼륨에 **minmax scaler를 적용한다.**
   
   가격에도 minmax scaler를 적용할 것인데, 하나의 feature가 다른 feature에 비해 큰 값을 가진다면 해당 feature에 대해 더 민감하게 학습할 수 있다.

   따라서 볼륨에 minmax scaler를 적용하였다.
7. 2개의 feature를 가지는 LSTM 인풋을 만든다. (첫 번째 feature를 가격, 두 번째 feature를 볼륨으로 설정)

   LSTM의 인풋의 형태는 (sample size, time step, features)이다.

   본 코드에서는 비트코인의 경우 time step = 50, feature = 2로 설정하였다.
   알트코인의 경우 time step = 100으로 설정하였다.
   
   time step은 lstm 한 번에 들어갈 인풋의 개수이다. 즉 예측을 위해 사용할 이전 데이터의 개수이다.
9. 각 time step마다 가격에 minmax scaler를 적용하였다.

   가격은 time step마다 minmax scaler를 적용한 이유는 가격의 변동은 절대적인 데이터가 중요한 것이 아닌, 상대적인 높낮이가 중요하다.

   상대적인 데이터의 흐름을 더 잘 파악하기 위해 각 time step마다 minmax scaler를 적용한다.

### LSTM 학습
1. LSTM 학습을 진행
    
    LSTM 레이어 2개, 각 레이어에 유닛 100개씩 설정한다.

    loss는 mse, optimizer는 adam을 사용하였다.
    
2. 학습한 모델에 대해 validation을 진행한다.

    학습한 데이터를 다시 모델에 넣어 실험한다.
    
    **296개의 샘플의 mean absolute error의 평균은 약 14704원이 나왔다.**
3. 학습한 모델에 대해 test를 진행한다.

    학습되지 않은 데이터를 모델에 넣어 실험한다.

   **956개의 샘플의 mean absolute error의 평균은 약 13864원이 나왔다.**

    validation과 test의 mean absolute error는 비트코인의 급하락, 급상승 시 달라질 수 있다.
*****************
### 추가 요구 사항
* 1분 뿐만 아니라 5분, 30분, 1시간, 하루 간격의 가격 예측을 할 수 있는 모델 개발
* feature에 랜덤한 요소를 추가하여 LSTM 모델이 하나의 input에 여러가지 output을 내놓을 수 있도록 개발
  * 더 정확한 예측을 위해 한 번의 예측이 아니라 여러번의 예측을 통해 사용자에게 앞으로 추이를 확률로 보여주기 위함이다.
