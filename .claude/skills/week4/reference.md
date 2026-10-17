# 4주차 평가(Evaluation) — 강의 요약 & 문제 은행

> 출처: `2026년_2학기_4주차_머신러닝_평가(수정).pdf` (40쪽). 이 파일의 정답은 채팅에 노출하지 말고 WEEK4.md에만 쓴다.

## A. 강의 요약

### 학습 목표
정확도(Accuracy), 오차행렬(Confusion Matrix), 정밀도(Precision), 재현율(Recall), F1 스코어, ROC AUC

### 정확도(Accuracy)
- 정확도 = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수 = (TN + TP) / (TN + FP + FN + TP)
- 모델 성능을 평가하는 대표 지표, 전체 데이터 중 맞게 예측한 비율
- 장점: 직관적, 이해 쉬움, **클래스 분포가 균형** 잡혀 있을 때 유용
- 단점: **불균형 데이터에 취약**(클래스 비율이 치우치면 성능 왜곡), 오탐/누락(어떤 오류인지) 정보 부족, 위험한 판단 유도
  - 예: Class 0 95개, Class 1 5개 → 무조건 0으로 예측해도 정확도 95%
  - 예: 정상 990, 암환자 10 → 모두 정상으로 예측 시 TN 990, FN 10, TP 0, FP 0 → 정확도 0.99(99%)지만 암환자를 하나도 못 찾음 (슬라이드는 "0.99%"로 표기 — 실제 의미는 0.99 = 99%)

### 오차행렬(Confusion Matrix)
- 분류 모델 성능을 자세히 분석. 정확도만으로 모르는 예측 결과의 세부 구조, **어떤 유형의 오류**가 발생하는지 보여줌
- 행 = 실제 클래스(Actual), 열 = 예측 클래스(Predicted). 0 = Negative, 1 = Positive
```
                예측 Negative(0)   예측 Positive(1)
실제 Negative(0)     TN                FP
실제 Positive(1)     FN                TP
```
- TN: 예측 0, 실제 0 / TP: 예측 1, 실제 1 / FN: 예측 0, 실제 1 / FP: 예측 1, 실제 0
- 앞 글자 T/F = 예측이 맞았는지, 뒤 글자 P/N = **예측한** 클래스
- 암 예시(Negative=정상, Positive=암환자): FP = 정상인에게 "암입니다", FN = 암환자에게 "정상입니다"
- sklearn `confusion_matrix(y_true, y_pred)` 출력 `[[TN, FP], [FN, TP]]` 예: `[[970 5] [10 15]]`

### 정밀도와 재현율
- **정밀도(Precision) = TP / (TP + FP)**: Positive로 **예측한 것** 중 실제 Positive 비율. "양성이라고 예측한 것 중 진짜 양성이 몇 %?" FP가 줄수록 증가
- **재현율(Recall) = 민감도(Sensitivity) = TPR = TP / (TP + FN)**: **실제 Positive** 중 Positive로 맞게 예측한 비율. "실제 양성 중 몇 %를 찾았나?" **FN**이 줄수록 증가 (슬라이드 9쪽은 'FP(실제 값 중 놓친거)'라고 오타 — 정확히는 FN)
- 예시 비교
  - 정상 990/암 10, 전부 정상 예측: accuracy 0.99, precision 0/0 = 0, recall 0/10 = 0
  - 정상 975/암 25, TN 970·FP 5·FN 10·TP 15: accuracy 985/1000 = 0.985, precision 15/20 = 0.75, recall 15/25 = 0.6
- sklearn: `from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix`
  - 입력: y_true(정답, 필수), y_pred(예측, 필수), labels, pos_label=1, **average**(기본 'binary'), zero_division
  - average: 이진분류 'binary'(기본), 다중분류 None(클래스별 배열), 'micro', 'macro', 'weighted'
    - micro: 전체 TP·FP·FN을 합쳐 계산(다중분류에서 정확도와 같음)
    - macro: 클래스별 값의 단순 평균
    - weighted: 클래스별 값을 실제 샘플 수(support)로 가중 평균
- 리스트로 정답/예측 만들기: `[0 for x in range(975)] + [1 for x in range(25)]` 등

### 붓꽃 다중분류 평가 (test 30개, random_state=11)
- 오차행렬 `[[9 0 0] [0 10 0] [0 2 9]]` (virginica 2개를 versicolor로 오분류), 정확도 0.9333
- 정밀도: None → [1. 0.8333 1.], micro 0.9333, macro 0.9444, weighted 0.9444
- 재현율: None → [1. 1. 0.8182], micro 0.9333, macro 0.9394, weighted 0.9333

### 정밀도/재현율 트레이드오프
- **정밀도가 더 중요**: 실제 Negative를 Positive로 잘못 판단(FP)하면 위험·비용이 큰 경우 → 스팸메일(정상 메일을 스팸 처리), 추천 시스템, 광고 타겟팅. "찾아낸 것 중 얼마나 맞는가?"
- **재현율이 더 중요**: 실제 Positive를 Negative로 놓치면(FN) 위험·비용이 큰 경우 → 암 등 의료 진단, 금융 사기 판별, 구조 요청 분석. "실제로 맞는 것을 얼마나 놓치지 않고 찾는가?"
- **분류 결정 임곗값(Threshold)**: 업무 특성상 정밀도/재현율 중 하나를 강조해야 할 때 조정. 기본 0.5
  - 임곗값을 **낮추면** Positive 예측이 늘어 **재현율↑, 정밀도↓**. 높이면 정밀도↑, 재현율↓
  - 예: Positive 확률 0.4 → 임곗값 0.5면 0, 임곗값 0.3이면 1

### 타이타닉 실습 (이진 분류)
- 전처리 함수: `fillna()` 결측치 대체(Age→평균, Cabin→'N', Embarked→'N', Fare→0), `drop_features()` 학습에 불필요한 PassengerId·Name·Ticket 삭제, `format_features()` Cabin 첫 글자만 추출 후 Cabin·Sex·Embarked를 **LabelEncoder**로 숫자 인코딩. `transform_features()`가 세 함수를 차례로 호출
- `y_titanic_df = titanic_df['Survived']`(타겟), `X_titanic_df = titanic_df.drop('Survived', axis=1)`(피처)
- train_test_split(test_size=0.2, random_state=11) → 테스트 179건
- 정확도: DecisionTreeClassifier 0.7877, RandomForestClassifier 0.8547, LogisticRegression(solver='liblinear') 0.8659
- **predict_proba()**: 각 클래스에 속할 확률 반환. shape (179, 2), 0열 = 클래스 0 확률, 1열 = 클래스 1 확률, 각 행 합 1. predict()는 더 큰 확률 쪽(=임곗값 0.5)으로 분류. 예: [0.449 0.551] → 1
- **Binarizer(threshold=값)**: `from sklearn.preprocessing import Binarizer`. 값이 threshold보다 **같거나 작으면 0, 크면 1**. `fit()` → `transform()` 또는 `fit_transform()`
  - 예: X=[[1,-1,2],[2,0,0],[0,1.1,1.2]], threshold=1.1 → [[0,0,1],[1,0,0],[0,0,1]]
  - 적용: `pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)` (Positive 열만, 2차원으로) → `Binarizer(threshold=custom_threshold).fit(pred_proba_1).transform(pred_proba_1)`
- 임곗값별 결과 (LogisticRegression)

| 임곗값 | 오차행렬 [[TN FP][FN TP]] | 정확도 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|---|
| 0.40 | [[97 21][11 50]] | 0.8212 | 0.7042 | 0.8197 | 0.7576 |
| 0.45 | [[105 13][13 48]] | 0.8547 | 0.7869 | 0.7869 | 0.7869 |
| 0.50 | [[108 10][14 47]] | 0.8659 | 0.8246 | 0.7705 | 0.7966 |
| 0.55 | [[111 7][16 45]] | 0.8715 | 0.8654 | 0.7377 | 0.7965 |
| 0.60 | [[113 5][17 44]] | 0.8771 | 0.8980 | 0.7213 | 0.8000 |
- **precision_recall_curve(y_test, pred_proba_class1)**: 임곗값별 정밀도·재현율 반환 → `precisions, recalls, thresholds`. 강의: thresholds shape (165,), precisions·recalls (166,) — 마지막에 정밀도 1·재현율 0 값이 하나 더 붙음. 입력은 정답과 **Positive 클래스 확률**
- 정밀도-재현율 곡선 그래프: x축 threshold, 정밀도는 점선(linestyle='--'), 임곗값이 커질수록 정밀도↑ 재현율↓, 두 곡선이 약 0.45~0.47 근처에서 교차

### F1-score
- 정밀도·재현율은 trade-off 관계 → 하나만 높아도 좋은 모델이 아님
- 정밀도와 재현율의 **조화 평균(Harmonic Mean)**: F1 = 2 / (1/recall + 1/precision) = **2 × (precision × recall) / (precision + recall)**
- 0~1, 클수록 좋음. 둘 다 높을 때 높음. 한쪽에 치우치면 낮아짐. 클래스 불균형 데이터에서 잘 동작
- `from sklearn.metrics import f1_score`, `f1_score(y_test, custom_predict)`

### ROC 곡선과 AUC
- **특이도(Specificity) = TN / (TN + FP)**: 실제 Negative 중 Negative로 맞게 예측한 비율. FP를 피하는 데 초점. 높을수록 불필요한 양성 오판정이 적음
- **FPR(False Positive Rate) = FP / (TN + FP) = 1 − 특이도**: 실제 Negative 중 Positive로 잘못 예측한 비율
- **TPR(True Positive Rate) = 재현율 = 민감도**
- **ROC 곡선**(Receiver Operation Characteristic): **FPR(x축)**이 변할 때 **TPR(y축)**의 변화를 보여주는 곡선. 임곗값을 바꿔 가며 그림
- **AUC(Area Under Curve)**: ROC 곡선 밑 면적. **1에 가까울수록 좋음**, 대각선(랜덤) = 0.5 → 실질적으로 최악. 왼쪽 위 (0, 1)이 최고의 모델
- `roc_curve(y_test, pred_proba_class1)` → `fprs, tprs, thresholds`. thresholds[0]은 max(예측확률)+1로 임의 설정되어 샘플 추출 시 index 1부터 시작
- `roc_auc_score(y_test, pred_proba)` — 입력은 정답과 **Positive 확률**(예측 클래스 아님). 강의 결과 0.8987
- 그래프: `plt.plot(fprs, tprs, label='ROC')`, 랜덤 대각선 `plt.plot([0,1],[0,1],'k--')`, xlabel 'FPR( 1 - Specificity )', ylabel 'TPR( Recall )'

## B. 문제 은행 (ID · 유형 · 문제 · 정답 · 해설)

### W4-01 · 공식
Q: 오차행렬의 TN, FP, FN, TP를 사용해 정확도 공식을 쓰시오.
A: (TN + TP) / (TN + FP + FN + TP)

### W4-02 · 개념
Q: 오차행렬에서 FP와 FN을 "예측값/실제값"으로 각각 설명하시오. 암 진단(Positive=암) 예시로 FN은 어떤 상황인가?
A: FP = 예측 1(Positive), 실제 0 / FN = 예측 0(Negative), 실제 1. FN은 실제 암환자를 정상이라고 진단한 상황
해설: 뒤 글자(P/N)는 예측값, 앞 글자(T/F)는 예측이 맞았는지.

### W4-03 · 개념
Q: sklearn의 confusion_matrix(y_true, y_pred)를 이진분류에 적용하면 2×2 배열의 각 위치에 무엇이 오는지 쓰시오.
A: [[TN, FP], [FN, TP]] — 행은 실제(0,1), 열은 예측(0,1)

### W4-04 · 계산
Q: 정상 990명, 암환자 10명인 데이터에서 모델이 모두 "정상"으로 예측했다. 정확도·정밀도·재현율은?
A: 정확도 990/1000 = 0.99(99%), 정밀도 0/0 → 0, 재현율 0/10 = 0
해설: 정확도만 보면 좋아 보이지만 암환자를 하나도 못 찾음 → 불균형 데이터에서 정확도의 한계.

### W4-05 · 개념
Q: 정확도 지표의 단점 3가지를 쓰시오.
A: ① 불균형 데이터에 취약(클래스 비율이 치우치면 성능 왜곡) ② 오탐/누락에 대한 정보 부족(어떤 종류의 오류인지 모름) ③ 위험한 판단 유도(성능이 좋아 보이지만 의미 없음)
해설: 장점은 직관적·이해 쉬움, 균형 잡힌 클래스 분포에 유용.

### W4-06 · 공식
Q: 정밀도와 재현율의 공식을 쓰고, 각각 어떤 오류 값이 줄어야 증가하는지 쓰시오.
A: 정밀도 = TP/(TP+FP), FP가 줄수록 증가 / 재현율 = TP/(TP+FN), FN이 줄수록 증가
해설: 슬라이드 9쪽 재현율 설명의 'FP'는 오타, 공식 기준으로 FN.

### W4-07 · 계산
Q: TN 970, FP 5, FN 10, TP 15일 때 정확도, 정밀도, 재현율을 구하시오.
A: 정확도 985/1000 = 0.985, 정밀도 15/20 = 0.75, 재현율 15/25 = 0.6
해설: 슬라이드 10·14쪽.

### W4-08 · 계산
Q: W4-07과 같은 오차행렬(TN 970, FP 5, FN 10, TP 15)의 F1-score와 특이도를 구하시오. (소수 셋째 자리까지)
A: F1 = 2×0.75×0.6/(0.75+0.6) = 0.9/1.35 ≈ 0.667, 특이도 = 970/975 ≈ 0.995
해설: FPR = 5/975 ≈ 0.005.

### W4-09 · 계산
Q: TN 50, FP 10, FN 20, TP 20일 때 정확도, 정밀도, 재현율, F1, FPR을 구하시오.
A: 정확도 70/100 = 0.7, 정밀도 20/30 ≈ 0.667, 재현율 20/40 = 0.5, F1 = 2×0.667×0.5/1.167 ≈ 0.571, FPR = 10/60 ≈ 0.167
해설: F1 = 2PR/(P+R).

### W4-10 · 개념
Q: 재현율(Recall)의 다른 이름 두 가지를 쓰시오.
A: 민감도(Sensitivity), TPR(True Positive Rate)
해설: ROC 곡선의 y축이 TPR.

### W4-11 · 상황 판단
Q: 다음 업무에서 정밀도와 재현율 중 무엇이 더 중요한가? (1) 스팸메일 필터 (2) 암 진단 (3) 금융 사기 탐지 (4) 광고 타겟팅
A: (1) 정밀도 (2) 재현율 (3) 재현율 (4) 정밀도
해설: FP 비용이 크면 정밀도(정상 메일을 스팸으로 보내면 안 됨), FN 비용이 크면 재현율(암·사기를 놓치면 안 됨).

### W4-12 · 개념
Q: 분류 결정 임곗값을 0.5에서 0.3으로 낮추면 정밀도와 재현율은 각각 어떻게 변하나? 이유는?
A: 재현율↑, 정밀도↓. Positive로 예측하는 기준이 낮아져 Positive 예측이 많아지므로 실제 Positive를 더 많이 찾지만(FN↓) 잘못된 Positive도 늘어남(FP↑)
해설: 예: Positive 확률 0.4인 샘플은 임곗값 0.5에서 0, 0.3에서 1.

### W4-13 · 코드
Q: precision_score, recall_score의 average 매개변수 기본값과, 다중분류에서 쓸 수 있는 값 4가지는?
A: 기본 'binary' / None, 'micro', 'macro', 'weighted'
해설: 다중분류에 'binary'를 쓰면 오류. None은 클래스별 값 배열 반환.

### W4-14 · 개념
Q: 다중분류 평가에서 average='macro'와 'weighted'의 차이를 설명하시오.
A: macro = 클래스별 지표의 단순 평균(클래스 크기 무시), weighted = 클래스별 지표를 각 클래스 실제 샘플 수(support)로 가중 평균
해설: micro = 전체 TP·FP·FN을 합쳐 계산 → 다중분류에서 정확도와 같음.

### W4-15 · 계산
Q: 붓꽃 오차행렬이 [[9,0,0],[0,10,0],[0,2,9]]일 때 클래스별 정밀도(average=None)와 macro 정밀도를 구하시오.
A: [1.0, 0.8333, 1.0], macro = (1 + 0.8333 + 1)/3 ≈ 0.9444
해설: 클래스 1 예측 열 = 10 + 2 = 12, 그중 맞은 것 10 → 10/12.

### W4-16 · 계산
Q: 같은 붓꽃 오차행렬 [[9,0,0],[0,10,0],[0,2,9]]에서 클래스별 재현율과 정확도를 구하시오.
A: 재현율 [1.0, 1.0, 9/11 ≈ 0.8182], 정확도 28/30 ≈ 0.9333
해설: 클래스 2 실제 행 = 0+2+9 = 11, 그중 맞은 것 9. micro 재현율도 0.9333.

### W4-17 · 개념
Q: predict_proba()는 무엇을 반환하나? 타이타닉 테스트 179건에 적용했을 때 결과 shape은?
A: 각 샘플이 각 클래스에 속할 확률(행 합 = 1). shape (179, 2) — 0열은 클래스 0 확률, 1열은 클래스 1(Positive) 확률
해설: predict()는 더 높은 확률의 클래스로 분류(임곗값 0.5와 동일).

### W4-18 · 코드 결과
Q: `Binarizer(threshold=1.1).fit_transform([[1,-1,2],[2,0,0],[0,1.1,1.2]])`의 결과는?
A: [[0,0,1],[1,0,0],[0,0,1]]
해설: threshold보다 **같거나 작으면 0**, 크면 1 → 1.1은 0, 1.2는 1.

### W4-19 · 코드 빈칸
Q: Positive 확률에 임곗값을 적용하는 코드를 완성하시오.
```python
pred_proba_1 = pred_proba[:, ( ① )].reshape(( ② ), 1)
binarizer = Binarizer(( ③ )=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.( ④ )(pred_proba_1)
```
A: ① 1 ② -1 ③ threshold ④ transform
해설: Positive(클래스 1) 확률 열만 추출, Binarizer는 2차원 입력이 필요하므로 reshape(-1, 1).

### W4-20 · 개념
Q: 타이타닉 데이터 전처리에서 fillna, drop_features, format_features 함수가 각각 하는 일은?
A: fillna: 결측치 대체(Age→평균, Cabin·Embarked→'N', Fare→0) / drop_features: 학습에 불필요한 PassengerId, Name, Ticket 삭제 / format_features: 문자열(object) 피처 Cabin·Sex·Embarked를 LabelEncoder로 숫자 인코딩
해설: transform_features가 세 함수를 차례로 호출.

### W4-21 · 데이터 해석
Q: 타이타닉 LogisticRegression에서 임곗값을 0.4 → 0.6으로 올렸을 때 정밀도와 재현율 값의 변화(강의 결과)는?
A: 정밀도 0.7042 → 0.8980 (증가), 재현율 0.8197 → 0.7213 (감소)
해설: 임곗값↑ → Positive 예측↓ → FP↓(정밀도↑), FN↑(재현율↓). 강의 임곗값 중 F1이 가장 높은 것은 0.6(0.8000).

### W4-22 · 코드
Q: precision_recall_curve()의 입력값 2가지와 반환값 3가지를 쓰시오.
A: 입력: 실제 정답(y_test), Positive 클래스 예측 확률(predict_proba(X_test)[:, 1]) / 반환: precisions, recalls, thresholds
해설: 강의에서 thresholds 165개, precisions·recalls는 166개(하나 더 많음).

### W4-23 · 공식
Q: F1-score의 정의와 공식을 쓰시오.
A: 정밀도와 재현율의 조화 평균. F1 = 2 / (1/recall + 1/precision) = 2 × precision × recall / (precision + recall)
해설: 0~1, 클수록 좋음. 둘 중 하나만 높으면 F1은 낮음.

### W4-24 · 계산
Q: 정밀도 0.9, 재현율 0.1인 모델과 정밀도 0.5, 재현율 0.5인 모델의 F1을 각각 구하고 어느 쪽이 높은지 쓰시오.
A: 첫째 2×0.09/1.0 = 0.18, 둘째 2×0.25/1.0 = 0.5 → 둘째(균형 잡힌 모델)가 높음
해설: 조화 평균은 한쪽에 치우치면 크게 낮아짐.

### W4-25 · 공식
Q: 특이도(Specificity)와 FPR의 공식을 쓰고 둘의 관계를 쓰시오.
A: 특이도 = TN/(TN+FP), FPR = FP/(TN+FP) = 1 − 특이도

### W4-26 · 개념
Q: ROC 곡선의 x축과 y축은 각각 무엇인가?
A: x축 FPR(1 − 특이도), y축 TPR(재현율, 민감도)
해설: 임곗값을 바꿔 가며 (FPR, TPR) 점을 이어 그림.

### W4-27 · 개념
Q: AUC는 무엇이며, 값이 1일 때와 0.5일 때 각각 의미는?
A: ROC 곡선 아래 면적. 1 = 완벽한 분류(좌상단 (0,1) 지점 통과), 0.5 = 대각선 = 랜덤 추측 수준(실질적으로 최악)
해설: 1에 가까울수록 좋은 모델.

### W4-28 · 코드
Q: roc_auc_score에 넣어야 하는 두 번째 인자는 예측 클래스(predict 결과)인가, 예측 확률인가? 코드를 쓰시오.
A: Positive 클래스의 예측 확률. `roc_auc_score(y_test, lr_clf.predict_proba(X_test)[:, 1])`
해설: 강의 결과 0.8987. roc_curve도 같은 입력을 사용.

### W4-29 · 개념
Q: roc_curve()의 반환값 3가지와, thresholds[0]의 특징은?
A: fprs, tprs, thresholds / thresholds[0]은 max(예측확률)+1로 임의 설정되어 샘플 추출 시 index 1부터 시작
해설: 슬라이드 37쪽.

### W4-30 · 코드
Q: ROC 그래프에서 랜덤 분류기를 나타내는 검은 점선 대각선을 그리는 코드를 쓰시오.
A: `plt.plot([0, 1], [0, 1], 'k--', label='Random')`
해설: 'k' = black, '--' = 점선(3주차 내용).

### W4-31 · O·X
Q: 클래스가 95:5로 불균형한 데이터에서는 정확도가 높으면 좋은 모델이라고 판단할 수 있다. (O/X)
A: X
해설: 다수 클래스로만 예측해도 정확도 95%. 정밀도·재현율·F1·AUC를 함께 봐야 함.
