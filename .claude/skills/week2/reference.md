# 2주차 Scikit-Learn — 강의 요약 & 문제 은행

> 출처: `2026년_2학기_2주차_머신러닝_Scikit-Learn.pdf` (31쪽). 이 파일의 정답은 채팅에 노출하지 말고 WEEK2.md에만 쓴다.

## A. 강의 요약

### 학습 목표
Scikit-Learn 소개 및 설치 / 주요 모듈 / 사용 방법 및 순서 / 실습: 붓꽃(Iris) 데이터셋으로 품종 예측

### Scikit-Learn 소개
- 파이썬 기반 머신러닝 라이브러리. 다양한 알고리즘과 편리한 프레임워크·API 제공
- 간단하고 효율적으로 **데이터 분석, 예측 모델 구축**을 돕는 도구. 오랜 기간 실전 환경에서 검증됨
- 주로 **Numpy, Scipy, Matplotlib** 기반 위에서 구축
- 장점: 간편한 사용, 풍부한 알고리즘, 강력한 문서화, Numpy·Pandas 연동 용이, 테스트 샘플 데이터 제공(MNIST, 보스턴 주택가격, 붓꽃 등)
- 설치: Anaconda Prompt 실행 → 가상환경 활성화 → `conda install scikit-learn`
- 버전 확인: `import sklearn; print(sklearn.__version__)` (강의 화면: 1.0.2)

### 기계학습 알고리즘 적용 분야
| 분야 | 설명 |
|---|---|
| 분류(Classification) | SVM, k-NN, 랜덤 포레스트 등 |
| 회귀(Regression) | 선형 회귀, 릿지 회귀 등 |
| 클러스터링(Clustering) | K-Means, DBSCAN 등 비지도 학습 기반 군집화 |
| 차원 축소(Dimensionality Reduction) | PCA, t-SNE 등 고차원 데이터 시각화 및 압축 |
| 모델 선택(Model Selection) | 교차 검증, Grid Search 등 하이퍼파라미터 튜닝 |
| 전처리(Preprocessing) | 정규화, 스케일링, 인코딩 등 |

### Estimator 구조
- **Estimator**: 학습 `fit()`, 예측 `predict()`
  - **Classifier(분류)**: DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, GaussianNB, SVC
  - **Regressor(회귀)**: LinearRegression(슬라이드 표기 LinerRegression), Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor

### 주요 모듈
| 분류 | 모듈 | 설명 |
|---|---|---|
| 예제 데이터 | sklearn.datasets | 내장 예제 데이터 세트 |
| 데이터 분리·검증·파라미터 튜닝 | sklearn.model_selection | 학습/테스트 분리, 교차 검증, Grid Search |
| 피처 처리 | sklearn.preprocessing | 인코딩(문자열→숫자), 정규화, 스케일링 |
| 피처 처리 | sklearn.feature_selection | 영향이 큰 피처를 우선순위대로 선택 |
| 피처 처리 | sklearn.feature_extraction | 텍스트/이미지 벡터화 피처 추출 (CountVectorizer, Tf-Idf Vectorizer). text / image 하위 모듈 |
| 피처 처리 & 차원 축소 | sklearn.decomposition | PCA, NMF, Truncated SVD |
| 평가 | sklearn.metrics | Accuracy, Precision, Recall, ROC-AUC, RMSE 등 |
| ML 알고리즘 | sklearn.ensemble | 랜덤 포레스트, 에이다 부스트, 그래디언트 부스팅 |
| ML 알고리즘 | sklearn.linear_model | 선형 회귀, 릿지, 라쏘, 로지스틱 회귀, SGD |
| ML 알고리즘 | sklearn.naive_bayes | 가우시안 NB, 다항 분포 NB |
| ML 알고리즘 | sklearn.neighbors | K-NN |
| ML 알고리즘 | sklearn.svm | 서포트 벡터 머신 |
| ML 알고리즘 | sklearn.tree | 의사 결정 트리 |
| ML 알고리즘 | sklearn.cluster | K-평균, 계층형, DBSCAN (비지도) |
| 유틸리티 | sklearn.pipeline | 피처 처리 변환 + 학습·예측을 묶어서 실행 |

### 내장 예제 데이터셋
| API | 설명 | 용도 |
|---|---|---|
| load_boston() | 보스턴 집 피처·가격 | 회귀 |
| load_breast_cancer() | 위스콘신 유방암, 악성/음성 레이블 | 분류 |
| load_diabetes() | 당뇨 | 회귀 |
| load_digits() | 0~9 숫자 이미지 픽셀 | 분류 |
| load_iris() | 붓꽃 피처 | 분류 |
(참고: load_boston은 최신 버전(1.2+)에서 제거됨 — 시험은 슬라이드 기준)

### 사용 방법 및 순서 (머신러닝 프로세스)
1. **데이터 수집**: 문제 정의, 데이터 수집(공개 데이터, 웹 크롤링, 로그). 수집 데이터(10) = 학습 데이터(7) + 검증 데이터(3)
2. **데이터 전처리(Preprocessing)**: 학습에 적용할 형태로 변형, 불필요 요소·중복 제거, 누락 확인. "데이터 정제"라고도 함
3. **모델 선택(Model Selection)**: 학습 방법·알고리즘 선택. 지도/비지도/강화학습
4. **학습(Training)**: 학습 데이터(Training Data) 사용
5. **평가·검증(Evaluation)**: 검증 데이터(Test Data) 사용, 성능 검증. 결과에 따라 알고리즘·조건 바꿔 다시 학습(이전 단계로 피드백) → 학습 완료

코드 관점 순서: 데이터 로드·준비(학습/검증 각각 입력값·출력값) → 모델 객체 생성 및 `fit()`(학습 데이터 입력) → `predict()`(검증 데이터 입력) → `metrics.accuracy_score()`로 예측값과 검증 정답 비교

### 특성(Feature)과 타겟(Target)
- **특성(Feature)**: 학습 모델에 주어지는 입력값(Input data), 데이터의 속성·변수, **독립변수(Independent Variable)**, 타겟 예측의 기준 정보 → "문제"
- **타겟/레이블/클래스/정답(Target/Label/Class)**: 모델이 예측하려는 결과값(Output data), **종속변수(Dependent Variable)**. 문제 종류에 따라 분류/회귀 → "정답"
- 예시: 키(cm)·몸무게(kg) = 특성, 비만 여부(예=1/아니오=0) = 타겟(레이블)

### 붓꽃(Iris) 데이터셋
- 1936년 통계학자 **Ronald A. Fisher**(1890~1962, 영국)가 논문에서 사용. 꽃잎·꽃받침 크기 측정
- 머신러닝·통계학의 대표적 입문용 분류 데이터셋. (슬라이드: 프랑스의 국화)
- 샘플 **150개**, 클래스 **3개**(setosa=0, versicolor=1, virginica=2, 각 50개), 특성 **4개**, **다중 클래스 분류**
- 특성: Sepal Length(꽃받침 길이), Sepal Width(꽃받침 너비), Petal Length(꽃잎 길이), Petal Width(꽃잎 너비), 단위 cm
- CSV(Comma-Separated Values) 형태 원본도 있음: 입력(Feature) 4열 + 출력(Target_name) 1열

### 결정 트리(Decision Tree) 개념
- 데이터의 **규칙을 학습**으로 자동으로 찾아 **트리 기반 분류 규칙**을 만드는 것
- 지도 학습 모델, 분류와 회귀 모두 적용. 결정트리/의사결정나무라고도 부름
- **1986년 로스 퀸란(Ross Quinlan)** 발표 (ID3)
- 관측값과 목표값을 연결하는 예측 모델. 여러 결정(Decision)을 통해 **이진 트리** 구성. '스무고개'처럼 진행
- **분기 속성 선정의 중요성**: 동물 예시(Milk, Feather, Aquatic, Egg, Legs, Size, Color → Whale, Lion, Frog, Pigeon, Lobster)에서 어떤 속성으로 먼저 나누느냐에 따라 트리 깊이·복잡도가 달라짐. Color로 먼저 나누면 Whale/Lion/Frog가 바로 분리되고 Grey만 Feather로 한 번 더 나누면 됨 → 가장 얕음

### 실습 코드 (붓꽃 품종 예측)
```python
from sklearn.datasets import load_iris              # 붓꽃 데이터셋
from sklearn.tree import DecisionTreeClassifier     # 결정트리 분류 모델
from sklearn.model_selection import train_test_split # 학습/테스트 분리
import pandas as pd

iris = load_iris()
iris_data = iris.data        # 특징(피처) 데이터, numpy
iris_label = iris.target     # 레이블(정답) 데이터, numpy  [0 ... 1 ... 2 ...]
print(iris.target_names)     # ['setosa' 'versicolor' 'virginica']
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head(3)

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                    test_size=0.2, random_state=11)
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))   # 0.9333
```
- iris 객체 속성: `data`(특징=문제), `target`(타겟 숫자=정답), `feature_names`(특징 이름), `target_names`(레이블 이름)
- 반환 순서: **X_train, X_test, y_train, y_test** (X=특징/문제, y=타겟/정답)
- test_size=0.2 → 학습:테스트 = 8:2 → 150개 중 **120 : 30**
- train_test_split 기본값: test_size=None(→ 0.25), train_size=None, random_state=None, **shuffle=True**, stratify=None
- random_state: 난수 시드 고정 → 실행할 때마다 같은 분할/결과
- 정확도 0.9333 = 30개 중 28개 맞춤 (예측 1, 정답 2인 것 2건)
- 학습된 모델 활용 프로그램: 사용자에게 SL, SW, PL, PW 4개 값을 입력받아 `dt_clf.predict([[SL, SW, PL, PW]])` 결과를 품종명으로 출력, 0 입력 시 종료

## B. 문제 은행 (ID · 유형 · 문제 · 정답 · 해설)

### W2-01 · 개념
Q: Scikit-Learn은 주로 어떤 세 라이브러리를 기반으로 구축되었나?
A: Numpy, Scipy, Matplotlib
해설: 슬라이드 3쪽. Pandas는 "연동이 쉽다"는 장점으로 언급되지 기반 라이브러리는 아님.

### W2-02 · 개념
Q: Scikit-Learn의 장점을 3가지 이상 쓰시오.
A: 간편한 사용, 풍부한 알고리즘, 강력한 문서화, Numpy·Pandas 연동 용이, 테스트용 샘플 데이터 제공(MNIST, 보스턴 주택가격, 붓꽃 등)
해설: 5개 중 3개 이상이면 O.

### W2-03 · 코드 빈칸
Q: Anaconda Prompt에서 Scikit-Learn을 설치하는 명령어를 쓰시오.
A: `conda install scikit-learn`
해설: 설치 전 패키지를 설치할 가상환경을 활성화. (pip이면 `pip install scikit-learn`)

### W2-04 · 객관식
Q: 다음 중 "교차 검증, Grid Search 등 하이퍼파라미터 튜닝 도구"에 해당하는 적용 분야는? ① 전처리 ② 모델 선택 ③ 차원 축소 ④ 클러스터링
A: ② 모델 선택(Model Selection)
해설: 전처리=정규화·스케일링·인코딩, 차원 축소=PCA·t-SNE, 클러스터링=K-Means·DBSCAN.

### W2-05 · 개념
Q: Scikit-Learn에서 모든 학습 모델(Estimator)이 공통으로 갖는 학습 메서드와 예측 메서드 이름은?
A: 학습 `fit()`, 예측 `predict()`
해설: Estimator는 Classifier(분류)와 Regressor(회귀)로 나뉨.

### W2-06 · 분류
Q: 다음 클래스를 분류(Classifier)와 회귀(Regressor)로 나누시오: DecisionTreeClassifier, Ridge, GaussianNB, Lasso, SVC, LinearRegression, GradientBoostingRegressor
A: 분류 — DecisionTreeClassifier, GaussianNB, SVC / 회귀 — Ridge, Lasso, LinearRegression, GradientBoostingRegressor
해설: 슬라이드 6쪽 구현 클래스 목록.

### W2-07 · 모듈 매칭
Q: 다음 기능을 제공하는 sklearn 모듈명을 쓰시오. (1) 학습/테스트 데이터 분리, 그리드 서치 (2) 문자열을 숫자형 코드로 인코딩, 정규화, 스케일링 (3) Accuracy, Precision, Recall, ROC-AUC 등 성능 측정 (4) 랜덤 포레스트, 에이다 부스트, 그래디언트 부스팅
A: (1) sklearn.model_selection (2) sklearn.preprocessing (3) sklearn.metrics (4) sklearn.ensemble
해설: 슬라이드 7~8쪽 모듈 표.

### W2-08 · 모듈 매칭
Q: PCA, NMF, Truncated SVD 같은 차원 축소 알고리즘을 지원하는 모듈과, 피처 처리 변환과 학습·예측을 묶어 실행하는 유틸리티 모듈은?
A: sklearn.decomposition / sklearn.pipeline
해설: 텍스트 CountVectorizer·Tf-Idf는 sklearn.feature_extraction(.text).

### W2-09 · 모듈 매칭
Q: 로지스틱 회귀(LogisticRegression)와 SGD 관련 알고리즘은 어느 모듈에 있나?
A: sklearn.linear_model
해설: 이름은 "회귀"지만 로지스틱 회귀는 분류 알고리즘이며 linear_model에 속함.

### W2-10 · 매칭
Q: 내장 데이터셋 load_boston, load_breast_cancer, load_diabetes, load_digits, load_iris의 학습 용도를 각각 분류/회귀로 쓰시오.
A: boston 회귀, breast_cancer 분류, diabetes 회귀, digits 분류, iris 분류
해설: 슬라이드 9쪽.

### W2-11 · 순서 배열
Q: 머신러닝(Scikit-Learn) 사용 순서 5단계를 순서대로 쓰시오. (데이터 수집 포함)
A: 데이터 수집 → 데이터 전처리 → 모델 선택 → 학습 → 평가·검증 (→ 학습 완료)
해설: 평가 결과에 따라 이전 단계로 돌아가 다시 학습하는 피드백 구조.

### W2-12 · 개념
Q: 수집된 데이터를 학습에 적용하기 위한 형태로 변형하고, 불필요한 요소·중복 제거, 누락 여부 확인 등을 하는 단계의 이름과 다른 이름은?
A: 데이터 전처리(Preprocessing), 다른 이름은 데이터 정제
해설: 슬라이드 10쪽.

### W2-13 · 빈칸
Q: 강의에서 수집된 데이터 10을 학습 데이터 ( ① )과 검증 데이터 ( ② )으로 나눈다고 했다.
A: ① 7 ② 3
해설: 7:3 비율. 실습 코드는 test_size=0.2로 8:2.

### W2-14 · 개념
Q: 특성(Feature)과 타겟(Target)을 각각 독립변수/종속변수, 입력/출력과 연결하여 설명하시오.
A: 특성 = 모델에 주어지는 입력값(Input), 독립변수, "문제" / 타겟(레이블·클래스·정답) = 모델이 예측하는 출력값(Output), 종속변수, "정답"
해설: 비만 예시에서 키·몸무게는 특성, 비만 여부(0/1)는 타겟.

### W2-15 · 개념
Q: 붓꽃 데이터셋의 샘플 수, 클래스 수, 특성 수, 문제 유형을 쓰시오.
A: 150개, 3개, 4개, 다중 클래스 분류
해설: 각 클래스 50개씩.

### W2-16 · 매칭
Q: 붓꽃 레이블 0, 1, 2에 해당하는 품종명을 쓰시오.
A: 0 = Setosa, 1 = Versicolor, 2 = Virginica
해설: `iris.target_names` → ['setosa' 'versicolor' 'virginica'].

### W2-17 · 개념
Q: 붓꽃 데이터셋의 4가지 특성(Feature)을 영어와 한글로 쓰시오.
A: Sepal Length(꽃받침 길이), Sepal Width(꽃받침 너비), Petal Length(꽃잎 길이), Petal Width(꽃잎 너비), 단위 cm
해설: Sepal=꽃받침, Petal=꽃잎 구분이 자주 출제됨.

### W2-18 · 인물
Q: 붓꽃 데이터셋을 1936년 논문에서 사용한 통계학자와, 1986년 의사결정트리를 발표한 사람은?
A: Ronald A. Fisher / Ross Quinlan(로스 퀸란)
해설: 슬라이드 14, 19쪽.

### W2-19 · 개념
Q: 결정 트리(Decision Tree)를 설명하시오. (학습 방식, 구조 포함)
A: 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 지도 학습 모델. 여러 결정(조건)을 통해 이진 트리를 구성하며 '스무고개'처럼 진행. 분류와 회귀에 모두 쓰임
해설: 슬라이드 18~19쪽.

### W2-20 · 개념
Q: 동물 분류 예시(Milk, Feather, Aquatic, Egg, Legs, Size, Color)에서 강조한 결정 트리의 핵심 포인트는? 가장 간단한 트리를 만든 첫 분기 속성은?
A: 분기 속성 선정의 중요성 — 어떤 속성으로 먼저 분기하느냐에 따라 트리의 깊이·복잡도가 달라짐. Color로 먼저 나눈 트리가 가장 단순(Grey만 Feather로 한 번 더 분기)
해설: Feather 먼저 → Aquatic → Color로 깊어짐, Size 먼저 → Feather → Color.

### W2-21 · 코드 빈칸
Q: 빈칸을 채우시오.
```python
from sklearn.datasets import ( ① )
from sklearn.tree import ( ② )
from sklearn.model_selection import ( ③ )
```
A: ① load_iris ② DecisionTreeClassifier ③ train_test_split
해설: 슬라이드 21쪽.

### W2-22 · 코드
Q: `iris = load_iris()`에서 특징 데이터, 타겟 숫자 값, 특징 이름, 레이블 이름을 가져오는 속성 4개를 쓰시오.
A: iris.data, iris.target, iris.feature_names, iris.target_names
해설: data=문제, target=정답.

### W2-23 · 코드 빈칸
Q: 빈칸을 채우시오. `( ① ), ( ② ), ( ③ ), ( ④ ) = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)`
A: ① X_train ② X_test ③ y_train ④ y_test
해설: 순서 주의 — X 두 개 먼저, y 두 개 나중. X=특징(문제), y=타겟(정답).

### W2-24 · 계산
Q: 150개 붓꽃 데이터에 `test_size=0.2`를 적용하면 학습/테스트 데이터는 각각 몇 개인가? test_size를 지정하지 않으면 기본 테스트 비율은?
A: 120개 / 30개. 기본값은 0.25(25%)
해설: 슬라이드에는 test_size=None으로 표시되지만, train_size도 None이면 0.25가 적용됨.

### W2-25 · 개념
Q: train_test_split의 `random_state`와 `shuffle`의 역할과 shuffle의 기본값은?
A: random_state: 난수 시드를 고정해 실행할 때마다 같은 결과로 분할 / shuffle: 분할 전 데이터를 섞을지 여부, 기본값 True
해설: stratify는 기본 None(레이블 비율 유지 분할 옵션).

### W2-26 · 코드 빈칸
Q: 결정트리 모델 객체를 만들고 학습·예측하는 코드를 완성하시오.
```python
dt_clf = ( ① )(random_state=11)
dt_clf.( ② )(X_train, y_train)
pred = dt_clf.( ③ )(X_test)
```
A: ① DecisionTreeClassifier ② fit ③ predict
해설: fit에는 학습용 특징+정답, predict에는 테스트용 특징만 들어감.

### W2-27 · O·X
Q: predict() 메서드에는 테스트용 특징 데이터(X_test)와 테스트용 정답(y_test)을 함께 넣는다. (O/X)
A: X
해설: predict()에는 X_test만 입력. y_test는 accuracy_score에서 예측값과 비교할 때 사용.

### W2-28 · 코드·계산
Q: 예측 정확도를 계산하는 함수 이름과 모듈, 인자 순서를 쓰시오. 테스트 30개 중 28개를 맞혔다면 결과값은(소수 4자리)?
A: `from sklearn.metrics import accuracy_score`, `accuracy_score(y_test, pred)` (정답, 예측 순), 28/30 = 0.9333
해설: 강의 결과 0.9333.

### W2-29 · 코드
Q: 붓꽃 데이터를 DataFrame으로 바꾸고 정답 열 'label'을 추가하는 코드 두 줄을 쓰시오.
A: `iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)` / `iris_df['label'] = iris.target`
해설: 확인은 `iris_df.head(3)`.

### W2-30 · 응용
Q: 학습된 모델 dt_clf로 꽃받침 길이 5.1, 꽃받침 너비 2.5, 꽃잎 길이 6.2, 꽃잎 너비 2.2인 붓꽃 하나의 품종을 예측하는 코드를 쓰시오.
A: `dt_clf.predict([[5.1, 2.5, 6.2, 2.2]])` (2차원 형태로 입력) → 결과 숫자를 target_names로 변환 (강의 결과 Virginica)
해설: predict는 2차원 배열(샘플 수 × 특성 수)을 받음.
