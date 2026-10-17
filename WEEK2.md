# 2주차 퀴즈 기록 — Scikit-Learn

### W2-11. 머신러닝 사용 순서 5단계 — 2026-10-17
- **유형**: 순서 배열
- **문제**: 머신러닝(Scikit-Learn) 사용 순서 5단계를 순서대로 쓰시오. (첫 단계는 "데이터 수집")
- **내 답변**: 데이터 수집, 데이터 전처리, 모델 선택, 모델 학습, 평가 및 검증
- **판정**: O
- **정답**: 데이터 수집 → 데이터 전처리(Preprocessing) → 모델 선택(Model Selection) → 학습(Training) → 평가·검증(Evaluation) → 학습 완료
- **해설**: 슬라이드 10쪽. 평가 결과가 좋지 않으면 이전 단계(수집·전처리·모델 선택·학습)로 돌아가 알고리즘이나 조건을 바꿔 다시 학습하는 피드백 구조. 전처리는 "데이터 정제"라고도 하며, 수집 데이터(10) = 학습 데이터(7) + 검증 데이터(3).

### W2-23. train_test_split 반환 순서 — 2026-10-17
- **유형**: 코드 빈칸
- **문제**: 빈칸을 채우시오. `( ① ), ( ② ), ( ③ ), ( ④ ) = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)`
- **내 답변**: X_train, y_train, X_test, y_test
- **판정**: X
- **정답**: ① X_train ② X_test ③ y_train ④ y_test
- **해설**: 반환 순서는 **X 두 개(학습, 테스트) 먼저, y 두 개(학습, 테스트) 나중**이다. 입력한 배열 순서(iris_data → iris_label)대로 각각 train/test 쌍이 나온다고 기억하면 된다. X = 특징(문제), y = 타겟(정답). 순서를 바꿔 쓰면 X_test에 정답 배열이 들어가는 식으로 변수가 뒤섞여 fit()에서 오류가 나거나 엉뚱하게 학습된다. (슬라이드 24쪽)

### W2-15. 붓꽃 데이터셋 기본 정보 — 2026-10-17
- **유형**: 개념
- **문제**: 붓꽃(Iris) 데이터셋의 샘플 수, 클래스 수, 특성 수, 문제 유형을 쓰시오.
- **내 답변**: 150, 3, 4, ?
- **판정**: △
- **정답**: 샘플 150개, 클래스 3개(setosa, versicolor, virginica), 특성 4개, 문제 유형 = **다중 클래스 분류**
- **해설**: 숫자 3개는 맞았고 문제 유형을 빠뜨림. 클래스가 2개면 이진 분류(예: 유방암 0/1), 3개 이상이면 다중 클래스 분류. 클래스마다 50개씩이고, 특성 4개는 sepal length·sepal width·petal length·petal width(cm). (슬라이드 14쪽)

### W2-07. 기능별 sklearn 모듈 — 2026-10-17
- **유형**: 모듈 매칭
- **문제**: 다음 기능을 제공하는 sklearn 모듈명을 쓰시오. (1) 학습/테스트 데이터 분리, 그리드 서치 (2) 문자열을 숫자형 코드로 인코딩, 정규화, 스케일링 (3) Accuracy, Precision, Recall, ROC-AUC 등 성능 측정 (4) 랜덤 포레스트, 에이다 부스트, 그래디언트 부스팅
- **내 답변**: 1: sklearn.model_selection / 2: sklearn.preprocessing / 3: sklearn.metrics / 4: sklearn.ensemble
- **판정**: O
- **정답**: (1) sklearn.model_selection (2) sklearn.preprocessing (3) sklearn.metrics (4) sklearn.ensemble
- **해설**: 슬라이드 7~8쪽 모듈 표. 같이 외워 둘 것: 차원 축소(PCA·NMF·Truncated SVD) = sklearn.decomposition, 텍스트·이미지 피처 추출(CountVectorizer·Tf-Idf) = sklearn.feature_extraction, 중요 피처 선택 = sklearn.feature_selection, 변환+학습 묶기 = sklearn.pipeline, 로지스틱 회귀·SGD = sklearn.linear_model.

### W2-27. predict()의 입력값 — 2026-10-17
- **유형**: O·X
- **문제**: predict() 메서드에는 테스트용 특징 데이터(X_test)와 테스트용 정답(y_test)을 함께 넣는다. (O/X)
- **내 답변**: X
- **판정**: O
- **정답**: X
- **해설**: predict()에는 특징 데이터 X_test만 넣는다(`pred = dt_clf.predict(X_test)`). 정답 y_test는 예측 후 `accuracy_score(y_test, pred)`로 비교할 때 사용. 반면 fit()에는 특징과 정답을 함께 넣는다(`fit(X_train, y_train)`). (슬라이드 25~26, 29쪽)

### W2-24. test_size 분할 개수와 기본값 — 2026-10-17
- **유형**: 계산
- **문제**: 150개 붓꽃 데이터에 `test_size=0.2`를 적용하면 학습/테스트 데이터는 각각 몇 개인가? test_size를 지정하지 않으면 기본 테스트 비율은?
- **내 답변**: (120, 30)개, 기본 테스트 비율은 0.3
- **판정**: △
- **정답**: 학습 120개, 테스트 30개 / 기본 테스트 비율 **0.25**(25%)
- **해설**: 개수 계산은 맞음(150 × 0.2 = 30). 기본값이 틀림 — train_test_split은 test_size와 train_size가 모두 None이면 0.25를 사용한다(슬라이드 시그니처에는 test_size=None으로 보임). 0.3은 강의에서 "학습 7 : 검증 3"으로 설명한 일반적인 분할 비율과 헷갈린 것으로 보임. 기본값 정리: test_size=None(→0.25), shuffle=True, random_state=None, stratify=None.

### W2-14. 특성(Feature)과 타겟(Target) — 2026-10-17
- **유형**: 개념
- **문제**: 특성(Feature)과 타겟(Target)을 각각 설명하고, 독립변수/종속변수, 입력/출력 중 어느 쪽에 해당하는지 쓰시오.
- **내 답변**: 특성: 학습 모델에 주어지는 입력 값 / 타겟: 머신러닝 모델이 예측하고자 하는 결과 값 또는 출력 값 / 특성: 독립변수, 타겟: 종속변수 / 특성: 입력, 타겟: 출력
- **판정**: O
- **정답**: 특성 = 학습을 위해 모델에 주어지는 입력값(Input data), 데이터의 속성·변수, 독립변수, 타겟 예측의 기준 정보("문제") / 타겟(레이블·클래스·정답) = 모델이 예측하려는 결과값(Output data), 종속변수, 문제 종류에 따라 분류·회귀로 적용("정답")
- **해설**: 슬라이드 12~13쪽. 비만 예시에서 키·몸무게 = 특성, 비만 여부(예=1/아니오=0) = 타겟(레이블). 코드에서는 iris.data = 특성(X), iris.target = 타겟(y).

### W2-04. 하이퍼파라미터 튜닝 적용 분야 — 2026-10-17
- **유형**: 객관식
- **문제**: 다음 중 "교차 검증, Grid Search 등 하이퍼파라미터 튜닝 도구"를 제공하는 적용 분야는? ① 차원 축소 ② 클러스터링 ③ 모델 선택 ④ 전처리
- **내 답변**: 3
- **판정**: O
- **정답**: ③ 모델 선택(Model Selection)
- **해설**: 슬라이드 5쪽 적용 분야 표. 차원 축소 = PCA·t-SNE(고차원 데이터 시각화·압축), 클러스터링 = K-Means·DBSCAN(비지도 군집화), 전처리 = 정규화·스케일링·인코딩, 분류 = SVM·k-NN·랜덤 포레스트, 회귀 = 선형 회귀·릿지 회귀.

### W2-18. 붓꽃 데이터 · 결정트리 관련 인물 — 2026-10-17
- **유형**: 인물
- **문제**: 붓꽃 데이터셋을 1936년 논문에서 사용한 통계학자와, 1986년 의사결정트리를 발표한 사람은?
- **내 답변**: Ronald A. Fisher / 로스 퀸란
- **판정**: O
- **정답**: Ronald A. Fisher(로널드 피셔, 1890~1962, 영국 수학자·통계학자) / Ross Quinlan(로스 퀸란)
- **해설**: 슬라이드 14, 19쪽. 퀸란이 발표한 알고리즘은 ID3(Iterative Dichotomiser 3)로 6주차에서 자세히 다룸.

### W2-26. 결정트리 객체 생성·학습·예측 코드 — 2026-10-17
- **유형**: 코드 빈칸
- **문제**: 결정트리 모델 객체를 만들고 학습·예측하는 코드를 완성하시오. `dt_clf = ( ① )(random_state=11)` / `dt_clf.( ② )(X_train, y_train)` / `pred = dt_clf.( ③ )(X_test)`
- **내 답변**: 모름 ("?")
- **판정**: X
- **정답**: ① DecisionTreeClassifier ② fit ③ predict
- **해설**: 모든 Estimator의 공통 패턴 = **객체 생성 → fit(학습) → predict(예측)**. fit에는 학습용 특징+정답(X_train, y_train), predict에는 테스트용 특징만(X_test). import는 `from sklearn.tree import DecisionTreeClassifier`. 이어서 `accuracy_score(y_test, pred)`로 평가(강의 결과 0.9333). (슬라이드 21, 25~26쪽)

## 세션 요약 — 2026-10-17
- 맞힌 문제: 6 / 총 10 (△ 2개)
- 틀린 문제: W2-23, W2-26 / △: W2-15, W2-24
- 복습 포인트:
  - train_test_split 반환 순서는 **X_train, X_test, y_train, y_test** (X 두 개 먼저).
  - 학습 코드 3단계: `DecisionTreeClassifier(random_state=11)` → `.fit(X_train, y_train)` → `.predict(X_test)`.
  - 붓꽃 문제 유형은 **다중 클래스 분류**(클래스 3개).
  - test_size 기본값은 **0.25** (강의의 7:3 설명과 헷갈리지 말 것).

<!-- 기록 끝 -->
