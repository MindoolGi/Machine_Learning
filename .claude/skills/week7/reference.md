# 7주차 결정트리 하이퍼 파라미터 — 강의 요약 & 문제 은행

> 출처: `2026년_2학기_7주차_머신러닝_결정트리 하이퍼 파라미터.pdf` (32쪽). 이 파일의 정답은 채팅에 노출하지 말고 WEEK7.md에만 쓴다.

## A. 강의 요약

### 학습 목표
결정트리 하이퍼 파라미터 / 결정 트리 피처 중요도 / 결정 트리 과적합(샘플 데이터 생성) / 결정 트리 실습(위스콘신 유방암 데이터셋)

### DecisionTreeClassifier 기본값
`criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0`
- criterion: 분할 품질 기준 'gini'(지니 불순도, 기본) 또는 'entropy'(정보 이득/엔트로피)
- 지니 = 1 − Σpᵢ², 엔트로피 = −Σpᵢ·log₂pᵢ (두 클래스 반반이면 지니 0.5, 엔트로피 1.0)

### 시각화 노드 정보
- **분할 조건**(예 petal length (cm) <= 2.45): 자식 노드를 만들기 위한 규칙. **조건이 없으면 리프 노드**
- **gini**: 지니 계수(불순도) / **samples**: 현재 노드의 데이터 수 / **value**: 클래스 값 기반 데이터 수 / **class**: value에서 가장 많은 건수의 클래스

### 주요 하이퍼 파라미터
| 파라미터 | 설명 |
|---|---|
| **max_depth** | 트리의 최대 깊이. 기본 None이면 클래스가 완벽하게 결정될 때까지(순수해질 때까지), 또는 노드 데이터 수가 min_samples_split보다 작아질 때까지 계속 분할 → 깊어지면 과적합 위험, 적절히 제어 필요 |
| **max_features** | 최적 분할을 위해 고려할 최대 피처 개수. 기본 None = 모든 피처 사용. int = 개수, float = 전체 피처 중 비율, 'sqrt'(/'auto') = √(전체 피처 수), 'log2' = log₂(전체 피처 수) |
| **min_samples_split** | 노드를 분할하기 위한 최소 샘플 수(기본 2). 과적합 제어에 사용. **작게 설정할수록 분할 노드가 많아져 과적합 가능성 증가** |
| **min_samples_leaf** | 분할될 경우 왼쪽·오른쪽 자식(리프) 노드가 가져야 할 최소 샘플 수(기본 1). **크게 설정할수록** 조건을 만족하기 어려워 분할을 덜 수행 → 과적합 억제 |
| **max_leaf_nodes** | 말단 노드(Leaf)의 최대 개수 |
- 붓꽃 예시
  - max_depth=None: gini가 0이 될 때까지 깊이 5까지 분할 / max_depth=4, 3: 깊이 제한, max_depth=3이면 리프에 gini 0.5(value [0,2,2]) 같은 불순 노드가 남음
  - **min_samples_split=4**: samples가 3인 노드는 클래스가 섞여 있어도(gini 0.444, value [0,2,1]) 더 이상 분할하지 않음 (4 이상이어야 분할 가능)
  - **min_samples_leaf=4**: 분할 후 자식 노드의 데이터 수가 4 이상인지 확인하고 분할 → 리프에 samples=4 노드(value [0,3,1], [0,2,2], [0,1,3])가 생김

### 피처 중요도 (feature_importances_)
- 노드를 분할할 때 해당 피처가 클래스를 나누는 데 얼마나 영향을 미쳤는지 나타내는 척도. 학습 시 분할에 사용된 피처의 상대적 중요도 (모든 값의 합 = 1, 분할에 안 쓰인 피처는 0)
- 속성: `결정트리객체.feature_importances_` (학습 후 생기는 속성, 끝에 언더바)
- 붓꽃: [0.025, 0., 0.555, 0.42] → sepal length 0.025, sepal width 0.000, **petal length 0.555**(가장 중요), petal width 0.420
```python
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print("{0} : {1:.3f}".format(name, value))
import seaborn as sns
sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
```

### 과적합(Overfitting)
- 모델이 훈련 데이터의 패턴뿐 아니라 그 안의 **노이즈나 우연한 특성까지 학습** → 학습 데이터에는 잘 맞지만 **새로운 데이터에는 일반화 성능이 떨어지는** 현상
- 그림: Underfitting(직선, 너무 단순) / Just right / Overfitting(모든 점을 지나는 구불구불한 곡선)
- **발생 원인**: ① 모델 복잡도 과다(파라미터가 너무 많음: 깊은 신경망, 고차 다항 회귀) ② 데이터 부족(특이 샘플에 과하게 적응) ③ 노이즈 포함(잡음을 의미 있는 패턴으로 잘못 학습) ④ 정규화(규제, Regularization) 부족
- 슬라이드의 "해결 방법": 더 복잡한 모델 사용, 학습 시간 늘리기, 더 풍부한 feature 추가, 학습률 조정 및 적절한 하이퍼 파라미터 탐색
  - ※ 앞의 세 항목은 엄밀히는 **과소적합(Underfitting)** 해결책. 결정트리 과적합의 실제 대처: max_depth 제한, min_samples_split/min_samples_leaf 크게, max_leaf_nodes 제한, 데이터 늘리기·노이즈 제거, 규제·가지치기(ccp_alpha), 교차 검증으로 하이퍼 파라미터 탐색

### make_classification (샘플 데이터 생성)
- `from sklearn.datasets import make_classification` — 분류용 가상 데이터셋 생성(학습·실험·시각화 테스트용)
- 파라미터(기본값): n_samples(100) 샘플 수, n_features(20) 전체 특징 수, n_informative(2) 종속변수와 상관 있는 성분 수, n_redundant(2) 다른 특성의 선형 결합으로 만들어지는 특성 수, n_classes(2) 클래스 수, n_clusters_per_class(2) 클래스당 클러스터 수, random_state 등
- 반환 (X, y): X = [n_samples, n_features] ndarray(독립변수), y = [n_samples] ndarray(종속변수)
- 실습: `make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)` → X (100, 2), y (100,), 2차원 시각화용 2개 피처·3개 클래스
- 산점도: `plt.scatter(X[:,0], X[:,1], marker='o', c=y_labels, s=40, cmap='rainbow', edgecolor='k')`
- 한글 깨짐 방지: `matplotlib.rcParams['font.family'] = 'Malgun Gothic'`, `matplotlib.rcParams['axes.unicode_minus'] = False`
- **visualize_boundary(model, X, y)**: 모델을 fit한 뒤 `np.meshgrid`로 만든 200×200 격자 모든 좌표를 predict하고 `contourf()`로 클래스 영역(Decision Boundary)을 색칠
- 결과 비교
  - 기본 `DecisionTreeClassifier(random_state=156)`: 이상치 하나까지 맞추려는 가늘고 복잡한 경계 → **과적합**
  - `min_samples_leaf=6` 추가: 단순하고 일반화된 경계 → 과적합 완화 (리프 노드가 최소 6개 샘플을 가져야 함)

### 위스콘신 유방암 데이터 실습
- `from sklearn.datasets import load_breast_cancer`, scikit-learn 대표 **이진 분류** 데이터셋
- 출처 UCI ML Repository, 제공자 Dr. William H. Wolberg(University of Wisconsin Hospitals)
- 샘플 **569개**(악성 212, 양성 357), 특징 **30개**(연속형), 타깃 **0 = 악성(Malignant), 1 = 양성(Benign)**
- 특징 = 세포 핵의 형태적 속성 10가지 × 통계치 3종(**mean 평균, error 표준오차, worst 최대값**) = 30
  - radius(반지름), texture(조직, 회색조 표준편차), perimeter(둘레), area(면적), smoothness(매끄러움), compactness(조밀도: 둘레²/면적 − 1.0), concavity(오목성), concave points(오목한 부분 개수), symmetry(대칭성), fractal dimension(프랙탈 차원, 경계 복잡도)
```python
bc = load_breast_cancer()
bc_df = pd.DataFrame(data=bc.data, columns=bc.feature_names); bc_df['label'] = bc.target
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.25,
                                                    random_state=42, stratify=bc.target)
param_grid = {"criterion": ["gini", "entropy"], "max_depth": [None, 3, 4, 5, 6, 8],
              "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
              "class_weight": [None, "balanced"]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=param_grid,
                    scoring="roc_auc", cv=5, n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)
best_clf = grid.best_estimator_
proba = best_clf.predict_proba(X_test)[:, 1]; y_pred = best_clf.predict(X_test)
roc_auc_score(y_test, proba); accuracy_score(y_test, y_pred)
```
- **stratify=bc.target**: 학습/테스트에 레이블 비율(악성:양성)을 원본과 같게 유지하며 분할. test 0.25 → 테스트 143개
- 파라미터 조합 수: 2 × 6 × 3 × 3 × 2 = **216**, cv=5 → 1080번 학습
- **GridSearchCV** 입력: estimator(기계학습 모델), param_grid(찾을 파라미터, 딕셔너리), scoring(평가 지표), n_jobs(-1 = CPU 코어 모두 사용), refit=True(최적 파라미터로 전체 학습 데이터 재학습 → 모델 생성), cv(교차검증 폴드 수), return_train_score
- 학습 후 속성: `cv_results_`(하이퍼파라미터별 교차검증 결과 딕셔너리), `best_params_`(최적 하이퍼 파라미터), `best_score_`(최고 점수 — scoring 지표 기준, 여기서는 ROC-AUC. 슬라이드 표기는 "최고의 정확도"), `best_estimator_`(최적 파라미터로 학습된 모델)
- 결과: Best Params `{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 10}`, CV Best ROC-AUC 0.9663, **Test ROC-AUC 0.9306, Test Accuracy 0.9161**
- class_weight='balanced': 클래스 빈도에 반비례하는 가중치로 불균형 보정 → 트리 루트 value = [0.5, 0.5], entropy = 1.0
- 피처 중요도(합 1): **worst perimeter 0.6876**(최대), worst concave points 0.1911, mean texture 0.0689, area error 0.0339, worst radius 0.0185, 나머지 25개는 0. `pd.Series(data=best_clf.feature_importances_, index=bc.feature_names)`
- 트리: 루트 `worst perimeter <= 102.4` → 왼쪽 `worst concave points <= 0.18`, 오른쪽 `worst concave points <= 0.15` → 깊이 3
- ROC 곡선: `roc_curve_plot(y_test, best_clf.predict_proba(X_test)[:, -1])` ([:, -1] = 마지막 열 = 클래스 1 확률)
- Graphviz: `export_graphviz(best_clf, out_file=None, feature_names=..., class_names=..., filled=True, rounded=True, proportion=True, precision=2)` → out_file=None이면 DOT 문자열 반환, `graphviz.Source(dot_data)`, `graph.render("breast_cancer_tree", format="png", cleanup=True)`로 PNG 저장
  - filled = 색상 채우기(불순도 표현), rounded = 모서리 둥글게, proportion = samples·value를 비율로 표시, precision = 소수점 자리수

## B. 문제 은행 (ID · 유형 · 문제 · 정답 · 해설)

### W7-01 · 기본값
Q: DecisionTreeClassifier의 다음 하이퍼 파라미터 기본값을 쓰시오: criterion, max_depth, min_samples_split, min_samples_leaf, max_features
A: criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None

### W7-02 · 노드 해석
Q: 결정트리 시각화 노드에 나오는 분할 조건, gini, samples, value, class의 의미를 쓰시오. 분할 조건이 없는 노드는?
A: 분할 조건 = 자식 노드를 만드는 규칙 / gini = 지니 계수(불순도) / samples = 현재 노드 데이터 수 / value = 클래스별 데이터 수 / class = value 중 가장 많은 클래스. 조건이 없으면 리프 노드

### W7-03 · 개념
Q: max_depth를 기본값(None)으로 두면 트리는 언제까지 분할하며, 어떤 문제가 생길 수 있나?
A: 클래스가 완벽하게 결정될 때(노드가 순수해질 때)까지, 또는 노드 데이터 수가 min_samples_split보다 작아질 때까지 계속 깊이를 키움 → 과적합 위험
해설: 적절한 값으로 제어 필요.

### W7-04 · 파라미터 매칭
Q: 다음 설명에 맞는 하이퍼 파라미터는? (1) 노드를 분할하기 위한 최소 샘플 수 (2) 분할 후 자식(리프) 노드가 가져야 할 최소 샘플 수 (3) 말단 노드의 최대 개수 (4) 최적 분할을 위해 고려할 최대 피처 수
A: (1) min_samples_split (2) min_samples_leaf (3) max_leaf_nodes (4) max_features

### W7-05 · 상황 판단
Q: 과적합을 줄이려면 min_samples_split과 min_samples_leaf를 각각 크게 해야 하나 작게 해야 하나? max_depth는?
A: min_samples_split 크게, min_samples_leaf 크게, max_depth 작게(제한)
해설: min_samples_split을 작게 하면 분할 노드가 많아져 과적합 증가. min_samples_leaf를 크게 하면 분할을 덜 수행.

### W7-06 · 개념
Q: max_features에 int, float, 'sqrt', 'log2'를 지정하면 각각 어떻게 동작하나? 피처가 16개일 때 'sqrt'와 'log2'는 몇 개를 고려하나?
A: int = 해당 개수, float = 전체 피처 중 비율, 'sqrt' = √(전체 피처 수), 'log2' = log₂(전체 피처 수) / 16개면 sqrt → 4개, log2 → 4개
해설: 기본 None = 모든 피처 사용.

### W7-07 · 노드 해석
Q: min_samples_split=4로 설정했을 때, samples=3이고 value=[0, 2, 1]인 노드는 분할되는가? 이유는?
A: 분할되지 않음. 분할하려면 노드 샘플 수가 최소 4 이상이어야 하므로 클래스가 섞여 있어도(gini 0.444) 리프로 남음

### W7-08 · 노드 해석
Q: min_samples_leaf=4로 설정하면 트리의 리프 노드에는 어떤 특징이 나타나는가?
A: 분할 후 자식 노드의 데이터 수가 4 이상인 경우에만 분할하므로, 모든 리프 노드의 samples가 4 이상(예: samples=4, value [0,3,1])이 되고 불순한 리프가 생길 수 있음

### W7-09 · 계산
Q: max_depth=3인 붓꽃 트리의 한 리프가 value=[0, 2, 2]이다. 이 노드의 지니 불순도와 class 표시는?
A: 1 − (0.5² + 0.5²) = 0.5, class는 동률이므로 앞 순서의 versicolor
해설: 강의 그림의 gini = 0.5, class = versicolor 리프.

### W7-10 · 코드
Q: 학습된 결정트리 dt_clf의 피처 중요도를 확인하는 속성 이름은? 값들의 합은?
A: `dt_clf.feature_importances_` / 1
해설: 분할에 쓰이지 않은 피처는 0. 학습(fit) 후에만 존재(끝의 언더바).

### W7-11 · 데이터 해석
Q: 붓꽃 결정트리의 피처 중요도가 [0.025, 0., 0.555, 0.42]일 때 가장 중요한 피처와 분할에 쓰이지 않은 피처는? (순서: sepal length, sepal width, petal length, petal width)
A: 가장 중요: petal length(0.555) / 쓰이지 않음: sepal width(0.0)

### W7-12 · 코드 빈칸
Q: 피처 이름과 중요도를 짝지어 출력하는 코드를 완성하시오.
```python
for name, value in ( ① )(iris_data.feature_names, dt_clf.( ② )):
    print("{0} : {1:.3f}".format(name, value))
```
A: ① zip ② feature_importances_
해설: 시각화는 `sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)`.

### W7-13 · 개념
Q: 과적합(Overfitting)을 정의하시오.
A: 모델이 훈련 데이터의 패턴뿐 아니라 노이즈나 우연한 특성까지 학습하여, 학습 데이터에는 매우 잘 맞지만 새로운 데이터에 대한 일반화 성능이 떨어지는 현상

### W7-14 · 개념
Q: 강의에서 제시한 과적합 발생 원인 4가지를 쓰시오.
A: ① 모델 복잡도 과다(파라미터가 너무 많음) ② 데이터 부족 ③ 노이즈 포함 ④ 정규화(규제) 부족

### W7-15 · 개념
Q: 강의 슬라이드에 "과적합 해결 방법"으로 제시된 4가지를 쓰시오.
A: 더 복잡한 모델 사용, 학습 시간 늘리기, 더 풍부한 feature 추가, 학습률 조정 및 적절한 하이퍼 파라미터 탐색
해설: 슬라이드 12쪽 그대로. 단, 앞의 세 가지는 원래 과소적합 해결책이며, 결정트리 과적합은 max_depth 제한·min_samples_leaf 증가·데이터 추가·가지치기 등으로 줄인다는 점도 같이 알아 둘 것.

### W7-16 · O·X
Q: 결정트리에서 max_depth를 크게 하고 min_samples_leaf를 1로 두면 과적합이 줄어든다. (O/X)
A: X
해설: 트리가 더 깊고 세밀하게 분할되어 과적합이 커짐.

### W7-17 · 매개변수
Q: make_classification의 n_informative와 n_redundant의 의미는? n_samples와 n_features의 기본값은?
A: n_informative = 종속변수(타겟)와 상관관계가 있는 독립변수 성분 수, n_redundant = 다른 특성의 선형 결합으로 만들어지는 특성 수 / n_samples 100, n_features 20

### W7-18 · 코드 결과
Q: `X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)`의 X.shape와 y.shape는?
A: X (100, 2), y (100,)
해설: n_samples 기본값 100.

### W7-19 · 상황 판단
Q: 기본 결정트리와 min_samples_leaf=6을 준 결정트리의 Decision Boundary를 비교했을 때 차이와 그 의미는?
A: 기본 트리는 이상치까지 맞추려고 가늘고 복잡한 경계를 만들어 과적합, min_samples_leaf=6 트리는 경계가 단순해져 일반화 성능이 더 좋을 것으로 기대
해설: 슬라이드 18~20쪽.

### W7-20 · 개념
Q: 위스콘신 유방암 데이터셋의 샘플 수(악성/양성), 특징 수, 타깃 0과 1의 의미, 문제 유형은?
A: 569개(악성 212, 양성 357), 30개, 0 = 악성(Malignant), 1 = 양성(Benign), 이진 분류
해설: 0이 악성이라는 점이 자주 헷갈림.

### W7-21 · 개념
Q: 유방암 데이터의 특징이 30개인 이유를 설명하시오.
A: 세포 핵의 형태적 속성 10가지(radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) × 통계치 3종(mean 평균, error 표준오차, worst 최대값) = 30

### W7-22 · 코드
Q: train_test_split에서 `stratify=bc.target`의 역할은?
A: 학습/테스트 데이터의 레이블(악성:양성) 비율을 원본 데이터와 같게 유지하며 분할(층화 분할)

### W7-23 · 계산
Q: 다음 param_grid로 GridSearchCV(cv=5)를 수행하면 하이퍼 파라미터 조합 수와 총 학습 횟수는? criterion 2개, max_depth [None,3,4,5,6,8], min_samples_split [2,5,10], min_samples_leaf [1,2,4], class_weight [None,'balanced']
A: 2 × 6 × 3 × 3 × 2 = 216개 조합, × 5폴드 = 1080번 학습 (refit 1회 제외)

### W7-24 · 매개변수
Q: GridSearchCV의 매개변수 estimator, param_grid, scoring, cv, n_jobs=-1, refit=True의 의미를 쓰시오.
A: estimator = 기계학습 모델 / param_grid = 찾을 하이퍼 파라미터(딕셔너리) / scoring = 예측 평가 지표(예 "roc_auc") / cv = 교차검증 폴드 수 / n_jobs=-1 = 모든 CPU 코어 사용 / refit=True = 찾은 최적 파라미터로 모델을 다시 학습해 생성

### W7-25 · 속성 매칭
Q: GridSearchCV 학습 후 (1) 최적 하이퍼 파라미터 (2) 최고 점수 (3) 최적 파라미터로 학습된 모델 (4) 전체 교차검증 결과를 담은 속성 이름은?
A: (1) best_params_ (2) best_score_ (3) best_estimator_ (4) cv_results_
해설: best_score_는 scoring 지표 기준 점수(이 실습에선 ROC-AUC).

### W7-26 · 데이터 해석
Q: 유방암 GridSearchCV 실습의 최적 파라미터(criterion, max_depth)와 테스트 ROC-AUC, 테스트 정확도는?
A: criterion='entropy', max_depth=3 (class_weight='balanced', min_samples_leaf=4, min_samples_split=10), Test ROC-AUC ≈ 0.9306, Test Accuracy ≈ 0.9161
해설: CV Best ROC-AUC는 0.9663.

### W7-27 · 데이터 해석
Q: 유방암 최적 결정트리에서 피처 중요도가 가장 높은 피처와 두 번째 피처는? 루트 노드의 분할 피처는?
A: 1위 worst perimeter(약 0.688), 2위 worst concave points(약 0.191) / 루트: worst perimeter <= 102.4
해설: 중요도 1위 피처가 루트 분할에 쓰임.

### W7-28 · 계산
Q: class_weight='balanced'로 학습한 유방암 트리 루트 노드의 value가 [0.5, 0.5]일 때 entropy는? (criterion='entropy')
A: −(0.5·log₂0.5 + 0.5·log₂0.5) = 1.0
해설: 엔트로피 = −Σp·log₂p, 두 클래스가 반반이면 최댓값 1.

### W7-29 · 코드
Q: GridSearchCV로 찾은 최적 모델 best_clf로 테스트 ROC-AUC를 구하는 코드를 쓰시오.
A: `proba = best_clf.predict_proba(X_test)[:, 1]` → `roc_auc_score(y_test, proba)`
해설: ROC-AUC는 예측 클래스가 아니라 Positive(1) 확률을 넣음. `[:, -1]`도 같은 열.

### W7-30 · 매개변수
Q: export_graphviz에서 out_file=None, rounded=True, proportion=True, precision=2의 의미는? PNG 파일로 저장하는 코드는?
A: out_file=None: 파일 대신 DOT 문자열 반환 / rounded: 노드 모서리 둥글게 / proportion: samples·value를 비율로 표시 / precision=2: 소수점 2자리 / `graph = graphviz.Source(dot_data); graph.render("breast_cancer_tree", format="png", cleanup=True)`

### W7-31 · 개념
Q: criterion에 지정할 수 있는 두 값과 각각의 불순도 공식은?
A: 'gini': 지니 = 1 − Σpᵢ² / 'entropy': 엔트로피 = −Σpᵢ·log₂pᵢ
해설: 기본값은 'gini'.
