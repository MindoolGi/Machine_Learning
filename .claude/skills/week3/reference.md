# 3주차 데이터 시각화 / Matplotlib — 강의 요약 & 문제 은행

> 출처: `2026년_2학기_3주차_머신러닝_데이터시각화_Matplotlib.pdf` (38쪽). 이 파일의 정답은 채팅에 노출하지 말고 WEEK3.md에만 쓴다.

## A. 강의 요약

### Matplotlib 소개
- 데이터를 그래프·플롯(plot)으로 **시각화**하기 위한 파이썬 대표 라이브러리(플롯팅 라이브러리)
- **플롯(Plot)**: 그래픽으로 렌더링해 화면에 출력하기 전에 데이터를 가상의 공간에 미리 그려 놓는 것
- **존 D. 헌터(John D. Hunter)** 개발. 출판물 수준의 그래프가 목표. **MATLAB**에서 영감
- 분석 전 데이터 이해를 위한 시각화 / 분석 후 결과 시각화에 사용. **pyplot 모듈** 사용
- import: `from matplotlib import pyplot as plt` 또는 `import matplotlib.pyplot as plt`
- 샘플 갤러리: https://matplotlib.org/gallery/index.html
- 시각화 종류: plot(line), bar(막대), box, pie, scatter(산포도), histogram, back-to-back, stack bar

### 기본 함수
| 함수 | 기능 | 사용 |
|---|---|---|
| plot() | 입력 데이터로 시각화 데이터 생성. 입력: 리스트·튜플, pandas Series, numpy array | `plt.plot(x, y)` |
| xlabel()/ylabel() | x, y축 이름 | `plt.xlabel("문자열")` |
| title() | 그래프 제목 | `plt.title("문자열")` |
| show() | 시각화 데이터 출력 | `plt.show()` |
| savefig() | 이미지 파일로 저장 | `plt.savefig("my_plot.png")` |
- `plt.figure()`: 새 그림(figure) 생성. `plt.figure(figsize=(30, 5))` → 가로 30, 세로 5(인치) 크기
- 색상 문자: 'b' Blue, 'g' Green, 'r' Red, 'c' Cyan, 'm' Magenta, 'y' Yellow, 'k' **Black**, 'w' White. 색 이름도 가능('hotpink')
- 선 스타일: '-' 실선, '--' 점선(dashed), '-.' 일점쇄선(dash-dot), ':' 점선(dotted)
- 마커: '.' point, ',' pixel, 'o' circle, 'v' 아래 삼각형, '^' 위 삼각형, '<' 왼쪽 삼각형, '>' 오른쪽 삼각형, '1'~'4' tri, 's' square, 'p' pentagon, '*' star, 'h'/'H' hexagon, '+' plus, 'x' X, 'D' diamond, 'd' thin diamond, '|' vline, '_' hline
- 조합: `'--r'` = 빨간 점선, `"-ob"` = 실선 + 원 마커 + 파란색
- 구간: `plt.xlim(2, 5)`, `plt.ylim(20, 60)` — 축 표시 범위 지정
- 범례: `plt.plot(x, y, 'b', label="My Score")` 후 `plt.legend(loc='lower right')` 또는 `loc=4`
  - loc 코드: best 0, upper right 1, upper left 2, lower left 3, lower right 4, right 5, center left 6, center right 7, lower center 8, upper center 9, center 10
- 주석: `plt.annotate('Pick !', xy=(5,70), xytext=(3,60), arrowprops={'color':'green'})` — xy = 화살표가 가리키는 점, xytext = 텍스트 위치, arrowprops = 화살표 속성(딕셔너리)
- 스타일: `plt.style.use("ggplot")`, 목록 `print(plt.style.available)` (강의 기준 28종: 'Solarize_Light2', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8-...' 등)

### 막대그래프 bar() / barh()
- 데이터 값을 막대 길이로 표현. barh = 수평 막대
- 필수: x, y 리스트. 옵션: facecolor·color(막대 색), edgecolor(테두리 색), alpha(투명도 0~1), width(막대 크기, **기본 0.8**), linewidth(테두리 굵기)
- 예: `plt.barh(x_list, y_list, facecolor="red", edgecolor="black", alpha=0.3, linewidth=3)`

### 산포도 scatter()
- 데이터가 얼마나 어떻게 퍼져 있는지 보여줌
- 옵션: facecolor·color·**c**(점 색), edgecolor(테두리), **s**(점 크기), alpha, **marker**(점 모양), linewidth
- 예: `plt.scatter(x, y, c="yellow", edgecolor="black", s=300, linewidth=3, marker='<')`

### 파이 차트 pie()
- 입력값의 합을 100으로 하여 각 데이터의 비율을 원으로 표현
- 필수: 비율 데이터 리스트. 옵션: labels(범주 리스트), startangle(시작 각도), autopct(비율 표시 문자열 포맷 '%.2f'), counterclock(출력 순서, **기본 True=반시계 방향**)
- 예: ratio=[35,30,25,20,20], labels=['Apple','Samsung','Google','Naver','Kakao'], startangle=90, autopct='%.2f' → 합 130 기준 Apple 26.92, Samsung 23.08, Google 19.23, Naver·Kakao 15.38

### 히스토그램 hist()
- 전체 집단의 분포 상태를 파악·비교
- 옵션: **bins**(막대 개수, 기본 10), **density**(True면 확률밀도로 정규화), **histtype**('bar', 'barstacked', 'step', 'stepfilled'), alpha, facecolor·color, edgecolor
- 예: `plt.hist(hist_list, bins=7, density=True, histtype="step", edgecolor="black")` — step = 속이 빈 막대

### Pandas·Numpy와 Matplotlib
- Series 입력: `bc = pd.Series({"1":10, "2":30, ...}); plt.plot(bc)` → 인덱스가 x축, 값이 y축
- `plt.savefig("My_block_chain1.png")` 로 이미지 저장 (show() 전에 호출)
- DataFrame 자체 plot() 제공. **Jupyter Notebook에서는 show() 없이 출력 가능**
  - `df.plot()` / `df.plot.line()` / `df.plot.line(subplots=True)` — 선
  - `df.plot.bar()` / `df.plot(kind="bar")` / `df.plot(kind="bar", subplots=True)` — 막대
  - `df.plot.box()` / `df.plot(kind="box")` — 박스
  - `df.plot.pie()` / `df.plot(kind="pie", subplots=True)` / `df.plot(kind="pie", y="kor")` — 파이(여러 열이면 subplots=True 또는 y로 열 지정)
  - **subplots=True**: 여러 데이터(열)를 그래프를 나눠서 시각화
  - 예: `df = pd.DataFrame([[60,90,95],[80,75,100]], columns=["kor","math","eng"], index=["2019","2020"])` → 인덱스(연도)가 x축, 열(과목)마다 선/막대
- Numpy: `x = np.linspace(0, 10, 20)` = 0부터 10까지 **20개**의 균등 간격 값, `y = x**2.0`, `plt.plot(x, y, "-ob")`

### 실습: 기상자료개방포털 서울 기온 데이터
- 기상자료개방포털 https://data.kma.go.kr — 지상·해양·고층·항공·위성·레이더·수치예보 등 30종 날씨 데이터, 100년 이상 기후통계(기온, 강수량, 극값순위, 기후평년값, 장마, 황사, 폭염, 열대야 일수 등)
- 경로: [기후통계분석] - [조건별통계] / [통계분석] - [기온분석], 지역 **서울(108)**, 기간 일 단위, 파일명 `seoul.csv`
- CSV 열: 날짜, 지점, 평균기온, 최저기온, 최고기온 (슬라이드 28쪽 순서 → row[-1] = 최고기온). ※ 2000년 이후 데이터 화면(27쪽)과 32쪽 코드는 평균, 최고, 최저 순으로 row[3]=최고, row[4]=최저를 사용 — 슬라이드 간 열 순서가 다르므로 문제의 열 순서를 기준으로 판단
```python
import csv
f = open('seoul.csv')          # 파일 열기
data = csv.reader(f)           # 읽을 수 있는 reader 객체
next(data)                     # 첫 줄(헤더) 건너뛰기: 현재 값을 반환하고 다음 반복은 다음 줄부터
result = []
for row in data:               # 한 줄씩 리스트(문자열 원소)로 row에 저장
    if row[-1] != '':          # 값이 비어 있지 않을 때(결측 제외)
        if row[0].split('-')[1] == '08':      # '1907-10-01'.split('-') → ['1907','10','01'], [1]=월
            result.append(float(row[-1]))     # 문자열 → 실수 변환 후 추가
plt.plot(result, 'hotpink'); plt.show()      # x축은 인덱스(0,1,2,...)
```
- row의 각 원소는 **문자열**이므로 계산하려면 float() 변환 필요
- with 없이 open하면 `f.close()`로 직접 닫아야 함
- Numpy 활용: `np.array(리스트)`, `np.mean`(평균 13.21°C), `np.std`(표준편차 10.57), `np.max`(최고 39.6°C), `np.min`(최저 -18.6°C), `diff_arr = max_arr - min_arr`(일교차, 평균 8.49°C), `np.argmax/argmin`(최댓값/최솟값의 **인덱스** → dates[인덱스]로 날짜: 일교차 최대 2025-04-12 21.1°C, 최소 2024-07-29 1.0°C), 데이터 9752일
- 불리언 마스크: `hot_mask = max_arr >= 33`(폭염), `np.array(dates)[hot_mask]`, `max_arr[hot_mask]`, `hot_mask.sum()` = True 개수 = 폭염일수 312일
- Pandas: `pd.DataFrame({'날짜': pd.to_datetime(dates), '평균기온': ..., ...})`, `df.set_index('날짜', inplace=True)`, `df['일교차'] = df['최고기온'] - df['최저기온']`, `df.head()`
- `plt.plot(df.index, df['평균기온'])` → 연도별 계절 주기 패턴

## B. 문제 은행 (ID · 유형 · 문제 · 정답 · 해설)

### W3-01 · 개념
Q: Matplotlib을 개발한 사람과 영감을 받은 프로그램은?
A: 존 D. 헌터(John D. Hunter), MATLAB
해설: 출판물에 필요한 수준의 그래프를 만드는 데 초점.

### W3-02 · 개념
Q: 강의에서 정의한 "플롯(Plot)"의 의미를 쓰시오.
A: 그래픽으로 렌더링하여 화면으로 출력하기 전에 데이터를 가상의 공간에 미리 그려 놓는 것
해설: 슬라이드 3쪽.

### W3-03 · 코드
Q: matplotlib의 pyplot 모듈을 plt라는 이름으로 import하는 두 가지 방법을 쓰시오.
A: `from matplotlib import pyplot as plt` / `import matplotlib.pyplot as plt`

### W3-04 · 함수 매칭
Q: 다음 기능의 plt 함수를 쓰시오. (1) 그래프 제목 (2) x축 이름 (3) 화면에 출력 (4) 이미지 파일로 저장 (5) 그림 크기 지정
A: (1) title() (2) xlabel() (3) show() (4) savefig("파일명.png") (5) figure(figsize=(가로, 세로))

### W3-05 · 코드 해석
Q: `plt.plot(x, y, '--r')`과 `plt.plot(x, y, "-ob")`는 각각 어떤 그래프를 그리나?
A: '--r' = 빨간색 점선(dashed) / "-ob" = 파란색 실선 + 원(circle) 마커
해설: 선 스타일·마커·색 문자를 조합.

### W3-06 · 매칭
Q: 색상 문자 'k', 'c', 'm'과 선 스타일 '-.', ':'의 의미를 쓰시오.
A: 'k' Black, 'c' Cyan, 'm' Magenta / '-.' 일점쇄선(dash-dot), ':' 점선(dotted)
해설: 'b'가 Blue이므로 Black은 'k' — 자주 헷갈림.

### W3-07 · 코드
Q: x축 표시 구간을 2~5, y축 표시 구간을 20~60으로 제한하는 코드를 쓰시오.
A: `plt.xlim(2, 5)` / `plt.ylim(20, 60)`

### W3-08 · 코드 빈칸
Q: 두 선에 범례를 오른쪽 아래에 표시하려 한다. 빈칸을 채우시오.
```python
plt.plot(x, my_y, 'b', ( ① ) = "My Score")
plt.plot(x, you_y, 'r', ( ① ) = "Your Score")
plt.( ② )(loc = ( ③ ))
```
A: ① label ② legend ③ 'lower right' 또는 4
해설: label 없이 legend()만 호출하면 범례 항목이 없음.

### W3-09 · 매칭
Q: legend의 loc 코드 0, 1, 2, 4는 각각 어떤 위치인가?
A: 0 = best, 1 = upper right, 2 = upper left, 4 = lower right
해설: 3 lower left, 5 right, 6 center left, 7 center right, 8 lower center, 9 upper center, 10 center.

### W3-10 · 코드 해석
Q: `plt.annotate('Pick !', xy=(5,70), xytext=(3,60), arrowprops={'color':'green'})`에서 xy와 xytext의 의미는?
A: xy = 화살표가 가리키는(주석 대상) 좌표 (5,70), xytext = 텍스트 'Pick !'이 표시되는 위치 (3,60). arrowprops는 화살표 속성 딕셔너리(초록색)
해설: 슬라이드 11쪽.

### W3-11 · 코드
Q: 그래프 스타일을 ggplot으로 바꾸는 코드와, 사용 가능한 스타일 목록을 확인하는 코드는?
A: `plt.style.use("ggplot")` / `print(plt.style.available)`
해설: 강의 기준 28종.

### W3-12 · 옵션
Q: bar() 함수에서 (1) 막대 색 (2) 막대 테두리 색 (3) 투명도 (4) 막대 너비 (5) 테두리 굵기를 지정하는 옵션명은? 막대 너비의 기본값은?
A: (1) color 또는 facecolor (2) edgecolor (3) alpha(0~1) (4) width (5) linewidth / width 기본 0.8
해설: 슬라이드 표에는 기본 0.8이 linewidth 칸에 적혀 있지만 실제 0.8은 width의 기본값.

### W3-13 · 개념
Q: bar()와 barh()의 차이는?
A: bar는 수직 막대그래프, barh는 수평(horizontal) 막대그래프
해설: barh 예제는 xlabel/ylabel을 서로 바꿔 지정함.

### W3-14 · 옵션
Q: scatter()에서 점의 크기와 점의 모양을 지정하는 옵션명은? 점 색을 지정하는 옵션명 3가지는?
A: 크기 s, 모양 marker / 색 c, color, facecolor
해설: 예 `s=300, marker='<'`.

### W3-15 · 개념
Q: 산포도(Scatter)는 무엇을 보여주는 그래프인가?
A: 입력 데이터가 얼마나, 어떻게 퍼져(분포되어) 있는지를 보여줌 (두 변수의 관계)

### W3-16 · 옵션
Q: pie()에서 (1) 범주 이름 (2) 시작 각도 (3) 비율 값 표시 (4) 출력 방향을 지정하는 옵션은? (4)의 기본값은?
A: (1) labels (2) startangle (3) autopct (예 '%.2f') (4) counterclock / 기본 True(반시계 방향)

### W3-17 · 계산
Q: `plt.pie([35, 30, 25, 20, 20], labels=['Apple','Samsung','Google','Naver','Kakao'], autopct='%.2f')`를 실행하면 Apple과 Naver 조각에 표시되는 값은?
A: Apple 26.92, Naver 15.38
해설: 합 130을 100으로 환산. 35/130 = 26.92%, 20/130 = 15.38%.

### W3-18 · 옵션
Q: hist()에서 막대(구간) 개수를 정하는 옵션과, 확률밀도로 정규화하는 옵션은? histtype에 들어갈 수 있는 값 4가지는?
A: bins / density=True / 'bar', 'barstacked', 'step', 'stepfilled'
해설: 'step'은 속이 빈 막대. bins를 지정하지 않으면 10개.

### W3-19 · 개념
Q: 히스토그램과 막대그래프의 용도 차이를 설명하시오.
A: 히스토그램은 연속형 데이터(전체 집단)의 분포 상태를 구간별 빈도로 파악·비교, 막대그래프는 각 항목(범주)의 값을 막대 길이로 비교
해설: 슬라이드 13, 19쪽.

### W3-20 · O·X
Q: Jupyter Notebook에서 DataFrame.plot()을 사용할 때는 반드시 plt.show()를 호출해야 그래프가 출력된다. (O/X)
A: X
해설: 쥬피터노트북 환경에서는 show() 없이 출력 가능(슬라이드 22쪽).

### W3-21 · 코드
Q: DataFrame df의 열마다 그래프를 나누어 막대그래프로 그리는 코드를 두 가지 방식으로 쓰시오.
A: `df.plot(kind="bar", subplots=True)` / `df.plot.bar(subplots=True)`
해설: subplots=True = 여러 데이터를 그래프로 나누어 시각화.

### W3-22 · 코드
Q: 과목 점수 DataFrame(columns=kor, math, eng / index=2019, 2020)에서 kor 열만으로 파이 차트를 그리는 코드는?
A: `df.plot(kind="pie", y="kor")`
해설: 파이는 열이 여러 개면 subplots=True 또는 y로 하나를 지정해야 함. (슬라이드는 '행 이름'이라 적혀 있으나 실제로는 열 이름)

### W3-23 · 개념
Q: `pd.Series({"1":10, "2":30, "3":20})`를 `plt.plot(bc)`에 넣으면 x축과 y축은 각각 무엇이 되나?
A: x축 = Series의 인덱스(키 "1","2","3"), y축 = 값(10, 30, 20)

### W3-24 · 코드 결과
Q: `np.linspace(0, 10, 20)`이 만드는 배열을 설명하시오.
A: 0부터 10까지(10 포함) 균등한 간격의 원소 20개를 가진 numpy 배열
해설: arange와 달리 "개수"를 지정.

### W3-25 · 코드 빈칸
Q: CSV 파일을 읽고 헤더를 건너뛰는 코드를 완성하시오.
```python
import ( ① )
f = open('seoul.csv')
data = csv.( ② )(f)
( ③ )(data)
for row in data:
    print(row)
```
A: ① csv ② reader ③ next
해설: next()는 현재 줄(헤더)을 반환하고, 다음 반복은 그다음 줄부터 진행.

### W3-26 · 코드 해석
Q: 다음 코드에서 `if row[-1] != ''`와 `float(row[-1])`이 필요한 이유는?
A: `row[-1] != ''`: 최고기온 값이 비어 있는(결측) 행을 제외하기 위해 / `float()`: csv.reader가 읽은 값은 모두 문자열이므로 실수로 변환해야 계산·그래프에 사용 가능
해설: row[-1]은 리스트의 마지막 원소(최고기온 열).

### W3-27 · 코드 결과
Q: `row[0] = '1907-10-01'`일 때 `row[0].split('-')[1]`의 결과와, 8월 데이터만 고르는 조건식은?
A: '10' / `if row[0].split('-')[1] == '08':`
해설: split('-') → ['1907', '10', '01']. 결과는 문자열이므로 '08'과 비교.

### W3-28 · 개념
Q: `f = open('seoul.csv', 'r')`처럼 with 문 없이 파일을 열었을 때 작업 후 해야 할 일은?
A: `f.close()`로 파일을 직접 닫아야 함

### W3-29 · 코드
Q: 최고기온 배열 max_arr와 최저기온 배열 min_arr로 일교차가 가장 큰 날의 날짜를 구하는 코드를 쓰시오. (dates는 날짜 리스트)
A: `diff_arr = max_arr - min_arr` → `dates[np.argmax(diff_arr)]`
해설: argmax는 최댓값의 값이 아니라 **인덱스**를 반환. 강의 결과 2025-04-12 (21.1°C).

### W3-30 · 코드
Q: 최고기온 33도 이상인 폭염일수를 numpy 불리언 마스크로 구하는 코드를 쓰시오.
A: `hot_mask = max_arr >= 33` → `hot_mask.sum()` (폭염 날짜는 `np.array(dates)[hot_mask]`)
해설: True=1로 합산. 강의 결과 312일.

### W3-31 · 코드
Q: 날짜 리스트를 datetime으로 바꿔 DataFrame의 인덱스로 설정하고, '일교차' 열을 추가하는 코드를 쓰시오.
A: `df = pd.DataFrame({'날짜': pd.to_datetime(dates), '평균기온': avg_temps, '최고기온': max_temps, '최저기온': min_temps})` → `df.set_index('날짜', inplace=True)` → `df['일교차'] = df['최고기온'] - df['최저기온']`
해설: inplace=True는 원본 df를 직접 변경.

### W3-32 · 개념
Q: 강의 실습에서 사용한 공공데이터 사이트 이름과 서울 지점 번호는?
A: 기상자료개방포털(data.kma.go.kr), 서울 지점 108
