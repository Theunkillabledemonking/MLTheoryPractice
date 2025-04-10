# 계층화 분활

from sklearn.model_selection import train_test_split
from collections import Counter

# 1. 데이터 생성
X = [f"img_{i:2d}.jpg" for i in range(25)] # 이미지 파일 이름
y = [angle for angle in [30, 60, 90, 120, 150] for _ in range(5)] # 각도별 5개씩

# 2. 계층화 분할 (테스트 20%, 각도 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    stratify=y,
    random_state=42
)

# 3. 각도별 샘플 수 출력
print("훈련 세트 각도 분포: ", Counter(y_train))
print("테스트 세트 각도 분포:", Counter(y_test))