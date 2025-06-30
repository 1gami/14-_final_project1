import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 데이터 경로 설정
BASE_DIR = r'C:/Users/jk972/OneDrive/Desktop/workspace/open'
SAVE_DIR = r'C:/Users/jk972/OneDrive/Desktop/workspace/final project/git'

# 파일 경로
TRAIN_MEMBER = os.path.join(BASE_DIR, 'train/1.회원정보')
TEST_MEMBER = os.path.join(BASE_DIR, 'test/1.회원정보')
TRAIN_MARKETING = os.path.join(BASE_DIR, 'train/7.마케팅정보')
TEST_MARKETING = os.path.join(BASE_DIR, 'test/7.마케팅정보')

# 파일 리스트
train_member_files = sorted(glob.glob(os.path.join(TRAIN_MEMBER, '*.parquet')))
test_member_files = sorted(glob.glob(os.path.join(TEST_MEMBER, '*.parquet')))
train_marketing_files = sorted(glob.glob(os.path.join(TRAIN_MARKETING, '*.parquet')))
test_marketing_files = sorted(glob.glob(os.path.join(TEST_MARKETING, '*.parquet')))

def load_and_concat_parquet(file_list):
    df_list = [pd.read_parquet(f) for f in file_list]
    return pd.concat(df_list, ignore_index=True)

# 1. 데이터 불러오기
print('Loading data...')
df_train_member = load_and_concat_parquet(train_member_files)
df_test_member = load_and_concat_parquet(test_member_files)
df_train_marketing = load_and_concat_parquet(train_marketing_files)
df_test_marketing = load_and_concat_parquet(test_marketing_files)

# 2. EDA (간단 요약)
def eda_report(df, name):
    report = {}
    zero_count = {}
    for col in df.columns:
        report[col] = {
            'dtype': str(df[col].dtype),
            'n_null': df[col].isnull().sum(),
            'n_zero': (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'n_unique': df[col].nunique(),
            'sample_values': df[col].unique()[:5],
            'describe': df[col].describe() if pd.api.types.is_numeric_dtype(df[col]) else None
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            zero_count[col] = (df[col] == 0).sum()
    eda_df = pd.DataFrame(report).T
    eda_df.to_csv(os.path.join(SAVE_DIR, f'{name}_EDA_report.csv'), encoding='utf-8-sig')
    # 수치형 0값 개수 별도 저장
    zero_df = pd.DataFrame(list(zero_count.items()), columns=['col', 'zero_count'])
    zero_df.to_csv(os.path.join(SAVE_DIR, f'{name}_zero_count.csv'), index=False, encoding='utf-8-sig')
    print(f'EDA report saved: {name}_EDA_report.csv')
    print(f'Zero count report saved: {name}_zero_count.csv')
    # 남녀구분코드 비율 별도 저장
    if '남녀구분코드' in df.columns:
        gender_map = {1: '남성', 2: '여성'}
        gender_count = df['남녀구분코드'].map(gender_map).value_counts()
        gender_count.to_csv(os.path.join(SAVE_DIR, f'{name}_gender_count.csv'), encoding='utf-8-sig')
        print(f'Gender count report saved: {name}_gender_count.csv')

eda_report(df_train_member, '회원정보')
eda_report(df_train_marketing, '마케팅정보')

# 3. 전처리 함수 예시
def preprocess_member(df):
    df = df.copy()
    # 성별 컬럼 추가
    if '남녀구분코드' in df.columns:
        df['성별'] = df['남녀구분코드'].map({1: '남성', 2: '여성'})
    # 결측치 처리
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna('missing')
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(0)
    # 범주형 인코딩
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_marketing(df):
    df = df.copy()
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna('missing')
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# 4. 전처리 적용
df_train_member_p = preprocess_member(df_train_member)
df_test_member_p = preprocess_member(df_test_member)
df_train_marketing_p = preprocess_marketing(df_train_marketing)
df_test_marketing_p = preprocess_marketing(df_test_marketing)

# 5. 데이터 통합 (ID 기준 예시)
df_train = pd.merge(df_train_member_p, df_train_marketing_p, on=['ID', '기준년월'], how='left')
df_test = pd.merge(df_test_member_p, df_test_marketing_p, on=['ID', '기준년월'], how='left')

# 6. 스케일링 (수치형)
exclude_cols = ['Segment', 'ID', '기준년월']
num_cols = [col for col in df_train.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
scaler = StandardScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
df_test_num_cols = [col for col in num_cols if col in df_test.columns]
df_test[df_test_num_cols] = scaler.transform(df_test[df_test_num_cols])

# 7. 피처 엔지니어링 예시 (파생변수)
df_train['총카드신청수'] = df_train['카드신청건수'] + df_train.get('기타면제카드수_B0M', 0)
df_test['총카드신청수'] = df_test['카드신청건수'] + df_test.get('기타면제카드수_B0M', 0)

# 8. 머신러닝용 데이터셋 저장
# df_train.to_parquet(os.path.join(SAVE_DIR, 'train_final.parquet'))
# df_test.to_parquet(os.path.join(SAVE_DIR, 'test_final.parquet'))
df_train_member_p.to_parquet(os.path.join(SAVE_DIR, 'train_member_preprocessed.parquet'))
df_test_member_p.to_parquet(os.path.join(SAVE_DIR, 'test_member_preprocessed.parquet'))
df_train_marketing_p.to_parquet(os.path.join(SAVE_DIR, 'train_marketing_preprocessed.parquet'))
df_test_marketing_p.to_parquet(os.path.join(SAVE_DIR, 'test_marketing_preprocessed.parquet'))

# 9. 모델링 및 평가 (예시: RandomForest, F1 Score)
# 타겟 컬럼 예시: 'Segment' (실제 타겟에 맞게 수정)
# if 'Segment' in df_train.columns:
#     X = df_train.drop(['Segment', 'ID', '기준년월'], axis=1, errors='ignore')
#     y = df_train['Segment']
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
#     print('RandomForest 5-Fold F1 Score:', scores)
#     print('Mean F1:', np.mean(scores))

print('EDA & Preprocessing done!') 