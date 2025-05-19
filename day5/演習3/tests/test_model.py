import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_model_regression_against_baseline_file(train_model): # 関数名を変更
    """
    現在のモデルの精度を、ファイルとして保存されたベースラインモデルの精度と比較する。
    精度がベースラインを下回った場合にテストを失敗させ、ログに情報を出力する。
    """
    current_model, X_test, y_test = train_model
    
    # 現在のモデルの精度計算
    y_pred_current = current_model.predict(X_test)
    current_accuracy = accuracy_score(y_test, y_pred_current)
    print(f"INFO: 現在のプルリクエストのモデル精度: {current_accuracy:.4f}")

    baseline_model_dir = os.path.join(os.path.dirname(__file__), "../../演習1/models")
    baseline_model_path = os.path.join(baseline_model_dir, "titanic_model.pkl")
    
    print(f"INFO: ベースラインモデルのパスを探索: {baseline_model_path}")

    if not os.path.exists(baseline_model_path):
        # ベースラインモデルファイルが見つからない場合、テストをスキップまたは失敗させる
        # ここでは失敗させることで、ベースラインのセットアップ漏れに気づきやすくする
        message = f"FAIL: ベースラインモデルファイルが見つかりません: {baseline_model_path}。PRの承認/停止判断ができません。"
        print(message)
        pytest.fail(message)

    # ベースラインモデルの読み込みと精度計算
    try:
        with open(baseline_model_path, "rb") as f:
            baseline_model = pickle.load(f)
        print(f"INFO: ベースラインモデルを正常に読み込みました: {baseline_model_path}")
        
        y_pred_baseline = baseline_model.predict(X_test) # X_testは前処理前のデータ
        baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
        print(f"INFO: ベースラインモデルの精度: {baseline_accuracy:.4f}")

    except Exception as e:
        message = f"FAIL: ベースラインモデルの読み込みまたは評価中にエラーが発生しました: {e}。PRの承認/停止判断ができません。"
        print(message)
        pytest.fail(message)

    # 精度の比較と結果の判断
    if current_accuracy >= baseline_accuracy:
        print(
            f"PASS: 新モデルの精度 ({current_accuracy:.4f}) はベースラインモデルの精度 ({baseline_accuracy:.4f}) 以上です。PRは承認されます。"
        )
        # テストは成功 (assert True は不要、何もエラーが起きなければ成功)
    else:
        degradation = baseline_accuracy - current_accuracy
        message = (
            f"STOP: 新モデルの精度 ({current_accuracy:.4f}) がベースラインモデルの精度 ({baseline_accuracy:.4f}) より {degradation:.4f} 低下しています。PRは停止されます。"
        )
        print(message)
        assert current_accuracy >= baseline_accuracy, message 