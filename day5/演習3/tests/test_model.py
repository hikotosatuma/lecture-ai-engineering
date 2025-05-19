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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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


def test_model_regression_against_baseline_file(train_model):
    """
    現在のモデルの精度を、ファイルとして保存されたベースラインモデルの精度と比較する。
    共通の評価可能なサブセットで比較し、精度がベースラインを下回った場合にテストを失敗させる。
    """
    current_model, X_test, y_test = train_model

    # 1. 現在のモデルの精度を完全なテストセットで計算 (情報提供用)
    y_pred_current_full = current_model.predict(X_test)
    current_accuracy_full = accuracy_score(y_test, y_pred_current_full)
    print(f"INFO: 現在のPRモデルの精度 (完全テストセット): {current_accuracy_full:.4f}")

    # ベースラインモデルのパスを定義・ロード
    baseline_model_dir = os.path.join(os.path.dirname(__file__), "../../演習1/models")
    baseline_model_path = os.path.join(baseline_model_dir, "titanic_model.pkl")

    print(f"INFO: ベースラインモデルのパスを探索: {baseline_model_path}")

    if not os.path.exists(baseline_model_path):
        message = f"FAIL: ベースラインモデルファイルが見つかりません: {baseline_model_path}。PRの承認/停止判断ができません。"
        print(message)
        pytest.fail(message)

    try:
        with open(baseline_model_path, "rb") as f:
            baseline_model = pickle.load(f)
        print(f"INFO: ベースラインモデルを正常に読み込みました: {baseline_model_path}")
    except Exception as e:
        message = f"FAIL: ベースラインモデルの読み込み中にエラーが発生しました: {baseline_model_path}, Error: {e}"
        print(message)
        pytest.fail(message)

    # 2. ベースラインモデルが処理可能な共通サブセットを作成
    # 演習1のモデルは "Age", "Fare" の欠損値を除外して学習されたと仮定
    X_test_copy = X_test.copy()
    y_test_copy = y_test.copy()

    # "Age" と "Fare" が欠損していない行のインデックスを取得
    # (演習1の prepare_data で Pclass, Sex, Age, Fare, Survived の .dropna() を実施しているため、
    #  これらのうち欠損しうる Age, Fare を基準にする)
    common_valid_indices = X_test_copy[["Age", "Fare"]].dropna().index

    X_test_common_subset = X_test_copy.loc[common_valid_indices]
    y_test_common_subset = y_test_copy.loc[common_valid_indices]

    if X_test_common_subset.empty:
        print(
            "WARN: ベースラインモデルと比較可能な共通データサブセットが空です（Age/FareのNaNのため）。リグレッション比較をスキップします。"
        )
        pytest.skip("ベースラインモデルと比較するための共通データがありません。")
        return

    print(f"INFO: 完全テストセットのサンプル数: {len(X_test)}")
    print(
        f"INFO: ベースライン比較用共通サブセットのサンプル数: {len(X_test_common_subset)}"
    )

    # 3. 現在のモデルの精度を共通サブセットで計算
    try:
        y_pred_current_common = current_model.predict(X_test_common_subset)
        current_accuracy_common = accuracy_score(
            y_test_common_subset, y_pred_current_common
        )
        print(
            f"INFO: 現在のPRモデルの精度 (共通サブセット): {current_accuracy_common:.4f}"
        )
    except Exception as e:
        message = f"FAIL: 現在のモデルの共通サブセットでの評価中にエラー: {e}"
        print(message)
        pytest.fail(message)

    # 4. ベースラインモデルの精度を共通サブセットで計算
    try:
        # ベースラインモデルが期待する特徴量を選択
        baseline_features = ["Pclass", "Sex", "Age", "Fare"]
        X_baseline_input = X_test_common_subset[baseline_features].copy()

        # "Sex" 列をラベルエンコード (演習1の処理に合わせる)
        le = LabelEncoder()
        X_baseline_input["Sex"] = le.fit_transform(X_baseline_input["Sex"])

        # (演習1では Pclass, Sex, Age, Fare が float に変換されていたが、
        #  RandomForestClassifierはintとfloatの混在を通常許容するため、ここでは明示的な型変換は省略)

        y_pred_baseline_common = baseline_model.predict(X_baseline_input)
        baseline_accuracy_common = accuracy_score(
            y_test_common_subset, y_pred_baseline_common
        )
        print(
            f"INFO: ベースラインモデルの精度 (共通サブセット): {baseline_accuracy_common:.4f}"
        )
    except Exception as e:
        message = f"FAIL: ベースラインモデルの共通サブセットでの評価中にエラーが発生しました: {e}"
        print(message)
        pytest.fail(message)

    # 5. 共通サブセットでの精度を比較
    if current_accuracy_common >= baseline_accuracy_common:
        print(
            f"PASS: 新モデルの精度 (共通サブセット {current_accuracy_common:.4f}) はベースラインモデルの精度 (共通サブセット {baseline_accuracy_common:.4f}) 以上です。"
        )
    else:
        degradation = baseline_accuracy_common - current_accuracy_common
        message = f"STOP: 新モデルの精度 (共通サブセット {current_accuracy_common:.4f}) が、ベースラインモデルの精度 (共通サブセット {baseline_accuracy_common:.4f}) より {degradation:.4f} 低下しています。"
        print(message)
        assert current_accuracy_common >= baseline_accuracy_common, message
