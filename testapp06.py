""" Streamlitによる退職予測AIシステムの開発
"""

from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier


# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):
    """
    Streamlitでデータフレームを表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム

    Returns
    -------
    なし
    """

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df)

    # 参考：Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_graph(df: pd.DataFrame, x_col : str):
    """
    Streamlitでグラフ（ヒストグラム）を表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム
    x_col : str
        対象の列名（グラフのx軸）

    Returns
    -------
    なし
    """

    fig, ax = plt.subplots()    # グラフの描画領域を準備
    plt.grid(True)              # 目盛線を表示する

    # グラフ（ヒストグラム）の設定
    sns.countplot(data=df, x=x_col, ax=ax)

    st.pyplot(fig)              # Streamlitでグラフを表示する


def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int,
    ML_tec_choice: str) -> list:
    """
    決定木で学習と予測を行う関数
    
    Parameters
    ----------
    X : pd.DataFrame
        説明変数の列群
    y : pd.Series
        目的変数の列
    depth : int
        決定木の深さ

    Returns
    -------
    list: [学習済みモデル, 予測値, 正解率]
    """

    if ML_tec_choice == "決定木":
        # 決定木モデルの生成（オプション:木の深さ）
        clf = DecisionTreeClassifier(max_depth=depth)
    elif ML_tec_choice == "ランダムフォレスト":
        # ランダムフォレストの学習モデルを作成（（オプション:木の深さ）
        clf = RandomForestClassifier(max_depth=depth)
    elif ML_tec_choice == "LightGBM":
        # LightGBMの学習モデルを作成（（オプション:木の深さ）
        import lightgbm as lgb
        #pip install lightgbm
        clf = lgb.LGBMClassifier(max_depth=depth)
    else:
        st.write("学習手法が選択されていないため：決定木で予測します")
        clf = DecisionTreeClassifier(max_depth=depth)

    # train_test_split関数を利用してデータを分割する
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, train_size=0.7, random_state=0, stratify=y)

    # 訓練用データをオーバーサンプリング（水増し）をする
    oversample = SMOTE(sampling_strategy=0.5, random_state=0)
    train_x_over, train_y_over = oversample.fit_resample(train_x, train_y)

    # 標準化に必要なライブラリのインポートと準備
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()

    # 訓練用データの標準化
    std_x = stdsc.fit_transform(train_x_over)
    train_x_std = pd.DataFrame(std_x)

    # 検証用データの標準化
    std_x = stdsc.fit_transform(valid_x)
    valid_x_std = pd.DataFrame(std_x)

    # 学習
    clf.fit(train_x_std, train_y_over)

    # 予測：訓練用データで予測
    pred = clf.predict(train_x_std)
    y_train_true = pd.get_dummies(train_y_over, drop_first=True)
    y_train_true = y_train_true['Yes'] # 型をSeriesに変換
    y_train_pred = pd.get_dummies(pred, drop_first=True)
    y_train_pred = y_train_pred['Yes'] # 型をSeriesに変換

    # 予測：検証用データで予測
    dt_pred = clf.predict(valid_x_std)
    y_true = pd.get_dummies(valid_y, drop_first=True)
    y_true = y_true['Yes'] # 型をSeriesに変換
    y_pred = pd.get_dummies(dt_pred, drop_first=True)
    y_pred = y_pred['Yes'] # 型をSeriesに変換


    return [clf, y_train_true, y_train_pred, y_true, y_pred]


def st_display_dtree(clf, features):
    """
    Streamlitで決定木のツリーを可視化する関数
    
    Parameters
    ----------
    clf : 
        学習済みモデル
    features :
        説明変数の列群

    Returns
    -------
    なし
    """

    # 必要なライブラリのインポート    
    from sklearn.tree import plot_tree

    # 可視化する決定木の生成
    plot_tree(clf, feature_names=features, class_names=True, filled=True)

    # Streamlitで決定木を表示する
    st.pyplot(plt)

    # # 可視化する決定木の生成
    # dot = tree.export_graphviz(clf, 
    #     # out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
    #     # filled=True,    # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
    #     # rounded=True,   # Trueにすると、ノードの角を丸く描画する。
    # #    feature_names=['あ', 'い', 'う', 'え'], # これを指定しないとチャート上で特徴量の名前が表示されない
    #     # feature_names=features, # これを指定しないとチャート上で説明変数の名前が表示されない
    # #    class_names=['setosa' 'versicolor' 'virginica'], # これを指定しないとチャート上で分類名が表示されない
    #     # special_characters=True # 特殊文字を扱えるようにする
    #     )

    # # Streamlitで決定木を表示する
    # st.graphviz_chart(dot)



def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("退職予測AI\n（Maschine Learning)")

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'データ確認':

        # ファイルのアップローダー
        uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # データフレームをセッションステートに退避（名称:df）
                st.session_state.df = copy.deepcopy(df)

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                st_display_table(df.head(int(cnt)))

        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '要約統計量':

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 要約統計量の表示
            st.subheader("データの確認")
            st.write(df.describe())

            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == 'グラフ表示':

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # グラフ列の選択(サイドバー)
            activities_graph = df.columns.tolist()
            choice_col = st.sidebar.selectbox("グラフのx軸", activities_graph)

            # グラフの表示
            st.write(st_display_graph(df, choice_col))

            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '学習と検証':

        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 説明変数と目的変数の設定
            train_X = df.drop("退職", axis=1)   # 退職列以外を説明変数にセット
            train_Y = df["退職"]               # 退職列を目的変数にセット(型をSeriesに変換しておく)

            #決定木の深さを選択するスライダー
            depth=st.sidebar.number_input('決定木の深さ(サーバーの負荷軽減の為 Max=3)',1,3,2)

            #学習の手法を選択するスライダー
            ML_tec = ["決定木", "ランダムフォレスト", "LightGBM"]
            ML_tec_choice = st.sidebar.selectbox("学習の手法", ML_tec)

            # 決定木による予測
            clf, y_train_true, y_train_pred, y_true, y_pred = ml_dtree(train_X, train_Y, depth, ML_tec_choice)

            # 決定木のツリーを出力
            if ML_tec_choice == "決定木":
                st.caption('')
                st.caption('決定木の可視化')
                st_display_dtree(clf, train_X.columns)
            else:
                st.caption('')
                st.caption('特徴量重要度の可視化')
                feature_importances = clf.feature_importances_
                importances = pd.DataFrame({"features":train_X.columns, "importances" : feature_importances})
                importances_sort = importances.sort_values(by="importances",ascending = True)
                fig = plt.figure(figsize=(30, 30))
                ax = fig.add_subplot(1,1,1)
                ax.set_title("feature inportances")
                ax.tick_params(axis = "y",labelsize =30)
                x_position = np.arange(len(importances_sort["features"]))
                ax.barh(x_position, importances_sort["importances"], tick_label=importances_sort["features"])
                fig.tight_layout()
                st.pyplot(plt)


            # 正解率を出力
            st.subheader("訓練用データでの予測精度")
            st.caption("AIの予測が「全員、退職しない」に偏った場合は（意味がないので）全ての精度は0で表示します")
            st.write(f'正解率:{accuracy_score(y_train_true, y_train_pred)}')
            st.write(f'再現率:{recall_score(y_train_true, y_train_pred)}')
            st.write(f'適合率:{precision_score(y_train_true, y_train_pred)}')

            st.subheader("検証用データでの予測精度")
            st.write(f'正解率:{accuracy_score(y_true, y_pred)}')
            st.write(f'再現率:{recall_score(y_true, y_pred)}')
            st.write(f'適合率:{precision_score(y_true, y_pred)}')
            

        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == 'About':

        image = Image.open('logo_sato.png')
        st.image(image)
        st.write("Built by[wiz AIシステム科　佐藤光]")
        st.write("Version1.0")#14/15
        st.write("For More Information check out(https://wiz.ac.jp/)")


        

if __name__ == "__main__":
    main()
