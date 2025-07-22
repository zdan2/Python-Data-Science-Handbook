## 初級（基礎固め：Jupyter & NumPy）

> 対象章：Part Ⅰ 1–3 章、Part Ⅱ 4–12 章

1. Jupyter Notebook を起動し、新しいノートブックを作成せよ。
2. 1 行のセルに `print("Hello, Data Science!")` を書いて実行。
3. Markdown セルを追加し、`# 私の最初のノート` と見出しを書く。
4. `len?` を実行してヘルプを表示。
5. `np.mean??` でソースコードを確認。
6. Tab 補完で `numpy.random.default_rng` を補完。
7. `%lsmagic` で Magic コマンド一覧を表示。
8. `%timeit sum(range(1_000))` を計測。
9. `%run hello.py` で自作スクリプトを実行。
10. `%prun` で自作関数のプロファイルを取得。
11. Python リスト `[1,2,3]` を作り `type()` を確認。
12. 同じ要素で `np.array` を作り型を比較。
13. `arr = np.arange(10)` の `shape`, `dtype`, `ndim` を表示。
14. `np.zeros((10,10))` を生成。
15. `arr.reshape(2,5)` で形状変更。
16. `np.linspace(0,1,5)` を生成。
17. for ループで 1‒1e7 合計を計測。
18. 同じ計算を `np.sum` で行い速度比較。
19. `dtype='float32'` の乱数 100 万個を生成。
20. コピーとビューの挙動を確認。
21. 0–99 配列から 5:15 をスライス。
22. 10×10 配列の 1 行目を取得。
23. 同配列の 1 列目を取得。
24. 3×3 のサブ配列をスライス取得。
25. ビューを変更して元配列も変わることを確認。
26. `(2,3)` 要素を書き換え。
27. `arr[::-1]` で逆順。
28. `arr[:,::2]` で偶数列抽出。
29. 3 の倍数だけ抽出。
30. 奇数を ‑1 に置換。
31. `np.square(arr)` を計算。
32. `np.sin`, `np.cos`, `np.tan` を一括計算。
33. `np.exp(arr).sum()` を求める。
34. 乱数 100 万個の平均を計算。
35. 最大値と最小値の差を求める。
36. 2 次元配列で `axis=0` 合計。
37. `np.percentile` で 25/50/75 % 点。
38. `argsort` で昇順インデックス。
39. `partition` で上位 5 要素抽出。
40. `rng = np.random.default_rng(0)` で再現可能に。
41. 5×5 配列に `[0,1,2,3,4]` を足す。
42. 列方向に平均 0 のセンタリング。
43. メッシュグリッドで `z = sin(x)+cos(y)`。
44. ブロードキャストで距離行列を生成。
45. 形状 (3,1)+(1,4) の加算を試す。
46. 非互換形状で加算しエラーを観察。
47. RGB 画像 (H,W,3) に色補正のベクトルを加算。
48. 10×10 配列を行平均 0・列分散 1 に正規化。
49. `np.multiply.outer` で外積行列。
50. 3D テンソルにスカラーを掛ける。
51. 乱数 1000 個で `abs(x)>2` の個数を数える。
52. 2 配列の共通要素を論理演算で抽出。
53. 偶数番目要素を fancy‑index で取得。
54. ランダムに 10 行サンプリング。
55. 指定インデックスで抜き出し。
56. `np.isnan` を使い NaN を除外して平均。
57. `np.where` で符号を ±1 に置換。
58. `np.unique` でユニーク数を数える。
59. `np.in1d` で membership テスト。
60. `np.choose` で複数条件選択。
61. int 配列をソートし `%timeit` で測定。
62. 行ごとに降順ソート。
63. 構造化配列 (name,age,height) を作り name でソート。
64. 同配列を `age>30` でフィルタ。
65. `np.argsort` で上位 3 行を抽出。
66. `np.lexsort` で 2 キーソート。
67. `partition` で中央値を求める。
68. 配列を in‑place で逆順。
69. `mergesort` 指定で安定ソート確認。
70. `int8` と `int64` のソート速度比較。
71. `plt.plot` で線形データを描画。
72. タイトル・軸ラベルを追加。
73. 乱数 100 点で散布図。
74. bins=20 のヒストグラム。
75. サイズ・色をデータ依存で指定した散布図。
76. sin と cos を同図に描き凡例。
77. `errorbar` で平均±標準偏差。
78. 2×2 の `subplots` 配置。
79. `savefig` で PNG 保存。
80. 箱ひげ図を作成。
81. `ZeroDivisionError` を発生させ `%xmode` で表示形式変更。
82. `%debug` で対話的デバッグ。
83. `%lprun` で行レベルプロファイル。
84. `%memit` でメモリ計測。
85. 例外トレースを簡略表示。
86. `logging` で debug メッセージを出力。
87. `warnings` モジュールでカスタム警告。
88. `try/except` でファイル読込エラー処理。
89. `contextmanager` でタイマーを実装。
90. `!pip list` でパッケージ確認。
91. Python vs NumPy の速度差をグラフ化。
92. ランダムウォークをシミュレーションし分布を描画。
93. 配列演算でフィボナッチ 100 項を生成。
94. 画像の R チャンネルヒストグラムを描く。
95. 2D ガウス分布 1 万点を散布 + 等高線。
96. モンテカルロで円周率を推定。
97. 100×100 行列積を `%timeit`。
98. Broadcasting と mask でチェス盤模様を生成。
99. NumPy でパスカルの三角形 10 行を作成。
100. CSV を読み込み NumPy だけで平均を計算。

---

## 中級（実務応用：pandas & 可視化）

> 対象章：Part Ⅲ 13–24 章、Part Ⅳ 25–36 章

1. `Series` を `{'a':1,'b':2,'c':3}` から作成し index 指定。
2. dict of lists から `DataFrame` を作り列順を並べ替え。
3. 乱数 10×3 で `DataFrame` を作成し列名を `x,y,z`。
4. `planets` データを `read_csv` し `head()`。
5. `info()` で概要確認。
6. `'year'` 列を `astype('int32')`。
7. `'name'` を `set_index`。
8. `reset_index(drop=True)` で戻す。
9. `insert(0,'ones',1)` で列を挿入。
10. `describe().T` で転置表示。
11. `loc` で `'Mars'` 行抽出。
12. `iloc` で最初の 3 行 2 列目。
13. `mass>1e24` の行を抽出。
14. `'name'.isin(['Earth','Venus'])` で抽出。
15. `loc` で複数列抽出。
16. `at`,`iat` でスカラー取得。
17. `query('distance<1e8')` で抽出。
18. `eval('density=mass/volume')`。
19. 空列を追加し欠損値を生成。
20. `dropna` & `fillna` で処理。
21. 異なる `index` を持つ 2 Series を加算。
22. `add(..., fill_value=0)` で欠損を 0 扱い。
23. 同名列同士の DataFrame 加算。
24. 行方向の引き算 (`axis` 指定)。
25. `div`,`mul`,`rsub` を体験。
26. `broadcast_like` を使う。
27. `reindex` で行追加。
28. `combine_first` で重ね合わせ。
29. `clip(lower,upper)` で値制限。
30. `pipe` で関数チェーン。
31. `isnull` / `notnull` で欠損検出。
32. `interpolate('linear')` で補間。
33. `where` / `mask` で条件置換。
34. `dropna(thresh=3)` で行除去。
35. `assign` + `fillna(method='ffill')`。
36. `MultiIndex.from_product` で年×曜日。
37. `unstack` → `stack` を往復。
38. `swaplevel` で順序交換。
39. `sort_index(level=1, ascending=False)`。
40. `xs` でクロスセクション抽出。
41. `pd.concat` で縦結合。
42. `keys` 付き concat で階層キー付与。
43. `merge(left, right, how='left', on='key')`。
44. 多対 1 結合。
45. 多対多結合 & `validate`。
46. `join` で index 結合。
47. concat で 1 行追加（`append` 廃止）。
48. `combine_first` で穴埋め。
49. `indicator=True` で出所確認。
50. `suffixes=('_L','_R')` で重複解消。
51. `groupby('method').mean()`。
52. `groupby(['method','year']).size()`。
53. `agg({'mass':'mean','radius':'max'})`。
54. `transform` で z‑score。
55. `filter` で条件に合うグループのみ。
56. `apply(lambda d: d.head(2))` 各グループ先頭。
57. `nunique()` を計算。
58. `cov()` / `corr()` をグループで。
59. `rolling(window=7).mean()`。
60. `resample('M').sum()`。
61. `pivot_table(values='mass', index='method', columns='year')`。
62. `margins=True` で総計。
63. `fill_value=0` で欠損埋め。
64. 多重インデックス版 pivot\_table + `style.background_gradient()`。
65. `melt` で wide→long 変換。
66. `str.contains('Mars')` 行を抽出。
67. `str.replace('[^A-Za-z]','',regex=True)`。
68. `str.split(expand=True)` で列分割。
69. `str.len()>5` でフィルタ。
70. `extract(r'(\\d+)')` で数字抽出。
71. `pd.date_range('2020-01-01', periods=365)`。
72. 乱数シリーズを `cumprod` し日次グラフ。
73. `shift(1)` で前日比。
74. `tz_localize('UTC').tz_convert('Asia/Tokyo')`。
75. `between_time('09:00','17:00')` で営業時間。
76. `rolling('30D').mean()`。
77. `asfreq('W', method='ffill')`。
78. `period_range('2022Q1','2023Q4',freq='Q')`。
79. `resample('A').agg(['min','max'])`。
80. 移動平均を重ねて描画。
81. `query('mass>1e24 & year<1990')`。
82. `eval('density=mass/volume', inplace=True)`。
83. `query` で平均超過行を抽出。
84. 大量 DataFrame で `eval` の速度測定。
85. numexpr 無効時と比較。
86. 3×1 サブプロットに sin,cos,tan。
87. `set_xlim`,`set_ylim` で範囲指定。
88. `twinx` で二軸グラフ。
89. `hist2d` で 2D ヒストグラム。
90. `hexbin` で密度可視化。
91. カスタム `colorbar`。
92. `scatter` にエラーバー追加。
93. 凡例を 2 か所に配置。
94. `GridSpec` で複雑配置。
95. `annotate` でピークに注釈。
96. `np.memmap` で大容量ファイル読み込み。
97. `np.vectorize` と純 Python 関数の速度比較。
98. `np.einsum` で行列演算。
99. `numba.jit` で ufunc を高速化。
100. `np.broadcast_to` で無コピー確認。

---

## 上級（発展応用：機械学習 & モデル運用）

> 対象章：Part Ⅴ 37–47 章

1. `load_iris` → `train_test_split` で 8:2 分割。
2. `StandardScaler` を組み込んだ `Pipeline` を構築。
3. `KNeighborsClassifier` の k=1–15 を交差検証で比較。
4. `classification_report` を表示。
5. `confusion_matrix` を heatmap 化。
6. `GridSearchCV` でハイパラ探索。
7. 前処理→モデル→後処理を `Pipeline` で一体化。
8. ROC 曲線を描画し AUC を算出。
9. `SMOTE` でクラス不均衡を補正。
10. `joblib.dump` / `load` でモデル永続化。
11. `LinearRegression` で Boston Housing を回帰。
12. 係数を特徴名と対で表示。
13. 5‑fold CV の R² 平均を計算。
14. `PolynomialFeatures(2)` で非線形回帰。
15. `Lasso` / `Ridge` の α を変化させ学習曲線。
16. `ElasticNet(l1_ratio=0.5)` を比較。
17. `statsmodels` で OLS 回帰 & `summary()`。
18. Cook 距離で外れ値影響を調査。
19. PDP で部分依存を描写。
20. SHAP で特徴寄与を解析。
21. `LogisticRegression` で digits 多クラス分類。
22. `predict_proba` のヒストグラムを描く。
23. 正則化 C を logspace で評価。
24. 正規化 `confusion_matrix` を表示。
25. Stratified 10‑fold CV を実施。
26. しきい値を変えて F1 を最大化。
27. `VotingClassifier` で多数決アンサンブル。
28. `CalibratedClassifierCV` で確率校正。
29. `permutation_importance` で重要度計算。
30. 交互作用専用 `PolynomialFeatures(interaction_only=True)`。
31. `SVC(kernel='linear')` を適用。
32. マージン幅と support vector を可視化。
33. `SVC(kernel='rbf')` の γ・C をグリッドサーチ。
34. `OneClassSVM` で異常検知。
35. `LinearSVC` と速度比較。
36. `decision_function` スコアを取得。
37. Platt scaling で確率校正。
38. `Pipeline` で `StandardScaler`→`SVC`。
39. `SVR` で回帰し ε を調整。
40. `make_scorer` でカスタム評価指標。
41. `DecisionTreeClassifier` 深さ制限で過学習検証。
42. `export_text` で木構造を表示。
43. `export_graphviz`→Graphviz で可視化。
44. `RandomForestRegressor` で重要度バーグラフ。
45. `ExtraTreesClassifier` と OOB スコア比較。
46. `GradientBoostingClassifier` の learning\_rate 評価。
47. `HistGradientBoostingRegressor` を試す。
48. XGBoost / LightGBM と比較。
49. `feature_importances_` と SHAP を比較。
50. PDP 相互作用プロットを描く。
51. `PCA` で digits を 2 次元に削減し散布。
52. `explained_variance_ratio_` の累積和を描画。
53. `n_components` を変え再構築誤差を比較。
54. `IncrementalPCA` でバッチ処理。
55. `KernelPCA(rbf)` で非線形展開。
56. `TruncatedSVD` で TF‑IDF 行列を削減。
57. `TSNE` で可視化。
58. `UMAP` で高次元可視化。
59. `Isomap` で顔画像を学習。
60. Keras Autoencoder で表現学習。
61. `KMeans(10)` で digits をクラスタし ARI を計算。
62. `GaussianMixture` で soft cluster。
63. `DBSCAN` で密度クラスタリング。
64. `AgglomerativeClustering` でデンドログラム。
65. `SpectralClustering` を適用。
66. `silhouette_score` で最適クラスタ数。
67. `Birch` でオンラインクラスタ。
68. `HDBSCAN` で階層密度クラスタ。
69. `MiniBatchKMeans` で大規模データ。
70. t‑SNE 空間にクラスタ結果を重ね描く。
71. `KFold` でスコアをループ計算。
72. `RepeatedStratifiedKFold` で分散確認。
73. `TimeSeriesSplit` で時系列検証。
74. `learning_curve` でデータ量 vs スコア。
75. `validation_curve` でパラメータ vs スコア。
76. `permutation_test_score` で有意差検定。
77. `HalvingGridSearchCV` で効率探索。
78. `RandomizedSearchCV` を試す。
79. `BayesSearchCV` (scikit‑optimize) でベイズ最適化。
80. `cross_validate` で複数メトリクス取得。
81. ROC と PR 曲線を 1 枚に描画。
82. 2‑D decision boundary を複数モデル比較。
83. 特徴重要度を横棒グラフで比較。
84. 誤分類に矢印で注釈。
85. SHAP summary プロット生成。
86. 3D PDP 相互作用プロット。
87. 学習曲線 (train vs val) を描画。
88. Matplotlib で scatter‑matrix (pairplot 代替)。
89. レーダーチャートでメトリクス比較。
90. SHAP waterfall で個別説明。
91. Kaggle Titanic データでフルパイプライン構築。
92. 自作 CSV → 前処理→モデル→評価を自動化。
93. `Flask` API で joblib モデルを提供。
94. `dask-ml` で並列 GridSearch。
95. `skorch` で PyTorch MLP を scikit‑learn API 化。
96. `onnxruntime` で ONNX 推論。
97. 自作スコアを `cross_val_score` に適用。
98. Demographic parity で公平性指標を計算。
99. Dash で explainability ダッシュボードを構築。
100. MLflow で実験管理＆メトリクス追跡。

---

### 使い方のヒント

* **段階的に取り組む**：初級 → 中級 → 上級の順に進めると、自然に知識が積み上がります。
* **ノートブック推奨**：実験・検証の過程を Markdown とコードで残すと学習効果が高まります。
* **公式ドキュメント併用**：書籍と scikit‑learn / pandas / NumPy の公式ドキュメントを参照しながら進めてください。

これで 3 レベル × 約 100 題、合計 300 題の課題集が完成です。学習の進捗に合わせて活用してください。
