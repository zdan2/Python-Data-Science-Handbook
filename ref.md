# np.argsort とは

`np.argsort` は **NumPy 配列を「並べ替えたときの添え字（インデックス）」を返す** 関数です。
実際に配列の要素を並べ替えるのではなく、「どの順序で並べ替えれば昇順（既定）になるか」を教えてくれます。

---

## 基本的な使い方

```python
import numpy as np

a = np.array([30, 10, 20])
idx = np.argsort(a)     # → array([1, 2, 0])
sorted_a = a[idx]       # → array([10, 20, 30])
```

* `idx` は `[1, 2, 0]`。

  * 元配列 `a` を「1 → 2 → 0」の順に読むと昇順になります。
* 取得したインデックスを元に `a[idx]` とすれば、実際に並べ替えた配列を得られます。

---

## シグネチャ

```python
numpy.argsort(a, axis=-1, kind=None, order=None)
```

| 引数      | 説明                                                                             |
| ------- | ------------------------------------------------------------------------------ |
| `a`     | ソート対象の配列                                                                       |
| `axis`  | どの軸で並べ替えるか（既定は ‑1 = 最後の軸）                                                      |
| `kind`  | ソートアルゴリズム<br>`'quicksort'`（既定）, `'mergesort'`（安定ソート）, `'heapsort'`, `'stable'` |
| `order` | 構造化配列でフィールド名を指定すると、多段ソートができる                                                   |

---

## 2 次元以上の配列

```python
b = np.array([[8, 1, 4],
              [3, 0, 6]])

# 列ごと（axis=0）に昇順へ
idx_col = np.argsort(b, axis=0)
# ↓列 0: [1,0], 列 1: [1,0], 列 2: [0,1]
#    [[1 1 0]
#     [0 0 1]]

# 行ごと（axis=1）に昇順へ
idx_row = np.argsort(b, axis=1)
# ↓各行の並べ替え順
#    [[1 2 0]
#     [1 0 2]]
```

行・列どちらの並べ替え順か、`axis` で指定します。結果の形状は元配列と同じです。

---

## 降順（逆順）で並べ替えたい場合

`argsort` 自体は常に昇順インデックスを返します。
降順が欲しいときは **インデックスを後ろから読めば OK** です。

```python
idx_desc = np.argsort(a)[::-1]
a[idx_desc]        # → array([30, 20, 10])
```

---

## ランキング・順位付けに便利

### 1. ランク（小さい順に 0,1,2,…）

```python
rank = np.empty_like(idx)
rank[idx] = np.arange(len(a))
# rank → array([2, 0, 1])
```

### 2. 同点を同順位にしたい（安定ソート）

```python
score = np.array([90, 80, 90, 70])
idx_stable = np.argsort(score, kind='mergesort')  # 安定
```

---

## 上位 k 件だけ取り出す（Top‑k イディオム）

```python
k = 3
topk_idx = np.argsort(a)[-k:]        # 昇順 → 後ろから k 件
topk_sorted = a[topk_idx][::-1]      # 値を大きい順に
```

`k` が小さければ `np.argpartition` を使う方が高速です。

---

## `np.sort` との違い

|     | np.sort    | np.argsort                |
| --- | ---------- | ------------------------- |
| 返り値 | 値を並べ替えた配列  | 並べ替え順のインデックス              |
| メモリ | 並べ替えた分だけ追加 | インデックス分だけ追加               |
| 用途  | 値が欲しいとき    | 順位・ペアリング・複数配列を同じ順に整列したいとき |

---

## 複数配列を同じ順序で並べ替える例

```python
names  = np.array(['Alice', 'Bob', 'Charlie'])
scores = np.array([  85 ,   92 ,      78      ])

idx = np.argsort(scores)[::-1]  # 得点の降順
names_sorted  = names[idx]
scores_sorted = scores[idx]
```

---

## 計算量とアルゴリズム

* **平均計算量**: `O(N log N)`（クイックソートが既定）
* `kind='mergesort'` か `'stable'` を選ぶと **安定ソート**（同値の相対順が保存）
* 大規模データで「部分的ソート」で十分なら `argpartition` が `O(N)` 近くで高速

---

## よくある落とし穴

1. **浮動小数点の ±0、NaN**

   * `np.sort` と同様に `NaN` は末尾へ。必要なら `np.nanargmin / np.nanargmax` を検討。
2. **複数キーでソートしたい**

   * 構造化配列 + `order` か、タプルをキーに `np.lexsort`。
3. **ブール配列をそのまま使わない**

   * `True/False` は 1/0 扱いなので注意。

---

## まとめ

* **`np.argsort` は「並べ替え順だけ知りたい」場面で必須**。
* 多配列を同一順序に揃える、順位計算、Top‑k 抽出などで威力を発揮。
* 安定性が必要なら `kind='mergesort'`（または `'stable'`）を指定。
* 降順はインデックスを逆順に読む、部分ソートは `argpartition` が高速。

これで `np.argsort` の基本から応用まで押さえられます。ぜひ手元で試してみてください！


# np.lexsort とは

`np.lexsort` は **複数キー（多段階キー）による辞書式（レキシコグラフィック）ソート順のインデックス** を返す NumPy の関数です。
「姓で並べ、同姓なら名で並べ、同名なら年齢で…」といった **優先順位付きソート** を高速に行えます。

---

## シグネチャ

```python
numpy.lexsort(keys, axis=-1)
```

| 引数     | 説明                                                                       |
| ------ | ------------------------------------------------------------------------ |
| `keys` | **タプルまたは 2 D 配列**<br>各要素は同じ長さ 1 D 配列（または同じ形）で、「並べ替えのキー列」を表す。**後述の順序に注意** |
| `axis` | `keys` を 2 D 配列で渡した場合のみ使用。どの軸をキー列とみなすか（既定 -1 = 最後の軸）                     |

**返り値**: 1 D 整数配列（並べ替え順インデックス）
`np.argsort` の返り値と同じ形で、これを用いて元配列を並べ替えられます。

---

## 重要ポイント 1 — 「キーの優先順位は **最後のキーが最優先**」

他言語の `sort(key1, key2, …)` と逆なので注意してください。

```python
keys = (first_names, last_names)
idx  = np.lexsort(keys)  # last_names → first_names の順で比較
```

`keys = (key0, key1, …, keyN)`

1. **keyN（最後）** で比較
2. 同値なら keyN‑1
3. …
4. 最終的に key0

「姓→名」のように直感的な順にしたい場合は **タプルを逆順で渡す** か、`[::-1]` で反転させます。

---

## 例 1：姓→名の文字列ソート

```python
import numpy as np

last  = np.array(['Yamada', 'Sato', 'Yamada', 'Suzuki'])
first = np.array(['Taro'  , 'Hanako', 'Ichiro', 'Taro'  ])

idx = np.lexsort((first, last))  # (名, 姓) の順で比較
print(idx)        # [1 3 2 0]

print(last[idx])  # ['Sato' 'Suzuki' 'Yamada' 'Yamada']
print(first[idx]) # ['Hanako' 'Taro'   'Ichiro' 'Taro' ]
```

---

## 例 2：表（2 D 配列）の列キーでソート

```python
data = np.array([[201,  90, 1],   # ID, score, age
                 [105,  92, 2],
                 [105, 100, 1],
                 [201,  90, 3]])

# 「ID 昇順」→「score 降順」→「age 昇順」
id_key     = data[:, 0]
score_key  = -data[:, 1]   # 降順にしたい列は符号を反転
age_key    = data[:, 2]

idx = np.lexsort((age_key, score_key, id_key))
sorted_data = data[idx]
```

---

## 例 3：`axis` を使って行単位／列単位で

2 D 配列をそのまま渡すと **axis=0 で「列をキー、行をレコード」としてソート**、
`axis=1` を指定すれば行をキーに列をレコードとして扱えます。

```python
A = np.array([[ 3, 1, 2],    # shape (3,3)
              [ 2, 2, 2],
              [ 3, 0, 1]])

idx_rows = np.lexsort(A, axis=0)  # 列キー → 行番号
# 各列をキーと見なして行インデックスを返す

idx_cols = np.lexsort(A, axis=1)  # 行キー → 列番号
# 各行をキーと見なして列インデックスを返す
```

---

## 実装・計算量

* 内部的には **安定マージソート（`mergesort`）** を複数回走らせる実装

  * 同値のキーで元の順序が保たれる
* キー数を K、要素数を N とすると **O(K · N log N)**

  * キーが数本なら `argsort` をネストするより高速・メモリ効率も良い

---

## np.lexsort と他手法の比較

| 用途                              | 向き／不向き                 | 備考                            |
| ------------------------------- | ---------------------- | ----------------------------- |
| **`np.lexsort`**                | 複数独立配列をキーにした多段ソート      | キーの優先順は最後が最優先（要注意）            |
| 構造化配列 + `np.argsort(order=[…])` | 同じレコード構造に紐付くデータ        | 可読性は高いがデータを構造化し直すコスト          |
| `np.argsort` を多重に適用             | 1〜2 キー程度・キー優先順を手動制御したい | `stable` を使えば安定ソート可、ただしコードが冗長 |
| pandas の `sort_values`          | 表形式でラベル付き列を扱う          | 機能豊富だが pandas 依存              |

---

## よくある落とし穴 & 対策

1. **「キー順が逆」**

   * `np.lexsort((secondary, primary))` のように **タプルの並びを逆転**。
2. **降順キー**

   * 値の符号反転、または補数演算（整数なら `np.max(key) - key`）で対処。
3. **長さの不一致**

   * 各キーは必ず同じ長さ（または同形）にする。
4. **NaN**

   * 浮動小数点キーに NaN があると比較結果が未定義。`np.nan_to_num` や `np.isnan` マスクで処理。
5. **大量キー**

   * K が大きいならキーを一つにまとめて構造化配列＋`argsort` の方がキャッシュ効率が良い場合も。

---

## 応用テクニック

### A. 同順位（タイ）をグループ分け

```python
keys = (age, score)              # 例：age→score 順
idx = np.lexsort(keys)
sorted_keys = [k[idx] for k in keys]

# 先頭との差分で「グループ境界」を検出して同順位をまとめる、など
```

### B. Top‑k を複数キーで

```python
k = 5
idx = np.lexsort((sec, pri))[-k:][::-1]   # 優先：pri→sec／降順抽出
top_values = values[idx]
```

### C. 部分キーを動的に選択

キーリストを作って `np.lexsort(tuple(keys[::-1]))` とすれば、
ユーザー指定の優先順（第 1→第2→…）を直接渡せる。

---

## まとめ

* **`np.lexsort` は NumPy で「複数キーの辞書式ソート」を最も簡潔・高速に行う標準手段**。
* **最後に渡したキーが最優先** という独特の仕様に注意。
* 降順は符号反転・補数化で、タイの安定性は保証済み。
* キー数が少なく、構造化配列にしたくない場合に特に威力を発揮します。
* 多段ソートで毎回 `argsort` をネストしていた場合はぜひ `np.lexsort` に置き換えてみてください。

# np.partition とは  

`np.partition` は **配列の特定位置（`kth`）を基準に「前半を小さく、後半を大きく」二分 partition した結果を返す** NumPy の関数です。
完全な並べ替え（`np.sort`）は行わず、**「k 番目に小さい（または大きい）要素だけ順序が保証されればよい」** 場面で *O(N)* 近い高速処理が可能です。

---

## シグネチャ

```python
numpy.partition(a, kth, axis=-1, kind='introselect', order=None)
```

| 引数      | 説明                                                                       |
| ------- | ------------------------------------------------------------------------ |
| `a`     | 配列（array‑like）                                                           |
| `kth`   | **0‑始まりインデックス or その配列**<br>「ここが確定してほしい順位」を指定。複数指定可。                      |
| `axis`  | どの軸で partition するか（既定 ‑1 = 最後の軸）                                         |
| `kind`  | アルゴリズム（既定 `'introselect'` = Quick‑select 改良版）。 `'introselect'` しか許容されます。 |
| `order` | 構造化配列で並べ替えフィールドを指定するときに使用                                                |

**返り値**: `a` と同形・同 dtype の新配列（コピー）。
**計算量**: 平均 *O(N)*、最悪 *O(N log N)* 付近。

---

## 基本動作

```python
import numpy as np
a = np.array([30, 10, 50, 20, 40])
part = np.partition(a, kth=2)   # 2 番目 (0,1,**2**,3,4)
print(part)        # [10 20 30 50 40]
```

* 位置 `k=2` の値 **30** は、**配列を完全昇順に並べ替えても同じ位置** に来ます。
* **左側** (`k` 未満) は **「30 以下」**、右側は **「30 以上」** が保証されるだけで、**内部順序は未定**。

---

## axis と複数 kth

```python
B = np.array([[8, 1, 4],
              [3, 7, 2]])
# 行ごと (axis=1) に「最小と 2 番目」(k=0,1) を確定
part = np.partition(B, kth=[0,1], axis=1)
# 結果例
# [[1 4 8]   ← 0,1 列が小さい 2 要素
#  [2 3 7]]
```

複数 `kth` は **昇順に並べ替えて渡さなくても OK**（内部で整列して処理）。

---

## np.argpartition との違い

|          | `np.partition`           | `np.argpartition`           |
| -------- | ------------------------ | --------------------------- |
| **返り値**  | 値が入った配列                  | インデックス配列                    |
| **利用場面** | 値そのものが欲しい<br>（中央値・分位数など） | 上位 k 件の位置を使い、別配列を同じ順で取りたいとき |

例：上位 k 件を値付きで取り出すなら

```python
k = 3
idx = np.argpartition(a, -k)[-k:]   # 上位 k 件の添え字（順不同）
topk = a[idx]                       # 実際の値
```

---

## 代表的ユースケース

| 目的                       | イディオム                                                          |
| ------------------------ | -------------------------------------------------------------- |
| **中央値**                  | `median = np.partition(a, len(a)//2)[len(a)//2]`               |
| **p タイル (分位数)**          | `q = 0.9; k = int(q*len(a)); val = np.partition(a, k)[k]`      |
| **Top‑k / Bottom‑k**     | `top_idx = np.argpartition(a, -k)[-k:]` → 必要に応じて `np.sort` で整列 |
| **外れ値検出** (例：最小/最大数件を無視) | 部分 partition して端の要素を除外                                         |

---

## 内部アルゴリズム

* **Introselect**: Quick‑select をベースに、再帰深度が増えたらヒープセレクトへ切替え → **最悪時の時間増大を抑止**。
* **安定ではない**: 同じ値の相対順は保持されません。安定性が要る場合は `np.sort(kind='mergesort')` が必要。

---

## パフォーマンス比較 (概観)

| データ件数           | 要求                                               | 最速の可能性                                         |
| --------------- | ------------------------------------------------ | ---------------------------------------------- |
| **小さめ (≤1e4)**  | 完全ソート                                            | `np.sort` でも十分                                 |
| **中〜大 (≥1e5)**  | 一部順位だけ                                           | `np.partition / np.argpartition` が優位 (×2〜10 倍) |
| **Top‑k がごく少数** | `np.argpartition` → `take_along_axis` + `sort`   |                                                |
| **pandas 使用時**  | `Series.nsmallest`, `nlargest` が内部で argpartition |                                                |

---

## よくある落とし穴 & 回避策

1. **「戻り値が順不同」**

   * Top‑k を「大きい順に見たい」→ 取り出した部分を `np.sort` で再整列するか、`[::-1]` で逆順。
2. **降順 partition**

   * `kth` を負値で指定 (`-k`) すると「大きい側から k」番目基準にできます。

     ```python
     np.partition(a, kth=-1)  # 最大値が並ぶ
     ```
3. **NaN の存在**

   * `np.nanpartition` はない → 事前に `np.nan_to_num`、またはマスク除外。
4. **構造化配列**

   * `order` でフィールド指定可。ただし複数キーで部分選択を行うユースケースは稀。
5. **非常に大きい k (\~N/2)**

   * k が配列サイズに近いなら、結局大半を触るため `np.sort` と速度差が縮まる。

---

## 応用テクニック

### A. 複数配列の一致 Top‑k

```python
scores = np.random.rand(100000)
ids    = np.arange(100000)

k = 100
idx = np.argpartition(scores, -k)[-k:]     # 上位 k 件添え字
top_scores = scores[idx]
top_ids    = ids[idx]

# 表示用に整えておく
order = np.argsort(-top_scores)            # 降順インデックス
top_scores = top_scores[order]
top_ids    = top_ids[order]
```

### B. Axis 別に分位値ベクトルを一括計算

```python
p = 0.25
k = int(p * arr.shape[1])                  # 列ごとの kth
q1 = np.take_along_axis(np.partition(arr, k, axis=1), k, axis=1)
```

---

## まとめ

* **`np.partition` は「特定順位だけ確定したい」場合の部分ソート関数**。
* *平均 O(N)* と低オーバーヘッドで、完全ソート不要な統計・上位 k 抽出に最適。
* 位置だけ欲しいなら `np.argpartition` を組み合わせて多配列対応。
* 返ってくる配列は **内部順序がランダム** なので、必要なら別途ソートして整形。
* 降順・複数 k・NaN などの癖を理解して使えば、`np.sort` では届かないスピードを実感できます。
