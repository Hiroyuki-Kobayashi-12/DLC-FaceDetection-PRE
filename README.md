# DLC26 Face Detection

DLC26前半に実施する、WIDER FACEを用いた顔検出ワークショップのリポジトリです。

## 1. 目的

本ワークショップでは、共通の顔検出環境を土台として、参加者がそれぞれ自由にモデルや学習方法を試します。

モデルの性能向上だけでなく、以下を経験することも目的とします。

- GitHubを使用したコード管理
- Branchによる作業環境の分離
- Commitによる実験履歴の記録
- Kaggleを使用したモデル学習
- ローカル環境を使用した推論と評価
- Pull Requestを使用した成果共有
- 他の参加者とのディスカッション
- 複数の成果を一つにまとめる経験

完成した結果だけでなく、途中経過、困り事、予想と異なる結果、失敗した実験も共有します。

参加者同士が気軽に質問や相談を行い、それぞれの試行錯誤をチーム全体の知見へ変えていくことを大切にします。

---

## 2. 使用環境

- コード管理：GitHub
- コード編集・Git操作：VS Code
- モデル学習：Kaggle
- 推論・評価：ローカル環境
- データセット：WIDER FACE
- 学習データ：WIDER FACE Train
- 評価データ：WIDER FACE Validation
- 評価指標：Easy AP、Medium AP、Hard AP

学習はKaggleで実施します。

Kaggleで出力した学習済みモデルをローカル環境へ取得し、WIDER FACE Validationに対する推論と評価を行います。

---

## 3. ディレクトリ構造

```text
DLC-2026-FaceDetection/
├── data/
├── kaggle_line/
├── results/
├── tools/
├── .gitignore
└── README.md
```

### `data/`

WIDER FACEに関するデータ、アノテーション、評価に必要なファイルを配置します。

主に以下を想定しています。

- WIDER FACE Train
- WIDER FACE Validation
- 学習用アノテーション
- Validation用の正解情報
- Easy、Medium、Hardの評価情報
- WIDER FACEの評価スクリプト

データセットの画像など、容量が大きいファイルはGitHubでは管理しません。

### `kaggle_line/`

Kaggle Notebookで使用する学習コードを配置します。

Kaggle Notebookの各セルに対応する形で、Pythonファイルを分割します。

例：

```text
kaggle_line/
├── cell_01_config.py
├── cell_02_model_create.py
├── cell_03_loss_create.py
├── cell_04_optimizer_create.py
├── cell_05_scheduler_create.py
├── cell_06_transforms_create.py
└── cell_07_train.py
```

参加者は、自分のBranch上で`kaggle_line/`を自由に変更できます。

変更対象として、以下を想定しています。

- Config
- Model
- Backbone
- Loss
- Optimizer
- Scheduler
- Transform
- Data Augmentation
- Hyperparameter
- 学習処理

### `results/`

Kaggleで学習したモデル、ローカル評価の結果、可視化などを配置します。

例：

```text
results/
├── model/
│   ├── best_model.pth
│   └── model_info.json
│
├── evaluation/
│   ├── evaluation.md
│   ├── metrics.json
│   └── metrics.csv
│
└── visualization/
    ├── training_loss.png
    ├── learning_rate.png
    ├── easy_pr_curve.png
    ├── medium_pr_curve.png
    ├── hard_pr_curve.png
    └── prediction_samples/
```

学習済みモデルは容量が大きくなる可能性があります。

モデルをGitHubで共有する方法については、ファイルサイズを確認したうえで、Git LFSなどの利用を検討します。

### `tools/`

ローカルで使用する推論、評価、可視化などの処理を配置します。

例：

```text
tools/
├── predict.py
├── evaluate.py
└── visualize.py
```

`tools/`から`data/`内のWIDER FACE評価スクリプトや正解情報を使用し、以下を実施する想定です。

- 学習済みモデルの読み込み
- WIDER FACE Validationへの推論
- 予測結果の出力
- Easy APの計算
- Medium APの計算
- Hard APの計算
- Precision-Recall Curveの作成
- 推論結果の可視化

---

## 4. GitHub Branchの考え方

### `main`

`main`は、全参加者が実験を開始するための共通環境です。

```text
main
├── 共通の初期学習環境
├── 共通の初期評価環境
├── WIDER FACE評価基盤
└── 利用方法
```

`main`では、原則として直接作業しません。

共通環境を変更する場合は、Branch上で変更し、Pull Requestを通して反映します。

### 参加者用Branch

参加者ごとにBranchを作成します。

Branch名は、以下の形式を基本とします。

```text
member/<GitHubユーザー名または参加者名>
```

例：

```text
member/kobayashi
member/sato
member/suzuki
```

参加者用Branchを作成する理由は、新しい参加者が増えた場合でも、それぞれが独立した実験環境を持てるようにするためです。

各参加者は、自分のBranch内で以下を自由に変更できます。

- `kaggle_line/`
- `tools/`
- `results/`
- READMEや実験メモ
- その他、自分の実験に必要なファイル

GitHubのリポジトリ内に参加者別フォルダは作成しません。

誰の環境であるかは、Branch名で識別します。

---

## 5. 初めて参加する場合

### 5.1 リポジトリをCloneする

GitHubのリポジトリ画面からURLを取得し、VS CodeへCloneします。

```powershell
git clone https://github.com/Hiroyuki-Kobayashi-12/DLC-2026-FaceDetection.git
```

Cloneしたフォルダへ移動します。

```powershell
cd DLC-2026-FaceDetection
```

### 5.2 `main`を最新化する

```powershell
git switch main
git pull origin main
```

### 5.3 自分のBranchを作成する

例として、小林が作成する場合は以下です。

```powershell
git switch -c member/kobayashi
```

作成したBranchをGitHubへ登録します。

```powershell
git push -u origin member/kobayashi
```

### 5.4 現在のBranchを確認する

```powershell
git branch --show-current
```

以下のように、自分のBranch名が表示されることを確認します。

```text
member/kobayashi
```

作業を開始する前に、必ず現在のBranchを確認してください。

---

## 6. 実験の進め方

### Step 1：自分のBranchへ移動する

```powershell
git switch member/kobayashi
```

### Step 2：GitHub上の最新状態を取得する

```powershell
git pull origin member/kobayashi
```

### Step 3：`kaggle_line/`を変更する

実験内容に合わせて、Kaggle用コードを変更します。

例：

```text
kaggle_line/
├── cell_01_config.py
├── cell_02_model_create.py
├── cell_03_loss_create.py
├── cell_04_optimizer_create.py
├── cell_05_scheduler_create.py
├── cell_06_transforms_create.py
└── cell_07_train.py
```

変更例：

- Model構造を変更する
- Backboneを変更する
- Lossを変更する
- Optimizerを変更する
- Schedulerを変更する
- Data Augmentationを変更する
- Image Sizeを変更する
- Batch Sizeを変更する
- Learning Rateを変更する
- Epoch数を変更する

### Step 4：変更内容を確認する

```powershell
git status
```

差分を確認します。

```powershell
git diff
```

### Step 5：変更をCommitする

変更したファイルをステージします。

```powershell
git add kaggle_line
```

Commitします。

```powershell
git commit -m "experiment: モデル構成を変更"
```

変更内容ごとにCommitを分けることを推奨します。

例：

```text
experiment: 入力画像サイズを変更
experiment: Focal Lossを追加
experiment: AdamWの設定を変更
experiment: Cosine Schedulerを追加
experiment: Data Augmentationを変更
fix: Kaggle学習時のエラーを修正
```

### Step 6：GitHubへPushする

```powershell
git push
```

これにより、GitHub上の自分のBranchへ変更が反映されます。

---

## 7. Kaggleでの学習

自分のBranchにある`kaggle_line/`の内容を、Kaggle Notebookの対応するセルへ反映します。

例：

```text
cell_01_config.py
→ Kaggle NotebookのConfigセル

cell_02_model_create.py
→ Model作成セル

cell_03_loss_create.py
→ Loss作成セル

cell_04_optimizer_create.py
→ Optimizer作成セル

cell_05_scheduler_create.py
→ Scheduler作成セル

cell_06_transforms_create.py
→ Transform作成セル

cell_07_train.py
→ 学習実行セル
```

Kaggle上だけでコードを変更した場合は、同じ変更を自分のBranchにも反映してCommitしてください。

Kaggle上だけに存在する変更を残さないようにします。

学習後は、以下を取得します。

- 学習済みモデル
- 学習Loss
- Learning Rateの履歴
- Best Epoch
- 学習時間
- その他の学習結果

---

## 8. ローカルでの推論・評価

Kaggleで作成した学習済みモデルを、ローカル環境へ手動で取得します。

取得したモデルを`results/`へ配置します。

例：

```text
results/
└── model/
    └── best_model.pth
```

`tools/`内の処理を使用し、WIDER FACE Validationに対して推論と評価を行います。

評価指標は以下です。

- Easy AP
- Medium AP
- Hard AP

評価後は、以下のようなファイルを`results/`へ保存します。

```text
results/
├── model/
│   ├── best_model.pth
│   └── model_info.json
│
├── evaluation/
│   ├── evaluation.md
│   ├── metrics.json
│   └── metrics.csv
│
└── visualization/
    ├── training_loss.png
    ├── learning_rate.png
    ├── easy_pr_curve.png
    ├── medium_pr_curve.png
    ├── hard_pr_curve.png
    └── prediction_samples/
```

---

## 9. Experiment IDの考え方

本プロジェクトでは、実験ごとにフォルダを複製することを必須としません。

各参加者のBranchにある環境全体を、1つの実験環境として扱います。

```text
member/kobayashi
├── data/
├── kaggle_line/
├── results/
├── tools/
├── .gitignore
└── README.md
```

Model、Loss、Scheduler、学習済みモデル、評価結果、可視化を含む環境全体を、Experimentとして記録します。

Experiment IDの例：

```text
Experiment_001
Experiment_002
Experiment_003
```

実験の途中経過はCommitで記録し、実験が完了した時点はGit Tagで記録します。

---

## 10. ExperimentをCommitで記録する

実験中は、変更内容ごとにCommitします。

例：

```powershell
git add kaggle_line
git commit -m "experiment: モデル構造を変更"
```

```powershell
git add kaggle_line
git commit -m "experiment: Focal Lossを追加"
```

```powershell
git add kaggle_line
git commit -m "experiment: Schedulerを変更"
```

評価結果を保存した後、環境全体をCommitします。

```powershell
git add .
git commit -m "experiment: Experiment_001を完了"
git push
```

モデルファイルがGit管理対象外になっている場合は、モデル以外の結果だけがCommitされます。

学習済みモデルの共有方法は、ファイルサイズを確認して別途決定します。

---

## 11. ExperimentをTagで記録する

実験完了時点のCommitにTagを付けます。

Tag名には、参加者名とExperiment IDを含めます。

例：

```text
kobayashi-experiment-001
kobayashi-experiment-002
sato-experiment-001
suzuki-experiment-001
```

小林の1回目の実験にTagを付ける場合は以下です。

```powershell
git tag -a "kobayashi-experiment-001" -m "Experiment_001"
```

TagをGitHubへPushします。

```powershell
git push origin "kobayashi-experiment-001"
```

これにより、Tagが付いた時点の環境全体が`Experiment_001`として記録されます。

---

## 12. Experiment間の差分を確認する

実験間の変更内容は、Gitで比較できます。

例：

```powershell
git diff kobayashi-experiment-001 kobayashi-experiment-002
```

`kaggle_line/`だけを比較する場合は以下です。

```powershell
git diff `
  kobayashi-experiment-001 `
  kobayashi-experiment-002 `
  -- kaggle_line
```

特定のExperiment時点の内容を確認する場合は以下です。

```powershell
git show kobayashi-experiment-001
```

特定のExperiment時点へ一時的に移動する場合は以下です。

```powershell
git switch --detach kobayashi-experiment-001
```

確認後は、自分のBranchへ戻ります。

```powershell
git switch member/kobayashi
```

---

## 13. 次のExperimentへ進む

`Experiment_001`へTagを付けた後も、同じ参加者Branchで次の変更を行います。

```text
member/kobayashi

kobayashi-experiment-001
└── Experiment_001完了時点

現在のBranch先端
└── Experiment_002に向けた作業
```

`Experiment_002`が完了したら、同様にCommitとTagを作成します。

```powershell
git add .
git commit -m "experiment: Experiment_002を完了"
git push
```

```powershell
git tag -a "kobayashi-experiment-002" -m "Experiment_002"
git push origin "kobayashi-experiment-002"
```

この流れを繰り返します。

---

## 14. 他の参加者の環境を見る

GitHub上でBranchを切り替えることで、他の参加者の環境を確認できます。

例：

```text
main
member/kobayashi
member/sato
member/suzuki
```

ローカル環境で他のBranchを確認する場合は、最初に最新情報を取得します。

```powershell
git fetch --all --tags
```

他の参加者のBranchへ移動します。

```powershell
git switch member/sato
```

確認後は自分のBranchへ戻ります。

```powershell
git switch member/kobayashi
```

他の参加者のBranchを変更する場合は、事前に本人と相談してください。

---

## 15. `main`へ共有するもの

参加者Branchのすべてを、毎回`main`へ統合する必要はありません。

共有したい内容だけをPull Requestで提案します。

共有対象の例：

- 他の参加者にも有効なモデル構造
- 共通化できるLoss
- 共通化できるOptimizer
- 共通化できるScheduler
- 有効なData Augmentation
- Kaggle学習環境の改善
- 評価処理の改善
- バグ修正
- READMEや手順書の改善

実験環境全体を共有したい場合は、Branch名とTag名を共有します。

```text
Branch：member/kobayashi
Tag：kobayashi-experiment-001
```

他の参加者は、そのTagを確認することで、実験時点の環境全体を確認できます。

---

## 16. Pull Requestの使い方

共通環境へ反映したい変更がある場合は、Pull Requestを作成します。

```text
member/kobayashi
        ↓
Pull Request
        ↓
main
```

Pull Requestには、以下を記載します。

```markdown
## 変更内容

今回変更した内容を記載します。

## Experiment

- Branch：
- Tag：
- Experiment ID：

## 評価結果

- Easy AP：
- Medium AP：
- Hard AP：

## mainへ反映したい内容

共通環境へ反映したいファイルや処理を記載します。

## 確認してほしいこと

レビューやディスカッションで確認してほしいことを記載します。
```

`main`へ反映するかどうかは、Pull Request上の確認とディスカッションを通じて決定します。

---

## 17. GitHub運用ルール

### ルール1

`main`へ直接Pushしません。

### ルール2

参加者ごとに自分のBranchを作成します。

```text
member/<参加者名>
```

### ルール3

作業開始前に現在のBranchを確認します。

```powershell
git branch --show-current
```

### ルール4

変更内容ごとにCommitを作成します。

### ルール5

Kaggle上だけに変更を残しません。

### ルール6

Experiment完了時に、結果をCommitしてTagを付けます。

### ルール7

Tag名には、参加者名とExperiment IDを含めます。

### ルール8

共通環境へ変更を反映する場合は、Pull Requestを作成します。

### ルール9

他の参加者のBranchを勝手に変更しません。

### ルール10

秘密情報や個人情報をCommitしません。

---

## 18. 最初に試す手順

最初は、次の流れを一度試します。

```text
1. リポジトリをCloneする
2. 自分のmember Branchを作る
3. kaggle_lineのファイルを1つ変更する
4. 変更内容を確認する
5. Commitする
6. GitHubへPushする
7. Kaggleで学習する
8. 学習済みモデルをローカルへ取得する
9. WIDER FACE Validationで評価する
10. resultsへ評価結果を保存する
11. Experiment完了Commitを作る
12. Experiment Tagを作る
13. Experiment間の差分を確認する
14. 必要に応じてPull Requestを作る
```

最初からすべてを完成させる必要はありません。

Branch、Commit、Push、Tagの流れを確認しながら、少しずつ環境を整えていきます。

---

## 19. 大切にしたいこと

本ワークショップでは、最も高いAPを出すことだけを目的にしません。

以下の共有を歓迎します。

- 未完成のアイデア
- 初歩的な質問
- 実装途中の相談
- 学習が動かなかった原因
- Checkpointを読み込めなかった問題
- Easyだけ改善した結果
- Hardだけ低下した結果
- 仮説と異なった結果
- 精度が下がった実験
- 次に何を試すべきか分からない状況
- 他の参加者に一緒に確認してほしい結果

参加者Branchは、それぞれが自由に試行錯誤するための実験環境です。

Commitには変更の意図を残し、TagにはExperimentの完了時点を残します。

結果だけでなく、どのように考え、どのように環境を変えたかをGitHub上へ残すことで、個人の経験をチーム全体の知見へ変えていきます。