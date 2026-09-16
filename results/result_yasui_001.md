# 学習結果2026/09/16　保井

## 1. 使用したデータセット

- WIDER FACE データセット

## 2. 学習条件

### モデル

- Faster R-CNN 系顔検出モデル

### Optimizer

- AdamW

### Learning Rate

- 1e-4


### Epoch

- 1

### Batch Size

- 1

### 画像サイズ

- 長辺320pxへリサイズ
- 縦横比維持

### OOM対策

- bbox数上限: 50個/画像

### データ拡張

実装済み

- なし

追加検討

- Brightness変換
- Contrast変換


## 3. 評価条件

学習結果未出力。

そのため正式な評価条件は不明。



## 4. Easy AP

不明



## 5. Medium AP

不明



## 6. Hard AP

不明



## 7. 学習済みモデルの保存先

/kaggle/working/DLC-FaceDetection-PRE/face_detector_fasterrcnn.pth


## 8. モデルのハッシュまたは識別情報

### モデルファイル名

face_detector_fasterrcnn.pth

### ハッシュ値

未取得のため不明


## 9. 代表的なグラフ

現状なし


## 10. 分かったこと

- GPU メモリを大量消費する
- T4 GPUでも OOM（Out Of Memory）が発生する場合がある
- 画像リサイズと bbox 制限が必要だった
- 学習済みモデルの生成と保存には成功した
- 推論画像では耳や顔の一部、丸い影を顔として誤検出するケースが確認された


## 11. 失敗したこと

### P100 GPU

PyTorch 2.10 + CUDA 12.8 環境で互換性エラーが発生した。

### T4 GPU

学習途中で CUDA Out Of Memory が発生した。

### 学習設定

- Epoch=1 のため十分な学習ができていない可能性が高い
- 学習結果の出力ができていない
- データ拡張が未実装


## 12. 次に試したいこと

### データ拡張

- Horizontal Flip

### 学習条件改善

- Epoch: 1 → 10
- Brightness、Contrastのデータ拡張

### 評価

- Easy AP算出
- Medium AP算出
- Hard AP算出
- mAP算出

### 出力改善

- 学習結果の保存
