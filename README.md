# Looking Glass Go: 独自RGBDプレイヤー (DGX SPARK 統合版)

## 1. 概要
NVIDIA DGX SPARK (ARM64) 環境において、Looking Glass Go をスタンドアロンで動作させるための高精度RGBDプレイヤーです。
公式の Bridge / Studio が動作しない環境制約を克服し、デバイス固有の工場出荷時キャリブレーションを自動検出し、高品質な 3D インタリーブ描画を行います。

## 2. キャリブレーション・アーキテクチャ

本システムは、ハードウェア固有の値を最大限に活用しつつ、ユーザーがリアルタイムに微調整できる二層構造を採用しています。

### Calibration 優先順位

1. `--calib-file` で明示指定された個別 calibration JSON
2. Looking Glass Go 本体内の `LKG_calibration/visual.json`
3. リポジトリ内の `lkg_calibration.json` (主に runtimeOverride 保存用)

**補足**: キャリブレーションファイルが一切見つからない場合、Looking Glass Go の標準値（Pitch=49.818等）が適用され、その上で `lkg_calibration.json` のオーバーライドが適用されます。

**自動変換**: 読み込まれた `raw pitch` / `slope` は、光学計算に基づきシェーダー用の `shader pitch` / `tilt` へ自動変換されます。

### B. 実行時オーバーライド (Runtime Overrides)

- **ファイル**: `lkg_calibration.json`
- **共通設定 (`runtimeOverride`)**: `Max Parallax` や `Depthiness` など、コンテンツの見え方に関する全デバイス共通のパラメータ。
- **デバイス固有設定 (`deviceOverride[serial]`)**: `Pitch Offset` / `Tilt Offset` / `Center Offset` など、ハードウェア個体差に起因する補正値。
- **用途**: GUI（コントロールパネル）で操作した値が自動保存され、複数台接続時でも各デバイスに適した補正が自動適用されます。 serial 番号が不明な場合は、ファイル名がキーとして使用されます。

## 3. GUI コントロールパネルの仕様

### Pitch (Shader Space)
- **重要**: GUI 上に表示・操作される `Pitch` は、工場出荷値からの「Shader 用 Pitch」です。
- Raw Pitch (約49.8) ではなく、光学変換後の値（約143.7など）を扱います。
- モアレが発生する場合、この値を 0.01 単位で微調整することで、レンズと描画の位相を完全に一致させることができます。

### Max Parallax (Display Pixel 基準)
- **単位**: Display Pixel (画面上のピクセル数)
- **仕様**: 入力動画の解像度に関わらず、画面上で何ピクセル分の視差（ズレ）を許容するかを定義します。
- **推奨運用範囲**: 
  - `安定推奨`: `2.0` 〜 `3.0` display px (破綻が少なく高品質な表示)
  - `実験範囲`: `3.0` 〜 `5.0` display px (奥行きは増しますが、輪郭のノイズが目立ちやすくなります)
- **注意**: 現在の簡易 DIBR 方式では 3.0 以上を設定すると、前景の裏側の欠落によるテアリングが発生しやすくなります。深度マップの品質に合わせて調整してください。

## 4. 運用・実行方法

### 起動手順 (GUI付き)
```bash
./run_lkg_with_gui.sh [動画ファイル.mp4] --monitor 1
```

### 依存関係
- Python 3.12+ (PySide6, PyOpenGL, glfw, opencv-python)
- FFmpeg (動画デコード用)

## 5. 既知の制限と品質向上のヒント
- **入力解像度**: 現在の推奨設定は `2304x2048` (SBS) です（片側 `1152x2048`）。さらに品質を上げる場合は `2880x2560` を推奨します。
- **圧縮設定**: Depth map の劣化を防ぐため、CRF 10 以下の高品質エンコード、または `yuv444p` 形式での生成を推奨します。
- **動画結合 (`concat_lkg_videos.sh`)**: `display2` ディレクトリ内の入力 mp4 は、すべて Audio Stream を含んでいる必要があります（無音でも可）。Audio がないファイルが混ざると ffmpeg のエラーになります。
