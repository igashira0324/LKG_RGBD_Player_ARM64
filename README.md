# Looking Glass Go: 独自RGBDプレイヤー (DGX SPARK 統合版)

## 1. 概要
NVIDIA DGX SPARK (ARM64) 環境において、Looking Glass Go をスタンドアロンで動作させるための高精度RGBDプレイヤーです。
公式の Bridge / Studio が動作しない環境制約を克服し、デバイス固有の工場出荷時キャリブレーションを自動検出し、高品質な 3D インタリーブ描画を行います。

## 2. キャリブレーション・アーキテクチャ

本システムは、ハードウェア固有の値を最大限に活用しつつ、ユーザーがリアルタイムに微調整できる二層構造を採用しています。

### A. ベース・キャリブレーション (Factory Calibration)
- **ソース**: Looking Glass Go 本体内（USBドライブとして認識される `/media/` 等のパス）の `LKG_calibration/visual.json` を自動探索します。
- **優先順位**: 実機のファイルが最優先されます。見つからない場合は `lkg_calibration.json` のデフォルト値を使用します。
- **自動変換**: 読み込まれた `raw pitch` / `slope` は、光学計算に基づきシェーダー用の `shader pitch` / `tilt` へ自動変換されます。

### B. 実行時オーバーライド (Runtime Overrides)
- **ファイル**: `lkg_calibration.json` の `runtimeOverride` セクション。
- **用途**: GUI（コントロールパネル）で操作した `Pitch Offset` や `Max Parallax` などの微調整値がここに保存され、次回起動時に自動適用されます。

## 3. GUI コントロールパネルの仕様

### Pitch (Shader Space)
- **重要**: GUI 上に表示・操作される `Pitch` は、工場出荷値からの「Shader 用 Pitch」です。
- Raw Pitch (約49.8) ではなく、光学変換後の値（約143.7など）を扱います。
- モアレが発生する場合、この値を 0.01 単位で微調整することで、レンズと描画の位相を完全に一致させることができます。

### Max Parallax (Display Pixel 基準)
- **単位**: Display Pixel (画面上のピクセル数)
- **仕様**: 入力動画の解像度に関わらず、画面上で何ピクセル分の視差（ズレ）を許容するかを定義します。
- **推奨運用範囲**: 
  - `Max Parallax`: `2.0` 〜 `5.0` (視覚的に自然で破綻の少ない立体感)
  - `Depthiness`: `0.5` 〜 `1.5`
- **仕様**: スライダー (0.0 〜 5.0) で、Max Parallax の範囲内で立体感の強弱を調整します。現在の簡易 DIBR 方式では 2.0 以上は破綻が目立ちやすいため注意してください。

## 4. 運用・実行方法

### 起動手順 (GUI付き)
```bash
./run_lkg_with_gui.sh [動画ファイル.mp4] --monitor 1
```

### 依存関係
- Python 3.12+ (PySide6, PyOpenGL, glfw, opencv-python)
- FFmpeg (動画デコード用)

## 5. 既知の制限と品質向上のヒント
- **入力解像度**: 現在の推奨は 1152x1024 (SBS) ですが、Looking Glass Go のネイティブ解像度 (1440x2560) に合わせた高解像度 RGBD を使用することで、境界線の破綻が大幅に軽減されます。
- **圧縮設定**: Depth map の劣化を防ぐため、CRF 10 以下の高品質エンコード、または `yuv444p` 形式での生成を推奨します。
