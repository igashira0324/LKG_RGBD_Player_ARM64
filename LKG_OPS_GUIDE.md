# Looking Glass Go: RGBD 運用手順書

本ドキュメントは、Looking Glass Go を用いた同期再生システムおよび ComfyUI による RGBD 動画生成の運用・構築に関するガイドラインです。

## 1. ハードウェア構成 (DGX SPARK)
- **計算機**: NVIDIA Grace Blackwell (GB10)
- **VRAM**: 124GB (HBM3e)
- **OS**: Ubuntu 22.04 LTS (aarch64)
- **ディスプレイ**: 2x Looking Glass Go (USB-C Power + HDMI)

## 2. RGBD 動画生成の制約と推奨事項
過去の運用データに基づき、以下の制限事項を遵守してください。

### 動画秒数の許容範囲
- **推奨**: 15秒 ～ 30秒 (生成・結合ともに極めて安定)
- **許容**: 60秒 (生成時間は長くなるが、成功率は高い)
- **警告**: 90秒以上 (ffmpeg による最終結合時にシステムメモリを大量に消費し、フリーズするリスクがあります)

### メモリ使用量に関する注意
- ComfyUI での動画生成中、特に `Video Combined` ノードや `FFmpeg` によるエンコード時に VRAM だけでなくシステム RAM も急激に消費されます。
- 動作が不安定な場合は、他の重いプロセス（LLM 推論、音楽生成など）を停止してください。

### 推奨フォーマット
- **解像度**: 1152x1024 (Unified RGBD)
- **レイアウト**: Side-by-Side (左: Depth, 右: RGB) または Top-Bottom

## 3. 展示運用 (ハーネスの使用方法)
`/home/nttdmse/aipf/ComfyUI/output/RGBD/` 配下の `Makefile` を使用して操作します。

### 同期再生の開始
```bash
make sync
```
- `display1_combined.mp4` と `display2_combined.mp4` を2枚の Looking Glass に同期して再生します。
- どちらかの動画が終了すると、自動的に再同期(Re-sync)してループします。

### 動画の結合
個別の動画セグメントを結合してループ用の動画を作成します。
```bash
make concat
```
- `display1/` および `display2/` ディレクトリ内の全 MP4 ファイルをソートして結合します。

### 環境のクリーンアップ
```bash
make clean
```
- 一時的な同期信号ファイルを削除します。

## 4. トラブルシューティング
- **プレイヤーが起動しない**: `DISPLAY=:1` が正しく設定されているか確認してください。
- **同期がズレる**: `/tmp/lkg_sync_go` が残っている場合は `make clean` を実行してください。
- **PCがフリーズする**: 動画の秒数を短くするか、`da3_base` モデル（軽量版）の使用を検討してください。
