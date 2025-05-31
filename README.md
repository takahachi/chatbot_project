# Slack Contribution Monitor Bot

松尾・岩澤研究室の大規模受講生コミュニティにおける、Slack上の学習支援・貢献可視化を目的としたBotです。FastAPI・Slack Bolt・ngrokなどを用いて構築されています。

##  解決したい課題

現在、松尾・岩澤研究室ではデータサイエンスやAI関連の講座を多数開講しており、講座によっては11,000人以上の受講生が参加する大規模コミュニティが形成されています。
各講座ごとにSlackが活用されており、学び合いの場として機能していますが、次のような課題が生じています：

- すべてのコメントを目視で確認するのが困難
- ガイドライン違反投稿への対応が遅れるリスク
- コミュニティへの貢献が高い受講生の定量的評価ができていない


##  本Botが実現する機能

### （1）Slack内投稿の自動スコアリング・監視

- ガイドライン違反の可能性がある発言の検出とアラート通知
- 「ありがとう」「すごい」などのポジティブ反応の収集
- 貢献度が高い受講生を自動でスコアリング・ランキング
- 投稿数・リアクション・回答・ポジティブ投稿数などを指標にスコア算出

### （2）技術的要件

- 1日1回程度の更新で十分
- 数百件の投稿にも対応できる処理能力
- LLM/API使用コストの考慮

### （3）使用想定ツール

- Slack API（投稿取得）
- LLM（GPT-4o, Claude, Gemini など）
- テキスト分類器（違反検出・ポジティブ抽出）
- SQLite またはクラウドDB（貢献スコア記録）

### （4）今後の展開

- 「今月の貢献者紹介」機能の開発
- 優秀な受講生を称える表彰機能の実装


##  技術構成

- Python（FastAPI, Slack Bolt）
- ngrok（ローカルサーバの外部公開）
- dotenv（環境変数管理）
- Hugging Face 


##  セットアップ手順

### 1. 仮想環境作成と起動

```bash
cd chatbot_project
python -m venv chatbot
source chatbot/bin/activate
```

2. ライブラリインストール
```
pip install -r requirements.txt
```
.env ファイルの設定
```
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
NGROK_TOKEN=your-ngrok-token
HUGGINGFACE_TOKEN=your-huggingface-token  # 今後使用予定
```


