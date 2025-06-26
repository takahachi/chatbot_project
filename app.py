import os
import asyncio
import sqlite3
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, BertForSequenceClassification
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slack_bolt import App as BoltApp
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk import WebClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1) 環境変数読み込み
# -----------------------------------------------------------------------------
# ---設定---
# モデルの設定
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student" 
print(f"モデル名: {MODEL_NAME}")

load_dotenv()
SLACK_BOT_TOKEN      = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
NGROK_TOKEN          = os.getenv("NGROK_TOKEN")
REPORT_CHANNEL_ID    = os.getenv("REPORT_CHANNEL_ID")

assert SLACK_BOT_TOKEN, "SLACK_BOT_TOKEN が設定されていません"
assert SLACK_SIGNING_SECRET, "SLACK_SIGNING_SECRET が設定されていません"
assert REPORT_CHANNEL_ID, "REPORT_CHANNEL_ID が設定されていません"

# Slack WebClient (レポート送信用)
web_client = WebClient(token=SLACK_BOT_TOKEN)

# -- モデルロード----
def load_model():
    """
    Hugging Face Transformers の pipeline を使ってモデルをロードする。
    モデル名はグローバル変数 MODEL_NAME から取得。
    """
    global model
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"使用するデバイス: {device}")
        pipe = pipeline(
            "text-classification",
            model=MODEL_NAME,
            device=device,
        )
        print(f"モデル {MODEL_NAME} をロードしました。") 
        model = pipe
        return pipe
    except Exception as e:
        print(f"モデル{MODEL_NAME}のロードに失敗しました: {e}")
        return None

# === ファインチューニング済み禁止判定モデルのロード ===
def load_finetuned_prohibited_model():
    """
    ファインチューニング済みの禁止判定モデルをロードする。
    戻り値: tokenizer, model
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
        model = BertForSequenceClassification.from_pretrained("fine_tuned_model")
        model.eval()
        print("ファインチューニング済みモデルをロードしました。")
        return tokenizer, model
    except Exception as e:
        print(f"ファインチューニングモデルのロードに失敗しました: {e}")
        return None, None

finetuned_tokenizer, finetuned_model = load_finetuned_prohibited_model()
      
def load_model_task():
    """
    非同期でモデルをロードするタスク。
    FastAPI の起動時に呼び出される。
    """
    global model
    model = load_model()
    if model:
        print("モデルロード成功")
    else:
        print("モデルロード失敗")

# -----------------------------------------------------------------------------
# 2) 禁止判定関数
# -----------------------------------------------------------------------------

def is_prohibited(text: str) -> bool:
    """
    ファインチューニング済みの禁止判定モデルを使って、テキストが禁止行為かどうかを判定する。
    戻り値: True なら禁止行為、False なら許可
    """
    global finetuned_tokenizer, finetuned_model
    if finetuned_tokenizer is None or finetuned_model is None:
        print("禁止判定モデルが利用できません")
        return False
    try:
        inputs = finetuned_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = finetuned_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            print(f"禁止判定結果: {probs}")  # デバッグ用に確率を表示
            return probs[0, 1].item() > 0.60
    except Exception as e:
        print(f"禁止判定中にエラー: {e}")
        return False

def llm_based_classify(text: str) -> str:
    global model
    if model is None:
        load_model_task()  # モデルをロード
        if model is None:
            print("モデルロード失敗")
            raise HTTPException(status_code=503, detail="モデルが利用できません。後でもう一度お試しください。")
    try:
        print(f"分類を開始...", text[:50] + "...")  # デバッグ用にテキストの一部を表示
        result = model(text)
        label = result[0]['label']  # モデルの出力からラベルを取得
        score = result[0]['score']  # スコアも取得（必要に応じて）
        print(f"分類結果: {label}, スコア: {score:.4f}")  # デバッグ用にスコアを表示
        print(f"分類が完了しました")

        # 生成されたテキストからポジティブ/その他を判定
        if label == "positive" and score > 0.7:  # スコアが高い場合のみポジティブとする
            return "positive"
        else:
            return "other"
    except Exception as e:
        print(f"分類中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail="分類処理中にエラーが発生しました。後でもう一度お試しください。")

# -----------------------------------------------------------------------------
# 3) SQLite データベース初期化・ヘルパー
# -----------------------------------------------------------------------------
DB_PATH = "user_counts.db"

def get_db_connection():
    """
    SQLite データベースへの接続を返す。
    row_factory を設定して dict 風に取得できるようにする。
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    起動時に呼ぶ。テーブル user_counts を作成する（存在しなければ）。
    user_id と date の複合主キーで、positive を整数型で保持。
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # テーブルを作り直す場合は次の行を有効化
    # cursor.execute("DROP TABLE IF EXISTS user_counts")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_counts (
            user_id   TEXT,
            date      TEXT,
            positive  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, date)
        )
    """)
    conn.commit()
    conn.close()

def upsert_user_count(user_id: str, category: str, date: str = None):
    """
    指定した user_id, date の行がなければ INSERT
    存在すれば、positive カラムに +1 して UPDATE する。
    category が "positive" の場合のみ処理する。
    date: 'YYYY-MM-DD' 形式。省略時は今日の日付。
    """
    if category != "positive":
        return
    if date is None:
        date = datetime.utcnow().date().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_counts(user_id, date, positive)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, date) DO UPDATE SET positive = positive + 1
    """, (user_id, date))
    conn.commit()
    conn.close()

def fetch_all_counts(date: str = None):
    """
    日次レポート用に、指定日または全期間のすべてのユーザー行を dict にして返す。
    返り値の構造: { user_id: { "positive": int }, ... }
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    if date:
        cursor.execute("SELECT user_id, SUM(positive) as positive FROM user_counts WHERE date = ? GROUP BY user_id", (date,))
    else:
        cursor.execute("SELECT user_id, SUM(positive) as positive FROM user_counts GROUP BY user_id")
    rows = cursor.fetchall()
    conn.close()

    result = {}
    for row in rows:
        result[row["user_id"]] = {
            "positive": row["positive"]
        }
    return result

# -----------------------------------------------------------------------------
# 3.5) 月次・日次表彰用関数
# -----------------------------------------------------------------------------
def fetch_top_users_by_month(month_str: str):
    """
    指定した年月 (YYYY-MM) における positive 発言数トップのユーザーを返す。
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, SUM(positive) as total_positive
        FROM user_counts
        WHERE strftime('%Y-%m', date) = ?
        GROUP BY user_id
        ORDER BY total_positive DESC
        LIMIT 1
    """, (month_str,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"user_id": row["user_id"], "total_positive": row["total_positive"]}
    else:
        return None

def fetch_top_user_by_date(date_str: str):
    """
    指定した日付 (YYYY-MM-DD) における positive 発言数トップのユーザーを返す。
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, positive
        FROM user_counts
        WHERE date = ?
        ORDER BY positive DESC
        LIMIT 1
    """, (date_str,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"user_id": row["user_id"], "positive": row["positive"]}
    else:
        return None

# -----------------------------------------------------------------------------
# 4) Slack Bolt アプリを初期化
# -----------------------------------------------------------------------------
bolt_app = BoltApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler  = SlackRequestHandler(bolt_app)

# -----------------------------------------------------------------------------
# 5) メッセージイベントをキャッチして SQLite に upsert する
# -----------------------------------------------------------------------------
@bolt_app.event("message")
def handle_all_messages(event, say, logger):
    print(">>> handle_all_messages が呼ばれました")
    user_id = event.get("user")
    text    = event.get("text", "")

    # Bot 自身のメッセージや user がない場合は無視
    if not user_id or user_id == "USLACKBOT":
        return

    logger.info(f"[message] user={user_id} text={text}")
    print(f">>> logger.info の直後： user={user_id}, text={text}")  

    # 1) 分類
    if is_prohibited(text):
        say(text="⚠️ このメッセージは禁止行為に該当する可能性があります。")
        logger.info(f"[禁止判定] {user_id} の発言が禁止と判定されました。")
        return
    category = llm_based_classify(text)

    if category == "positive":
        today_str = datetime.utcnow().date().isoformat()
        upsert_user_count(user_id, category, today_str)
        current_counts = fetch_all_counts(today_str).get(user_id, {"positive": 0})
        logger.info(f"[集計] {user_id} の {category} を +1 → 現在値 {current_counts}")
    else:
        logger.info(f"[集計] {user_id} の発言は POSITIVE ではないため集計しません。category={category}")

# -----------------------------------------------------------------------------
# 7) APScheduler の設定
# -----------------------------------------------------------------------------
def setup_scheduler():
    scheduler = AsyncIOScheduler()
    # CronTrigger: UTC の 0:00 に実行。
    # 日本時間0:00にしたい場合は UTC15:00（hour=15）とする。
    trigger = CronTrigger(hour=15, minute=0)
    # 表彰メッセージ send_award_report を定期実行
    scheduler.add_job(send_award_report, trigger, id="award_report_job")
    scheduler.start()
    print("[Scheduler] 毎日 0:00 (UTC) に send_award_report を実行するよう設定しました。")

# -----------------------------------------------------------------------------
# 8) FastAPI アプリとルーティング
# -----------------------------------------------------------------------------
app = FastAPI(title="Slackメッセージ集計 Bot (SQLite)", version="1.0")

@app.on_event("startup")
async def startup_event():
    # 1) SQLite テーブルを初期化
    init_db()
    print("[Startup] SQLite テーブル user_counts を初期化（存在しなければ作成）しました。")

    # 2) スケジューラを起動
    setup_scheduler()
    print("[Startup] APScheduler を起動しました。")

    # 3) モデルをロード
    load_model_task()
    print("[Startup] モデルをロードしました。")

@app.post("/slack/events")
async def slack_events(request: Request):
    """
    - Slack の URL 検証 (type=url_verification) に対応
    - それ以外は Bolt に委譲
    """
    body = await request.json()
    if body.get("type") == "url_verification":
        return JSONResponse(content={"challenge": body["challenge"]})
    return await handler.handle(request)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Slackメッセージ集計 Bot (SQLite) が動作中です。"}

# -----------------------------------------------------------------------------
# 8.5) /award エンドポイント: 日次・月次ランキング
# -----------------------------------------------------------------------------
@app.get("/award")
async def award_top_users():
    today = datetime.utcnow().date()
    today_str = today.isoformat()
    month_str = today.strftime('%Y-%m')
    daily_top = fetch_top_user_by_date(today_str)
    monthly_top = fetch_top_users_by_month(month_str)
    return {
        "daily_top": daily_top,
        "monthly_top": monthly_top
    }

# -----------------------------------------------------------------------------
# 9) メインブロック: uvicorn + ngrok（任意）起動
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = 8000
    # ngrok トークンがある場合はトンネルを作成
    if NGROK_TOKEN:
        from pyngrok import ngrok as _ngrok
        _ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = _ngrok.connect(port)
        print(f"✅ ngrok で公開中: {tunnel.public_url}")
        print(f"  → Slack Event Subscription Request URL: {tunnel.public_url}/slack/events")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")