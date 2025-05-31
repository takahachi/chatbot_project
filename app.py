import os
from fastapi import FastAPI, Request, APIRouter
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from dotenv import load_dotenv
from pyngrok import ngrok
import uvicorn

# .envから環境変数を読み込む
load_dotenv()

# --- トークンの取得 ---
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")  # 今後使用予定 

# --- Slackアプリの初期化 ---
slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
app_handler = SlackRequestHandler(slack_app)

# --- Slackイベントの処理 ---
@slack_app.event("app_mention")
def handle_app_mentions(body, say):
    print(body)
    say("What's up?")

# --- FastAPIの設定 ---
router = APIRouter(prefix="/slack")

@router.post("/events")
async def events(request: Request):
    return await app_handler.handle(request)

fastapi_app = FastAPI()
fastapi_app.include_router(router)

# --- ngrokを通して公開 ---
if __name__ == "__main__":
    port = 8000
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)

    try:
        ngrok_tunnel = ngrok.connect(port)
        public_url = ngrok_tunnel.public_url
        print(f"✅ 公開URL: {public_url}")
        print(f"Slack Event Subscription の Request URL: {public_url}/slack/events")

				# --- アプリケーション起動 ---
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

    finally:
        print(" ngrokトンネルを閉じます")
        ngrok.disconnect(public_url)
        ngrok.kill()  # 念のため