import os
from datetime import datetime, timedelta
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from passlib.context import CryptContext

from database import db, create_document, get_documents
from schemas import User as UserSchema, Trade as TradeSchema, Feedback as FeedbackSchema

# Optional integrations
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
STRIPE_PRICE_MONTHLY = os.getenv("STRIPE_PRICE_MONTHLY")
STRIPE_PRICE_ANNUAL = os.getenv("STRIPE_PRICE_ANNUAL")

# Auth settings
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Utilities
from bson import ObjectId

def oid_str(value: Any) -> str:
    if isinstance(value, ObjectId):
        return str(value)
    return str(value)

# FastAPI app
app = FastAPI(title="Trade Journal SaaS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TradeCreateRequest(BaseModel):
    ticker: str
    entry_price: float
    exit_price: Optional[float] = None
    position_size: float
    direction: Literal['long','short'] = 'long'
    date: Optional[datetime] = None
    setup: Optional[str] = None
    reasoning: Optional[str] = None
    emotions: Optional[str] = None
    screenshots: Optional[List[str]] = None
    outcome: Optional[Literal['win','loss','breakeven']] = None

class TradeUpdateRequest(TradeCreateRequest):
    pass

class CreateCheckoutRequest(BaseModel):
    plan: Literal['monthly','annual'] = 'monthly'

# Auth helpers

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db["user"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Routes
@app.get("/")
def read_root():
    return {"message": "Trade Journal Backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
            response["database"] = "✅ Connected & Working"
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:80]}"
    return response


# Auth endpoints
@app.post("/auth/signup", response_model=TokenResponse)
def signup(body: SignupRequest):
    existing = db["user"].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = get_password_hash(body.password)
    user_doc = {
        "email": body.email,
        "password_hash": hashed,
        "name": body.name,
        "subscription_status": "none",
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    res = db["user"].insert_one(user_doc)
    token = create_access_token({"sub": str(res.inserted_id)})
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = db["user"].find_one({"email": body.email})
    if not user or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user["_id"])})
    return TokenResponse(access_token=token)


@app.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {
        "id": oid_str(user["_id"]),
        "email": user.get("email"),
        "name": user.get("name"),
        "subscription_status": user.get("subscription_status", "none"),
    }


# Trades CRUD
@app.post("/trades")
def create_trade(body: TradeCreateRequest, user: dict = Depends(get_current_user)):
    trade = TradeSchema(
        user_id=str(user["_id"]),
        ticker=body.ticker.upper(),
        entry_price=body.entry_price,
        exit_price=body.exit_price,
        position_size=body.position_size,
        direction=body.direction,
        date=body.date or datetime.utcnow(),
        setup=body.setup,
        reasoning=body.reasoning,
        emotions=body.emotions,
        screenshots=body.screenshots,
        outcome=body.outcome,
    )
    tid = create_document("trade", trade)
    # Trigger AI feedback (sync for simplicity)
    feedback = generate_feedback_for_trade(user_id=str(user["_id"]), trade_id=tid)
    return {"trade_id": tid, "feedback": feedback}


@app.get("/trades")
def list_trades(
    ticker: Optional[str] = None,
    outcome: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    q: Dict[str, Any] = {"user_id": str(user["_id"])}
    if ticker:
        q["ticker"] = ticker.upper()
    if outcome:
        q["outcome"] = outcome
    if start or end:
        q["date"] = {}
        if start:
            q["date"]["$gte"] = datetime.fromisoformat(start)
        if end:
            q["date"]["$lte"] = datetime.fromisoformat(end)
    trades = list(db["trade"].find(q).sort("date", -1))
    for t in trades:
        t["id"] = oid_str(t.pop("_id"))
    return trades


@app.get("/trades/{trade_id}")
def get_trade(trade_id: str, user: dict = Depends(get_current_user)):
    t = db["trade"].find_one({"_id": ObjectId(trade_id), "user_id": str(user["_id"])})
    if not t:
        raise HTTPException(status_code=404, detail="Trade not found")
    t["id"] = oid_str(t.pop("_id"))
    return t


@app.put("/trades/{trade_id}")
def update_trade(trade_id: str, body: TradeUpdateRequest, user: dict = Depends(get_current_user)):
    update = {k: v for k, v in body.model_dump().items() if v is not None}
    update["updated_at"] = datetime.utcnow()
    res = db["trade"].update_one({"_id": ObjectId(trade_id), "user_id": str(user["_id"])}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {"ok": True}


@app.delete("/trades/{trade_id}")
def delete_trade(trade_id: str, user: dict = Depends(get_current_user)):
    res = db["trade"].delete_one({"_id": ObjectId(trade_id), "user_id": str(user["_id"])})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Trade not found")
    db["feedback"].delete_many({"trade_id": trade_id, "user_id": str(user["_id"])})
    return {"ok": True}


# Feedback endpoints
@app.get("/trades/{trade_id}/feedback")
def trade_feedback(trade_id: str, user: dict = Depends(get_current_user)):
    items = list(db["feedback"].find({"trade_id": trade_id, "user_id": str(user["_id"])}).sort("created_at", -1))
    for f in items:
        f["id"] = oid_str(f.pop("_id"))
    return items


@app.get("/feedback")
def all_feedback(user: dict = Depends(get_current_user)):
    items = list(db["feedback"].find({"user_id": str(user["_id"])}).sort("created_at", -1))
    for f in items:
        f["id"] = oid_str(f.pop("_id"))
    return items


# Stats endpoint
@app.get("/stats")
def stats(user: dict = Depends(get_current_user)):
    uid = str(user["_id"]) 
    # Performance over time (monthly)
    pipeline = [
        {"$match": {"user_id": uid, "outcome": {"$in": ["win", "loss", "breakeven"]}}},
        {"$group": {
            "_id": {"y": {"$year": "$date"}, "m": {"$month": "$date"}},
            "wins": {"$sum": {"$cond": [{"$eq": ["$outcome", "win"]}, 1, 0]}},
            "losses": {"$sum": {"$cond": [{"$eq": ["$outcome", "loss"]}, 1, 0]}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.y": 1, "_id.m": 1}}
    ]
    perf = list(db["trade"].aggregate(pipeline))

    # Win rate by strategy/setup
    pipeline2 = [
        {"$match": {"user_id": uid, "setup": {"$ne": None}, "outcome": {"$in": ["win", "loss", "breakeven"]}}},
        {"$group": {
            "_id": "$setup",
            "wins": {"$sum": {"$cond": [{"$eq": ["$outcome", "win"]}, 1, 0]}},
            "losses": {"$sum": {"$cond": [{"$eq": ["$outcome", "loss"]}, 1, 0]}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    by_setup = list(db["trade"].aggregate(pipeline2))

    # Common positives/negatives from feedback
    def aggregate_feedback(field: str):
        pipeline = [
            {"$match": {"user_id": uid}},
            {"$unwind": f"${field}"},
            {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        return list(db["feedback"].aggregate(pipeline))

    positives = aggregate_feedback("positives")
    negatives = aggregate_feedback("negatives")

    return {
        "performance_over_time": perf,
        "winrate_by_setup": by_setup,
        "top_positives": positives,
        "top_negatives": negatives,
    }


# Billing / Stripe
@app.post("/billing/create-checkout-session")
def create_checkout_session(body: CreateCheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_API_KEY:
        # In demo, return mock URL
        return {"url": f"{FRONTEND_URL}/billing?mock=1"}
    import stripe  # type: ignore
    stripe.api_key = STRIPE_API_KEY

    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        customer = stripe.Customer.create(email=user.get("email"))
        customer_id = customer.id
        db["user"].update_one({"_id": user["_id"]}, {"$set": {"stripe_customer_id": customer_id}})

    price_id = STRIPE_PRICE_MONTHLY if body.plan == 'monthly' else STRIPE_PRICE_ANNUAL
    if not price_id:
        raise HTTPException(status_code=400, detail="Stripe price not configured")

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{FRONTEND_URL}/billing?success=1",
        cancel_url=f"{FRONTEND_URL}/billing?canceled=1",
    )
    return {"url": session.url}


@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_API_KEY:
        return {"received": True}
    import stripe  # type: ignore
    stripe.api_key = STRIPE_API_KEY

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        if endpoint_secret:
            event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
        else:
            event = stripe.Event.construct_from(request.json(), stripe.api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] in ("checkout.session.completed", "customer.subscription.updated", "customer.subscription.created"):
        data = event["data"]["object"]
        # Resolve customer -> user
        customer_id = data.get("customer")
        sub_id = data.get("subscription") or data.get("id")
        status = data.get("status") or data.get("subscription_details", {}).get("status")
        if customer_id:
            db["user"].update_one(
                {"stripe_customer_id": customer_id},
                {"$set": {"subscription_status": status or "active", "stripe_subscription_id": sub_id}}
            )
    return {"received": True}


# AI Feedback Engine

def generate_feedback_for_trade(user_id: str, trade_id: str) -> dict:
    trade = db["trade"].find_one({"_id": ObjectId(trade_id)})
    if not trade:
        return {}
    # Fetch some history context
    last_feedback = list(db["feedback"].find({"user_id": user_id}).sort("created_at", -1).limit(20))
    history_summary = {
        "top_positives": {},
        "top_negatives": {},
    }
    for fb in last_feedback:
        for p in fb.get("positives", []) or []:
            history_summary["top_positives"][p] = history_summary["top_positives"].get(p, 0) + 1
        for n in fb.get("negatives", []) or []:
            history_summary["top_negatives"][n] = history_summary["top_negatives"].get(n, 0) + 1

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=OPENAI_API_KEY)
            sys_prompt = (
                "You are a disciplined trading coach. Analyze the single trade provided. "
                "Highlight specific positives and negatives. Then give concise, actionable suggestions. "
                "Consider the user's recurring patterns if provided. Reply as strict JSON with keys: "
                "positives, negatives, suggestions, summary."
            )
            user_content = {
                "trade": {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in trade.items() if k not in ["_id"]
                },
                "history": history_summary,
            }
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": str(user_content)},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = completion.choices[0].message.content
        except Exception as e:
            content = None
    else:
        content = None

    if content:
        import json
        try:
            data = json.loads(content)
            positives = data.get("positives", [])
            negatives = data.get("negatives", [])
            suggestions = data.get("suggestions", [])
            summary = data.get("summary")
        except Exception:
            positives, negatives, suggestions, summary = heuristic_feedback(trade, history_summary)
    else:
        positives, negatives, suggestions, summary = heuristic_feedback(trade, history_summary)

    fb = FeedbackSchema(
        user_id=user_id,
        trade_id=trade_id,
        positives=positives,
        negatives=negatives,
        suggestions=suggestions,
        summary=summary,
    )
    fid = create_document("feedback", fb)
    return {"id": fid, "positives": positives, "negatives": negatives, "suggestions": suggestions, "summary": summary}


def heuristic_feedback(trade: dict, history: dict):
    positives = []
    negatives = []
    suggestions = []
    # Simple heuristics
    if trade.get("outcome") == "win":
        positives.append("Closed the trade profitably")
    if trade.get("outcome") == "loss":
        negatives.append("Loss recorded — review risk management and stop placement")
    if trade.get("entry_price") and trade.get("exit_price"):
        rr = None
        try:
            rr = round((trade["exit_price"] - trade["entry_price"]) / (trade["entry_price"] or 1), 3)
        except Exception:
            pass
        if rr is not None:
            suggestions.append(f"Review R multiple: raw return {rr}")
    if trade.get("emotions"):
        suggestions.append("Note emotional state in journal and compare with outcomes")
    if trade.get("setup"):
        suggestions.append(f"Tag more trades with setup '{trade['setup']}' for better stats")

    # Patterns
    if history.get("top_negatives"):
        worst = sorted(history["top_negatives"].items(), key=lambda x: -x[1])[0][0]
        suggestions.append(f"Recurring issue detected: {worst}. Create a checklist to mitigate it.")
    if history.get("top_positives"):
        best = sorted(history["top_positives"].items(), key=lambda x: -x[1])[0][0]
        positives.append(f"Consistent strength: {best}")

    summary = "Balanced review based on simple rules and your recent patterns."
    return positives, negatives, suggestions, summary


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
