import httpx
import logging
import random
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

# 模拟数据生成器 (作为兜底)
def get_mock_user_info(phone: str):
    return {
        "phone": phone,
        "name": "测试用户",
        "status": "正常",
        "plan": "5G尊享199套餐",
        "balance": 158.5,
        "arrears": 0.0
    }

async def get_user_info(phone: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.BUSINESS_API_URL}/user/{phone}", timeout=2.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch real user info for {phone}, using mock fallback. Error: {e}")
        return get_mock_user_info(phone)

async def get_plans() -> List[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.BUSINESS_API_URL}/plans", timeout=2.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch real plans, using mock fallback. Error: {e}")
        return [
            {"id": "v129", "name": "畅越129套餐", "price": 129, "data": "30GB", "voice": "500分钟"},
            {"id": "v199", "name": "5G尊享199套餐", "price": 199, "data": "60GB", "voice": "1000分钟"}
        ]

async def create_order(phone: str, plan_id: str, type: str = "new") -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            payload = {"phone": phone, "plan_id": plan_id, "type": type}
            resp = await client.post(f"{settings.BUSINESS_API_URL}/order/create", json=payload, timeout=2.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to create real order, using mock fallback. Error: {e}")
        return {"order_id": f"MOCK{random.randint(1000, 9999)}", "status": "已提交 (Mock)"}

async def get_bill(phone: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.BUSINESS_API_URL}/bill/{phone}", timeout=2.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch real bill, using mock fallback. Error: {e}")
        return {"phone": phone, "month": "2024-03", "amount": 129.0, "status": "已缴纳"}
