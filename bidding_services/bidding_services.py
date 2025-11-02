# bidding_services.py - UPDATED VERSION
import os
from fastapi import FastAPI
import random
import time
from typing import Dict
import logging

logger = logging.getLogger("bidding_service")

app = FastAPI()

class BiddingService:
    def __init__(self, service_id: int):
        self.service_id = service_id
        # Mỗi service có bidding strategy khác nhau
        self.bidding_strategies = {
            1: "aggressive",    # Bid cao cho high-score items
            2: "conservative",  # Bid thấp hơn
            3: "balanced",      # Bid trung bình
            4: "opportunistic", # Bid cao cho items có match score tốt
            5: "value_based"    # Dựa trên giá trị thực tế
        }
        self.strategy = self.bidding_strategies.get(service_id, "balanced")
    
    def calculate_bid_based_on_score(self, user_context: Dict, item_context: Dict, final_score: float = None) -> Dict:
        base_price = item_context.get("base_price", 3.0)
        
        # Lấy final score từ item context
        if final_score is None:
            final_score = item_context.get("final_score", 0.5)
        min_bid = 2.0
        max_bid = 8.0
        
        # Linear mapping: final_score → bid_price
        base_bid_price = min_bid + (final_score * (max_bid - min_bid))
        
        # Mỗi service có bidding strategy khác nhau
        strategy_adjustments = {
            "aggressive": random.uniform(1.1, 1.3),    # +10-30%
            "conservative": random.uniform(0.9, 1.1),  # -10% to +10%
            "balanced": random.uniform(1.0, 1.2),      # +0-20%
            "opportunistic": random.uniform(0.8, 1.4), # -20% to +40%
            "value_based": random.uniform(0.95, 1.15)  # -5% to +15%
        }
        
        adjustment = strategy_adjustments.get(self.strategy, 1.0)
        bid_price = base_bid_price * adjustment
        
        # Giới hạn bid price trong range hợp lý
        bid_price = max(1.5, min(bid_price, 12.0))
        
        # Thêm small random noise (±2%)
        bid_price *= random.uniform(0.98, 1.02)
        
        logger.debug(f"Service {self.service_id}: final_score={final_score:.3f}, base_bid={base_bid_price:.2f}, adjustment={adjustment:.2f}, final_bid={bid_price:.2f}")
        
        return {
            "service_id": self.service_id,
            "bid_price": round(bid_price, 2),
            "creative_id": f"creative_{self.service_id}_{random.randint(1000, 9999)}",
            "strategy": self.strategy,
            "final_score_used": round(final_score, 4),
            "base_bid_calculated": round(base_bid_price, 2),
            "strategy_adjustment": round(adjustment, 2),
            "timestamp": time.time()
        }

service_id = int(os.getenv("SERVICE_ID", 1))
base_port = 8000 + service_id

bidding_service = BiddingService(service_id)

@app.post("/bid")
async def bid_endpoint(user_context: Dict, item_context: Dict):
    # 🚨 NEW: Extract final_score từ enriched request nếu có
    final_score = None
    if "final_score" in item_context:
        final_score = item_context["final_score"]
    elif "enriched_data" in item_context and "final_score" in item_context["enriched_data"]:
        final_score = item_context["enriched_data"]["final_score"]
    
    return bidding_service.calculate_bid_based_on_score(user_context, item_context, final_score)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service_id": service_id,
        "strategy": bidding_service.strategy,
        "port": base_port
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=base_port, log_level="info")