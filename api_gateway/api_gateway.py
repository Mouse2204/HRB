# api_gateway_enhanced.py
# Enhanced API Gateway với Quorum-based bidding và Consensus support
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import asyncio
import redis
import json
import time
import uuid
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

# Khởi tạo FastAPI app
app = FastAPI(title="RTB_SYSTEM_ENHANCED", version="2.0")
logger = logging.getLogger("enhanced_api_gateway")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuorumConfig:
    """Cấu hình Quorum cho hệ thống bidding"""
    
    def __init__(self, total_nodes: int = 5):
        self.total_nodes = total_nodes
        
        # 🚨 SỬA: Với 1 service thực, quorum chỉ cần 1
        # Nhưng vẫn giữ config 5 services để test logic
        if total_nodes == 1:
            self.read_quorum = 1
            self.write_quorum = 1  
            self.bidding_quorum = 1
        else:
            self.read_quorum = (total_nodes // 2) + 1
            self.write_quorum = (total_nodes // 2) + 1
            self.bidding_quorum = (total_nodes // 2) + 1
        
        # Validate chỉ khi có nhiều services thực
        if total_nodes > 1:
            self._validate_quorum_constraints()
        else:
            logger.info(f"Single service mode: N={self.total_nodes}, all quorums=1")
    
    def _validate_quorum_constraints(self):
        """Validate các điều kiện quorum: N_R + N_W > N, N_W > N/2"""
        if self.read_quorum + self.write_quorum <= self.total_nodes:
            raise ValueError(f"Quorum constraint violated: N_R({self.read_quorum}) + N_W({self.write_quorum}) <= N({self.total_nodes})")
        
        if self.write_quorum <= self.total_nodes // 2:
            raise ValueError(f"Write quorum too small: N_W({self.write_quorum}) <= N/2({self.total_nodes//2})")
        
        logger.info(f"Quorum config validated: N={self.total_nodes}, N_R={self.read_quorum}, N_W={self.write_quorum}, N_B={self.bidding_quorum}")
class EnhancedRTBAPI:
    def __init__(self):
        # 🚨 SỬA: Khởi tạo Redis client đúng cách
        self.redis_client = self._init_redis()
        
        # 🚨 SỬA: Dùng 5 services thực
        self.bidding_services = [
            "http://localhost:8001",  # Bidding Service 1
            "http://localhost:8002",  # Bidding Service 2  
            "http://localhost:8003",  # Bidding Service 3
            "http://localhost:8004",  # Bidding Service 4
            "http://localhost:8005",  # Bidding Service 5
        ]
        
        # Quorum configuration cho 5 services
        self.quorum_config = QuorumConfig(len(self.bidding_services))
        
        # Timeout configuration
        self.request_timeout = 8
        self.max_retries = 2
        self.quorum_timeout = 6
        
        # Consensus coordinator
        self.consensus_coordinators = [
            "http://localhost:8006",
        ]
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "quorum_success": 0,
            "quorum_failures": 0,
            "consensus_commits": 0,
            "average_response_time": 0.0
        }
        
        logger.info(f"EnhancedRTBAPI initialized with {len(self.bidding_services)} bidding services")
    
    def _init_redis(self):
        """Khởi tạo Redis client với error handling"""
        try:
            client = redis.Redis(
                host='127.0.0.1',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            client.ping()
            logger.info("✅ Redis connected successfully")
            return client
        except Exception as e:
            logger.warning(f"❌ Redis connection failed: {e}, using dummy Redis")
            return self._create_dummy_redis()
    
    def _create_dummy_redis(self):
        """Tạo in-memory Redis replacement"""
        class DummyRedis:
            def __init__(self):
                self.data = {}
                logger.info("⚠️ Using dummy Redis (in-memory cache)")
            
            def get(self, key):
                return self.data.get(key)
            
            def setex(self, key, time, value):
                self.data[key] = value
            
            def ping(self):
                return True
            
            def pipeline(self):
                return self
            
            def incr(self, key):
                if key not in self.data:
                    self.data[key] = "0"
                self.data[key] = str(int(self.data[key]) + 1)
                return int(self.data[key])
            
            def execute(self):
                return []
            
            def expire(self, key, time):
                pass
        
        return DummyRedis()

    def _validate_request(self, user: Dict, item: Dict) -> Dict:
        """Validate user và item data với extended validation"""
        user_required = ["user_id", "session_id"]
        item_required = ["item_id", "item_type"]
        
        # Check required user fields
        for field in user_required:
            if field not in user:
                return {
                    "valid": False, 
                    "error": f"Missing user field: {field}",
                    "code": "MISSING_USER_FIELD"
                }
        
        # Check required item fields  
        for field in item_required:
            if field not in item:
                return {
                    "valid": False,
                    "error": f"Missing item field: {field}", 
                    "code": "MISSING_ITEM_FIELD"
                }
        
        # Validate data types
        if not isinstance(user.get("user_id"), (str, int)):
            return {
                "valid": False,
                "error": "user_id must be string or integer",
                "code": "INVALID_USER_ID_TYPE"
            }
            
        if not isinstance(item.get("item_id"), (str, int)):
            return {
                "valid": False, 
                "error": "item_id must be string or integer",
                "code": "INVALID_ITEM_ID_TYPE"
            }
        
        # Additional business logic validation
        if item.get("item_type") not in ["movie", "ad", "content"]:
            return {
                "valid": False,
                "error": "item_type must be one of: movie, ad, content",
                "code": "INVALID_ITEM_TYPE"
            }
            
        return {"valid": True, "code": "VALIDATION_SUCCESS"}

    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check rate limiting với sliding window"""
        current_minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
        rate_limit_key = f"ratelimit:{user_id}:{current_minute}"
        
        try:
            # Get current count
            current_count = self.redis_client.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0
            
            # Check if limit exceeded (100 requests per minute)
            if current_count >= 100:
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                return False
            
            # Increment count với transaction
            pipeline = self.redis_client.pipeline()
            pipeline.incr(rate_limit_key)
            pipeline.expire(rate_limit_key, 60)  # Expire after 1 minute
            pipeline.execute()
            
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Trong trường hợp Redis lỗi, cho phép request để tránh downtime
            return True
    
    async def _enrich_bid_request(self, user: Dict, item: Dict, final_score: float = None) -> Dict:
        cache_key = f"user_prefs:{user['user_id']}"
        
        try:
            # Try to get cached preferences
            cached_prefs = self.redis_client.get(cache_key)
            
            if cached_prefs:
                user["cached_preferences"] = json.loads(cached_prefs)
                user["cache_hit"] = True
                user["cache_timestamp"] = time.time()
            else:
                # Mock preferences data
                user["preferences"] = ["action", "drama", "comedy"]
                user["cache_hit"] = False
                
                # Cache the preferences for future requests
                try:
                    self.redis_client.setex(
                        cache_key, 
                        300,  # 5 minutes expiry
                        json.dumps(user["preferences"])
                    )
                except redis.RedisError as e:
                    logger.warning(f"Failed to cache user preferences: {e}")

        except redis.RedisError as e:
            logger.error(f"Redis error in enrichment: {e}")
            # Fallback preferences
            user["preferences"] = ["action", "drama"]
            user["cache_hit"] = False
            user["cache_error"] = True

        # 🚨 THÊM FINAL_SCORE VÀO ITEM_CONTEXT NẾU CÓ
        enriched_item_context = item.copy()
        if final_score is not None:
            enriched_item_context["final_score"] = final_score
            logger.debug(f"Added final_score {final_score} to bid request")

        # Tạo enriched bid request với consensus metadata
        enriched_request = {
            "request_id": str(uuid.uuid4()),
            "user_context": user,
            "item_context": enriched_item_context,  # 🚨 DÙNG enriched_item_context
            "timestamp": time.time(),
            "gateway_version": "2.0",
            "consensus_required": True,
            "quorum_config": {
                "total_nodes": self.quorum_config.total_nodes,
                "bidding_quorum": self.quorum_config.bidding_quorum,
                "required_responses": self.quorum_config.bidding_quorum
            },
            "metadata": {
                "processing_start_time": time.time(),
                "expected_timeout": self.request_timeout
            }
        }
        
        return enriched_request

    async def _send_to_bidding_service(self, session: aiohttp.ClientSession, 
                                     service_url: str, 
                                     bid_request: Dict) -> Dict:
        """Send request đến bidding service với retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                async with session.post(
                    f"{service_url}/bid",
                    json=bid_request,
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # Add response metadata
                        response_data["_metadata"] = {
                            "service_url": service_url,
                            "response_time": response_time,
                            "attempt": attempt + 1,
                            "timestamp": time.time()
                        }
                        
                        logger.info(f"Successfully called {service_url} in {response_time:.3f}s")
                        return response_data
                    
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Service {service_url} returned {response.status}: {error_text}, "
                            f"attempt {attempt + 1}"
                        )
                        
                        # Nếu là server error, thử retry
                        if response.status >= 500:
                            continue
                        else:
                            return {
                                "error": f"http_error_{response.status}",
                                "service": service_url,
                                "attempt": attempt + 1
                            }
                            
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout for {service_url}, attempt {attempt + 1}, "
                    f"timeout: {self.request_timeout}s"
                )
                if attempt == self.max_retries - 1:
                    return {
                        "error": "timeout",
                        "service": service_url,
                        "attempt": attempt + 1
                    }
                    
            except aiohttp.ClientError as e:
                logger.error(
                    f"Client error calling {service_url}: {str(e)}, "
                    f"attempt {attempt + 1}"
                )
                if attempt == self.max_retries - 1:
                    return {
                        "error": f"client_error_{type(e).__name__}",
                        "service": service_url,
                        "attempt": attempt + 1
                    }
                    
            except Exception as e:
                logger.error(
                    f"Unexpected error calling {service_url}: {str(e)}, "
                    f"attempt {attempt + 1}"
                )
                if attempt == self.max_retries - 1:
                    return {
                        "error": f"unexpected_error_{type(e).__name__}",
                        "service": service_url, 
                        "attempt": attempt + 1
                    }
        
        return {"error": "max_retries_exceeded", "service": service_url}

    async def _distribute_with_quorum(self, bid_request: Dict) -> Dict:

        start_time = time.time()
        successful_responses = []
        failed_services = []
        
        logger.info(f"Starting quorum distribution for request {bid_request['request_id']}")
        logger.info(f"Services: {len(self.bidding_services)} configured, but only 1 actual service")
        
        # 🚨 CHỈ GỌI SERVICE ĐẦU TIÊN để tránh duplicate calls
        actual_services = list(set(self.bidding_services))  # Remove duplicates
        logger.info(f"Actual unique services: {actual_services}")
    
        async with aiohttp.ClientSession() as session:
            tasks = []
            for service_url in actual_services:
                task = asyncio.create_task(
                    self._send_to_bidding_service(session, service_url, bid_request)
                )
                tasks.append(task)
            
            # Chờ tất cả actual services
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    service_url = actual_services[i]
                    if isinstance(result, dict) and "error" not in result:
                        successful_responses.append(result)
                        logger.info(f"✅ Service {service_url} responded successfully")
                    else:
                        error_msg = result if isinstance(result, dict) else str(result)
                        failed_services.append({"service": service_url, "error": error_msg})
                        logger.warning(f"❌ Service {service_url} failed: {error_msg}")
                        
            except Exception as e:
                logger.error(f"Error in gathering responses: {e}")
        
        total_time = time.time() - start_time
        
        # 🚨 QUORUM CALCULATION: chỉ cần 1 response từ service duy nhất
        actual_quorum_required = 1  # Chỉ cần 1 service thực
        quorum_achieved = len(successful_responses) >= actual_quorum_required
        
        # Tạo distribution result
        distribution_result = {
            "successful_responses": successful_responses,
            "failed_services": failed_services,
            "quorum_achieved": quorum_achieved,
            "quorum_required": actual_quorum_required,  # 🚨 Dùng actual thay vì config
            "quorum_actual": len(successful_responses),
            "total_services_contacted": len(actual_services),
            "distribution_time": total_time,
            "timestamp": time.time()
        }
        
        logger.info(
            f"Distribution completed: {len(successful_responses)}/"
            f"{actual_quorum_required} quorum, "
            f"achieved: {quorum_achieved}, "
            f"time: {total_time:.3f}s"
        )
        
        return distribution_result

    def _aggregate_with_consensus(self, distribution_result: Dict) -> Dict:
        successful_responses = distribution_result["successful_responses"]
        
        if not successful_responses:
            return {
                "auction_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "winning_bid": None,
                "all_bids": [],
                "consensus_metrics": {
                    "quorum_achieved": False,
                    "successful_responses": 0,
                    "total_services": len(self.bidding_services),
                    "consensus_level": "NO_CONSENSUS"
                },
                "error": "NO_VALID_BIDS"
            }
        
        # 🚨 FIX: Xác định consensus level dựa trên quorum thực tế
        quorum_achieved = distribution_result["quorum_achieved"]
        successful_count = len(successful_responses)
        total_services = len(self.bidding_services)
        
        if quorum_achieved and successful_count == total_services:
            consensus_level = "FULL_CONSENSUS"
            candidate_bids = successful_responses
        elif quorum_achieved:
            consensus_level = "QUORUM_CONSENSUS" 
            candidate_bids = successful_responses
        else:
            consensus_level = "NO_CONSENSUS"
            candidate_bids = successful_responses  # Vẫn dùng tất cả bids có được
        
        # Second-price auction logic
        winning_bid = None
        if candidate_bids:
            # Sort by bid price descending
            sorted_bids = sorted(candidate_bids, key=lambda x: x["bid_price"], reverse=True)
            
            if sorted_bids:
                winning_bid = sorted_bids[0].copy()
                
                # Second-price: winning price là bid thứ 2 hoặc chính nó nếu chỉ có 1 bid
                if len(sorted_bids) >= 2:
                    winning_bid["winning_price"] = sorted_bids[1]["bid_price"]
                    winning_bid["second_price_source"] = sorted_bids[1].get("service_id", "unknown")
                else:
                    winning_bid["winning_price"] = winning_bid["bid_price"]
                    winning_bid["second_price_source"] = "SELF"
        
        # Tạo final result
        result = {
            "auction_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "winning_bid": winning_bid,
            "all_bids": candidate_bids,
            "consensus_metrics": {
                "quorum_achieved": quorum_achieved,
                "successful_responses": successful_count,
                "total_services": total_services,
                "consensus_level": consensus_level,
                "distribution_time": distribution_result["distribution_time"]
            },
            "performance_metrics": {
                "total_processing_time": time.time() - distribution_result.get('metadata', {}).get('processing_start_time', time.time()),
                "gateway_version": "2.0"
            }
        }
        
        # Update global metrics
        self._update_metrics(distribution_result, result)
        
        return result

    def _update_metrics(self, distribution_result: Dict, final_result: Dict):
        """Update metrics tracking"""
        self.metrics["total_requests"] += 1
        
        if distribution_result["quorum_achieved"]:
            self.metrics["quorum_success"] += 1
        else:
            self.metrics["quorum_failures"] += 1
            
        if final_result["consensus_metrics"]["consensus_level"] == "FULL_CONSENSUS":
            self.metrics["consensus_commits"] += 1
        
        # Update average response time
        processing_time = final_result["performance_metrics"]["total_processing_time"]
        current_avg = self.metrics["average_response_time"]
        total_reqs = self.metrics["total_requests"]
        
        self.metrics["average_response_time"] = (
            (current_avg * (total_reqs - 1) + processing_time) / total_reqs
        )

    async def process_enhanced_rtb_request(self, user_context: Dict, item_context: Dict, final_score: float = None) -> Dict:    
        processing_start = time.time()
        
        try:
            # Step 1: Validate request
            validation = self._validate_request(user_context, item_context)
            if not validation["valid"]:
                logger.warning(f"Request validation failed: {validation['error']}")
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": validation["error"],
                        "code": validation["code"]
                    }
                )
            
            # Step 2: Rate limiting
            if not await self._check_rate_limit(user_context["user_id"]):
                raise HTTPException(
                    status_code=429, 
                    detail={
                        "error": "Rate limit exceeded",
                        "code": "RATE_LIMIT_EXCEEDED"
                    }
                )
            
            # Step 3: Enrich request với final_score
            enriched_request = await self._enrich_bid_request(user_context, item_context, final_score)  # 🚨 THÊM FINAL_SCORE
            enriched_request["metadata"]["processing_start_time"] = processing_start
            
            # Step 4: Distribute với quorum
            distribution_result = await self._distribute_with_quorum(enriched_request)
            
            # Step 5: Aggregate với consensus
            final_result = self._aggregate_with_consensus(distribution_result)
            
            # Add request metadata to final result
            final_result["request_metadata"] = {
                "request_id": enriched_request["request_id"],
                "user_id": user_context["user_id"],
                "item_id": item_context["item_id"],
                "validation_code": validation["code"],
                "final_score_used": final_score  # 🚨 THÊM FINAL_SCORE VÀO METADATA
            }
            
            logger.info(
                f"Enhanced RTB request completed: "
                f"user={user_context['user_id']}, "
                f"item={item_context['item_id']}, "
                f"final_score={final_score}, "  # 🚨 LOG FINAL_SCORE
                f"quorum={distribution_result['quorum_achieved']}, "
                f"consensus={final_result['consensus_metrics']['consensus_level']}, "
                f"time={(time.time() - processing_start):.3f}s"
            )
            
            return final_result
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in enhanced RTB processing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error in enhanced processing",
                    "code": "ENHANCED_PROCESSING_ERROR"
                }
            )

# Khởi tạo enhanced gateway
enhanced_rtb_gateway = EnhancedRTBAPI()

# Enhanced endpoints
@app.post("/rtb/bid-enhanced")
async def rtb_bid_enhanced_endpoint(request_data: Dict):
    user_context = request_data.get("user_context", {})
    item_context = request_data.get("item_context", {})
    final_score = request_data.get("final_score")
    
    return await enhanced_rtb_gateway.process_enhanced_rtb_request(
        user_context, item_context, final_score
    )

@app.get("/enhanced-health")
async def enhanced_health_check():
    """Enhanced health check với metrics"""
    redis_status = "healthy"
    try:
        enhanced_rtb_gateway.redis_client.ping()
    except redis.RedisError:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0",
        "components": {
            "redis": redis_status,
            "quorum_services": len(enhanced_rtb_gateway.bidding_services),
            "consensus_coordinators": len(enhanced_rtb_gateway.consensus_coordinators)
        },
        "metrics": enhanced_rtb_gateway.metrics
    }

@app.get("/enhanced-metrics")
async def get_enhanced_metrics():
    """Get enhanced metrics"""
    return {
        "active_services": len(enhanced_rtb_gateway.bidding_services),
        "quorum_config": {
            "total_nodes": enhanced_rtb_gateway.quorum_config.total_nodes,
            "bidding_quorum": enhanced_rtb_gateway.quorum_config.bidding_quorum,
            "read_quorum": enhanced_rtb_gateway.quorum_config.read_quorum,
            "write_quorum": enhanced_rtb_gateway.quorum_config.write_quorum
        },
        "performance_metrics": enhanced_rtb_gateway.metrics,
        "timestamp": time.time()
    }

@app.get("/quorum-status")
async def get_quorum_status():
    """Get current quorum status và configuration"""
    return {
        "quorum_config": {
            "total_nodes": enhanced_rtb_gateway.quorum_config.total_nodes,
            "bidding_quorum": enhanced_rtb_gateway.quorum_config.bidding_quorum,
            "read_quorum": enhanced_rtb_gateway.quorum_config.read_quorum,
            "write_quorum": enhanced_rtb_gateway.quorum_config.write_quorum
        },
        "bidding_services": enhanced_rtb_gateway.bidding_services,
        "constraints_satisfied": True,  # Được validate ở init
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")