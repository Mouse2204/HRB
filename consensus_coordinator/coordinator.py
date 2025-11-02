from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass

from consenus import RaftConsensus, create_raft_consensus
from quorumbid import QuorumBiddingManager, create_quorum_manager

logger = logging.getLogger("consensus_coordinator")

app = FastAPI(title="Consensus Coordinator", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class CoordinatorConfig:
    """Configuration for Consensus Coordinator"""
    node_id: str
    raft_nodes: List[str]
    bidding_nodes: List[str]
    heartbeat_interval: float = 5.0
    cleanup_interval: float = 60.0
    max_pending_auctions: int = 1000

class ConsensusCoordinator:
    """
    Main Consensus Coordinator
    Integrates Quorum Bidding Manager và Raft Consensus
    """
    
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.node_id = config.node_id
        
        # Initialize Raft Consensus
        self.raft_consensus = create_raft_consensus(
            node_id=config.node_id,
            node_addresses=config.raft_nodes
        )
        
        # Initialize Quorum Bidding Manager
        self.quorum_manager = create_quorum_manager(
            node_id=config.node_id,
            node_addresses=config.bidding_nodes
        )
        
        # State management
        self.pending_auctions: Dict[str, Dict[str, Any]] = {}
        self.completed_auctions: Dict[str, Dict[str, Any]] = {}
        self.auction_callbacks: Dict[str, Callable] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Metrics
        self.metrics = {
            "total_auctions_processed": 0,
            "successful_consensus": 0,
            "failed_consensus": 0,
            "raft_commits": 0,
            "quorum_achieved": 0,
            "average_processing_time": 0.0,
            "start_time": time.time()
        }
        
        # Network session
        self.session_timeout = aiohttp.ClientTimeout(total=3.0)
        
        logger.info(f"ConsensusCoordinator initialized for node {self.node_id}")
    
    async def start(self):
        """Start the consensus coordinator"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]
        
        # Register bid commit callback với Raft
        self.raft_consensus.register_bid_commit_callback = self._on_bid_committed
        
        logger.info("ConsensusCoordinator started")
    
    async def stop(self):
        """Stop the consensus coordinator"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("ConsensusCoordinator stopped")
    
    async def process_bid_consensus(self, auction_id: str, bid_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process bid through consensus pipeline: Quorum -> Raft
        
        Args:
            auction_id: ID của auction
            bid_data: Bid data từ bidding service
            
        Returns:
            Dict với consensus result
        """
        start_time = time.time()
        
        # Kiểm tra nếu auction đã tồn tại
        if auction_id in self.completed_auctions:
            logger.info(f"Auction {auction_id} already completed")
            return self.completed_auctions[auction_id]
        
        if auction_id in self.pending_auctions:
            logger.warning(f"Auction {auction_id} already in progress")
            return {
                "status": "pending",
                "auction_id": auction_id,
                "message": "Auction already in progress"
            }
        
        # Validate input
        if not self._validate_bid_data(bid_data):
            raise HTTPException(
                status_code=400,
                detail="Invalid bid data"
            )
        
        logger.info(f"Starting consensus process for auction {auction_id}")
        
        try:
            # Step 1: Quorum-based bidding consensus
            quorum_result = await self.quorum_manager.propose_bid(auction_id, bid_data)
            
            if not quorum_result.get("quorum_achieved", False):
                processing_time = time.time() - start_time
                
                result = {
                    "status": "quorum_failed",
                    "auction_id": auction_id,
                    "consensus_level": "NO_CONSENSUS",
                    "quorum_result": quorum_result,
                    "raft_committed": False,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
                
                self._update_metrics(False, processing_time)
                logger.warning(f"Quorum failed for auction {auction_id}")
                return result
            
            # Step 2: Raft consensus commit
            raft_success = await self.raft_consensus.process_bid_through_raft(auction_id, bid_data)
            
            processing_time = time.time() - start_time
            
            if raft_success:
                # Raft commit thành công
                result = {
                    "status": "committed",
                    "auction_id": auction_id,
                    "consensus_level": "FULL_CONSENSUS",
                    "quorum_result": quorum_result,
                    "raft_committed": True,
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "bid_data": bid_data
                }
                
                # Store in completed auctions
                self.completed_auctions[auction_id] = result
                
                self._update_metrics(True, processing_time)
                logger.info(f"Bid consensus committed for auction {auction_id}")
                
            else:
                # Raft commit thất bại
                result = {
                    "status": "raft_failed",
                    "auction_id": auction_id,
                    "consensus_level": "QUORUM_CONSENSUS",
                    "quorum_result": quorum_result,
                    "raft_committed": False,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
                
                self._update_metrics(False, processing_time)
                logger.warning(f"Raft commit failed for auction {auction_id}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Consensus process failed for auction {auction_id}: {e}")
            
            self._update_metrics(False, processing_time)
            
            raise HTTPException(
                status_code=500,
                detail=f"Consensus processing failed: {str(e)}"
            )
    
    async def _on_bid_committed(self, commit_data: Dict[str, Any]):
        """
        Callback khi bid được commit bởi Raft consensus
        
        Args:
            commit_data: Data từ committed log entry
        """
        try:
            auction_id = commit_data.get("auction_id")
            bid_data = commit_data.get("bid_data")
            
            if not auction_id:
                logger.warning("Received commit without auction_id")
                return
            
            logger.info(f"Raft commit callback for auction {auction_id}")
            
            # Gọi registered callback nếu có
            if auction_id in self.auction_callbacks:
                callback = self.auction_callbacks[auction_id]
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(commit_data)
                    else:
                        callback(commit_data)
                    
                    # Remove callback sau khi gọi
                    del self.auction_callbacks[auction_id]
                    logger.debug(f"Executed callback for auction {auction_id}")
                    
                except Exception as e:
                    logger.error(f"Callback execution failed for auction {auction_id}: {e}")
            
            # Broadcast commit notification đến bidding services
            await self._broadcast_commit_notification(auction_id, commit_data)
            
        except Exception as e:
            logger.error(f"Error in bid commit callback: {e}")
    
    async def _broadcast_commit_notification(self, auction_id: str, commit_data: Dict[str, Any]):
        """
        Broadcast commit notification đến tất cả bidding services
        """
        tasks = []
        
        for node in self.config.bidding_nodes:
            task = self._send_commit_notification(node, auction_id, commit_data)
            tasks.append(task)
        
        # Fire and forget
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.debug(f"Broadcasted commit notification for auction {auction_id}")
            except Exception as e:
                logger.error(f"Error broadcasting commit notification: {e}")
    
    async def _send_commit_notification(self, node: str, auction_id: str, commit_data: Dict[str, Any]):
        """
        Gửi commit notification đến một bidding service
        """
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                await session.post(
                    f"http://{node}/consensus/commit_notification",
                    json={
                        "auction_id": auction_id,
                        "commit_data": commit_data,
                        "committed_by": self.node_id,
                        "timestamp": time.time()
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to send commit notification to {node}: {e}")
    
    def _validate_bid_data(self, bid_data: Dict[str, Any]) -> bool:
        """
        Validate bid data structure
        
        Returns:
            bool: True nếu valid, False nếu không
        """
        try:
            required_fields = ["bid_price", "service_id", "creative_id"]
            
            for field in required_fields:
                if field not in bid_data:
                    logger.warning(f"Missing required field in bid data: {field}")
                    return False
            
            # Validate bid price
            bid_price = bid_data.get("bid_price")
            if not isinstance(bid_price, (int, float)) or bid_price <= 0:
                logger.warning(f"Invalid bid price: {bid_price}")
                return False
            
            # Validate service_id
            service_id = bid_data.get("service_id")
            if not isinstance(service_id, int) or service_id < 0:
                logger.warning(f"Invalid service_id: {service_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Bid data validation error: {e}")
            return False
    
    def _update_metrics(self, success: bool, processing_time: float):
        """
        Update performance metrics
        """
        self.metrics["total_auctions_processed"] += 1
        
        if success:
            self.metrics["successful_consensus"] += 1
            self.metrics["raft_commits"] += 1
            self.metrics["quorum_achieved"] += 1
        else:
            self.metrics["failed_consensus"] += 1
        
        # Update average processing time
        current_avg = self.metrics["average_processing_time"]
        total_processed = self.metrics["total_auctions_processed"]
        
        self.metrics["average_processing_time"] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self.is_running:
            try:
                # Check Raft health
                raft_status = self.raft_consensus.get_raft_status()
                
                # Check Quorum health
                quorum_metrics = self.quorum_manager.get_metrics()
                
                # Log health status
                if raft_status["state"] == "leader":
                    logger.debug(f"Health check - Raft: {raft_status['state']}, Term: {raft_status['current_term']}")
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                # Cleanup expired proposals trong quorum manager
                expired_count = await self.quorum_manager.cleanup_expired_proposals()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired proposals")
                
                # Cleanup old completed auctions (giữ 1000 auctions gần nhất)
                if len(self.completed_auctions) > 1000:
                    # Sort by timestamp và giữ lại 1000 entries mới nhất
                    sorted_auctions = sorted(
                        self.completed_auctions.items(),
                        key=lambda x: x[1].get("timestamp", 0),
                        reverse=True
                    )[:1000]
                    
                    self.completed_auctions = dict(sorted_auctions)
                    logger.debug("Cleaned up old completed auctions")
                
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60.0)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running:
            try:
                # Collect metrics từ các components
                raft_metrics = self.raft_consensus.get_raft_status()
                quorum_metrics = self.quorum_manager.get_metrics()
                
                # Log metrics periodically
                logger.info(
                    f"Metrics - Auctions: {self.metrics['total_auctions_processed']}, "
                    f"Success: {self.metrics['successful_consensus']}, "
                    f"Raft State: {raft_metrics['state']}, "
                    f"Avg Time: {self.metrics['average_processing_time']:.3f}s"
                )
                
                await asyncio.sleep(30.0)  # Log every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30.0)
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status"""
        raft_status = self.raft_consensus.get_raft_status()
        quorum_metrics = self.quorum_manager.get_metrics()
        
        uptime = time.time() - self.metrics["start_time"]
        
        return {
            "node_id": self.node_id,
            "status": "running" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "raft_status": {
                "state": raft_status["state"],
                "current_term": raft_status["current_term"],
                "leader_id": raft_status["leader_id"],
                "log_length": raft_status["log_length"]
            },
            "quorum_status": {
                "active_proposals": quorum_metrics["active_proposals"],
                "committed_bids": quorum_metrics["committed_bids"],
                "total_nodes": quorum_metrics["total_nodes"]
            },
            "coordinator_metrics": self.metrics.copy(),
            "pending_auctions_count": len(self.pending_auctions),
            "completed_auctions_count": len(self.completed_auctions),
            "timestamp": time.time()
        }
    
    def register_auction_callback(self, auction_id: str, callback: Callable):
        """
        Đăng ký callback cho auction
        
        Args:
            auction_id: Auction ID
            callback: Callback function
        """
        self.auction_callbacks[auction_id] = callback
    
    def get_auction_status(self, auction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status của auction cụ thể
        
        Returns:
            Dict với auction status hoặc None nếu không tìm thấy
        """
        if auction_id in self.completed_auctions:
            return self.completed_auctions[auction_id]
        
        if auction_id in self.pending_auctions:
            return {
                "status": "pending",
                "auction_id": auction_id,
                "message": "Auction in consensus process"
            }
        
        return None

# Global coordinator instance
coordinator: Optional[ConsensusCoordinator] = None

@app.on_event("startup")
async def startup_event():
    """Startup event - initialize coordinator"""
    global coordinator
    
    # Configuration từ environment variables
    node_id = "consensus-coordinator-1"  # Trong thực tế lấy từ env var
    raft_nodes = [
        "consensus-coordinator-1:8000",
        "consensus-coordinator-2:8000", 
        "consensus-coordinator-3:8000"
    ]
    bidding_nodes = [
        "bidding-service-1:8000",
        "bidding-service-2:8000",
        "bidding-service-3:8000",
        "bidding-service-4:8000", 
        "bidding-service-5:8000"
    ]
    
    config = CoordinatorConfig(
        node_id=node_id,
        raft_nodes=raft_nodes,
        bidding_nodes=bidding_nodes
    )
    
    coordinator = ConsensusCoordinator(config)
    await coordinator.start()
    
    logger.info("Consensus Coordinator service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - cleanup"""
    global coordinator
    
    if coordinator:
        await coordinator.stop()
        logger.info("Consensus Coordinator service stopped")

# REST API Endpoints
@app.post("/consensus/process_bid")
async def process_bid_endpoint(auction_data: Dict[str, Any]):
    """
    Process bid through consensus pipeline
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    auction_id = auction_data.get("auction_id")
    bid_data = auction_data.get("bid_data")
    
    if not auction_id or not bid_data:
        raise HTTPException(
            status_code=400, 
            detail="Missing auction_id or bid_data"
        )
    
    return await coordinator.process_bid_consensus(auction_id, bid_data)

@app.get("/consensus/status")
async def get_consensus_status():
    """
    Get consensus coordinator status
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return coordinator.get_coordinator_status()

@app.get("/consensus/auction/{auction_id}")
async def get_auction_status(auction_id: str):
    """
    Get status of specific auction
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    status = coordinator.get_auction_status(auction_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Auction not found")
    
    return status

@app.post("/raft/request_vote")
async def raft_request_vote(vote_request: Dict[str, Any]):
    """
    Handle Raft vote request từ nodes khác
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        result = await coordinator.raft_consensus.handle_vote_request(vote_request)
        return result
    except Exception as e:
        logger.error(f"Error handling vote request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/raft/append_entries")
async def raft_append_entries(append_entries: Dict[str, Any]):
    """
    Handle Raft append entries từ leader
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        result = await coordinator.raft_consensus.handle_append_entries(append_entries)
        return result
    except Exception as e:
        logger.error(f"Error handling append entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quorum/propose")
async def quorum_propose(proposal_data: Dict[str, Any]):
    """
    Handle Quorum proposal từ nodes khác
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        proposal_data_dict = proposal_data.get("proposal_data")
        requester = proposal_data.get("requester")
        
        if not proposal_data_dict or not requester:
            raise HTTPException(status_code=400, detail="Missing proposal_data or requester")
        
        success = await coordinator.quorum_manager.receive_proposal(proposal_data_dict, requester)
        
        return {"vote_granted": success}
        
    except Exception as e:
        logger.error(f"Error handling quorum proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consensus/commit_notification")
async def commit_notification(notification_data: Dict[str, Any]):
    """
    Handle commit notification từ nodes khác
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        auction_id = notification_data.get("auction_id")
        commit_data = notification_data.get("commit_data")
        
        logger.info(f"Received commit notification for auction {auction_id}")
        
        # Store in completed auctions
        if coordinator and auction_id:
            coordinator.completed_auctions[auction_id] = {
                "status": "committed",
                "auction_id": auction_id,
                "commit_data": commit_data,
                "timestamp": time.time()
            }
        
        return {"status": "acknowledged"}
        
    except Exception as e:
        logger.error(f"Error handling commit notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if not coordinator:
        return {"status": "unhealthy", "reason": "coordinator_not_initialized"}
    
    try:
        status = coordinator.get_coordinator_status()
        return {
            "status": "healthy",
            "coordinator": status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/metrics")
async def get_metrics():
    """
    Get detailed metrics
    """
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    raft_status = coordinator.raft_consensus.get_raft_status()
    quorum_metrics = coordinator.quorum_manager.get_metrics()
    
    return {
        "coordinator_metrics": coordinator.metrics,
        "raft_metrics": raft_status["metrics"],
        "quorum_metrics": quorum_metrics,
        "system_metrics": {
            "pending_auctions": len(coordinator.pending_auctions),
            "completed_auctions": len(coordinator.completed_auctions),
            "registered_callbacks": len(coordinator.auction_callbacks),
            "uptime": time.time() - coordinator.metrics["start_time"]
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")