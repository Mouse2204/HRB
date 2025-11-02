# quorum_bidding.py
# Quorum-based Bidding Protocol Manager
from typing import Dict, List, Set, Optional, Any
import asyncio
import aiohttp
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger("quorum_bidding")

class BidStatus(Enum):
    """Trạng thái của bid proposal"""
    PENDING = "pending"
    COMMITTED = "committed" 
    REJECTED = "rejected"
    EXPIRED = "expired"

class ConsensusLevel(Enum):
    """Mức độ consensus đạt được"""
    NO_CONSENSUS = "no_consensus"
    QUORUM_CONSENSUS = "quorum_consensus"
    FULL_CONSENSUS = "full_consensus"

@dataclass
class QuorumConfig:
    """Cấu hình Quorum parameters"""
    total_nodes: int = 5
    read_quorum: int = field(init=False)
    write_quorum: int = field(init=False) 
    bidding_quorum: int = field(init=False)
    proposal_timeout: float = 3.0  # seconds
    commit_timeout: float = 5.0   # seconds
    
    def __post_init__(self):
        """Tính toán quorum values sau khi init"""
        self.read_quorum = (self.total_nodes // 2) + 1
        self.write_quorum = (self.total_nodes // 2) + 1
        self.bidding_quorum = (self.total_nodes // 2) + 1
        
        # Validate quorum constraints
        self._validate_quorum_constraints()
    
    def _validate_quorum_constraints(self):
        """Validate quorum constraints"""
        if not (self.read_quorum + self.write_quorum > self.total_nodes):
            raise ValueError(
                f"Quorum constraint failed: N_R({self.read_quorum}) + "
                f"N_W({self.write_quorum}) <= N({self.total_nodes})"
            )
        
        if not (self.write_quorum > self.total_nodes // 2):
            raise ValueError(
                f"Write quorum too small: N_W({self.write_quorum}) <= "
                f"N/2({self.total_nodes // 2})"
            )
        
        logger.info(
            f"QuorumConfig validated: N={self.total_nodes}, "
            f"N_R={self.read_quorum}, N_W={self.write_quorum}, "
            f"N_B={self.bidding_quorum}"
        )

@dataclass 
class BidProposal:
    """Bid proposal data structure"""
    auction_id: str
    bid_data: Dict[str, Any]
    proposer: str
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    status: BidStatus = BidStatus.PENDING
    votes: Set[str] = field(default_factory=set)
    vector_clock: Dict[str, int] = field(default_factory=dict)
    commit_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "auction_id": self.auction_id,
            "bid_data": self.bid_data,
            "proposer": self.proposer,
            "proposal_id": self.proposal_id,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "votes": list(self.votes),
            "vector_clock": self.vector_clock,
            "commit_time": self.commit_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BidProposal':
        """Create from dictionary"""
        proposal = cls(
            auction_id=data["auction_id"],
            bid_data=data["bid_data"],
            proposer=data["proposer"]
        )
        proposal.proposal_id = data.get("proposal_id", proposal.proposal_id)
        proposal.timestamp = data.get("timestamp", proposal.timestamp)
        proposal.status = BidStatus(data.get("status", BidStatus.PENDING.value))
        proposal.votes = set(data.get("votes", []))
        proposal.vector_clock = data.get("vector_clock", {})
        proposal.commit_time = data.get("commit_time")
        return proposal

class QuorumBiddingManager:
    """
    Quorum-based Bidding Manager
    Quản lý bidding protocol với quorum consensus
    """
    
    def __init__(self, node_id: str, all_nodes: List[str], quorum_config: Optional[QuorumConfig] = None):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.quorum_config = quorum_config or QuorumConfig(len(all_nodes))
        
        # State management
        self.proposals: Dict[str, BidProposal] = {}  # auction_id -> proposal
        self.committed_bids: Dict[str, BidProposal] = {}
        self.pending_auctions: Dict[str, asyncio.Event] = {}
        
        # Vector clock for conflict detection
        self.vector_clock = {node: 0 for node in all_nodes}
        self.vector_clock[self.node_id] = 0
        
        # Metrics và monitoring
        self.metrics = {
            "proposals_sent": 0,
            "proposals_received": 0,
            "commits_achieved": 0,
            "quorum_failures": 0,
            "conflicts_detected": 0,
            "average_consensus_time": 0.0
        }
        
        # Network communication
        self.session_timeout = aiohttp.ClientTimeout(total=2.0)
        
        logger.info(f"QuorumBiddingManager initialized for node {node_id}")

    async def propose_bid(self, auction_id: str, bid_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose a new bid và chờ quorum consensus
        
        Args:
            auction_id: ID của auction
            bid_data: Bid data từ local calculation
            
        Returns:
            Dict với consensus result
        """
        start_time = time.time()
        logger.info(f"Node {self.node_id} proposing bid for auction {auction_id}")
        
        # Tạo bid proposal
        proposal = BidProposal(
            auction_id=auction_id,
            bid_data=bid_data,
            proposer=self.node_id,
            vector_clock=self.vector_clock.copy()
        )
        
        # Store proposal locally
        self.proposals[auction_id] = proposal
        proposal.votes.add(self.node_id)  # Self-vote
        
        # Update metrics
        self.metrics["proposals_sent"] += 1
        
        # Tạo event để chờ kết quả
        completion_event = asyncio.Event()
        self.pending_auctions[auction_id] = completion_event
        
        try:
            # Gửi proposal đến các nodes khác
            quorum_achieved = await self._broadcast_proposal(proposal)
            
            if quorum_achieved:
                # Đạt quorum, commit bid
                await self._commit_bid(proposal)
                consensus_time = time.time() - start_time
                
                logger.info(
                    f"Bid committed for auction {auction_id} with quorum. "
                    f"Consensus time: {consensus_time:.3f}s"
                )
                
                result = {
                    "consensus_status": "committed",
                    "auction_id": auction_id,
                    "proposal_id": proposal.proposal_id,
                    "quorum_achieved": True,
                    "votes_received": len(proposal.votes),
                    "quorum_required": self.quorum_config.bidding_quorum,
                    "consensus_time": consensus_time,
                    "commit_timestamp": proposal.commit_time,
                    "bid_data": bid_data
                }
                
                # Update consensus metrics
                self._update_consensus_metrics(consensus_time, True)
                
            else:
                # Không đạt quorum
                consensus_time = time.time() - start_time
                proposal.status = BidStatus.REJECTED
                
                logger.warning(
                    f"Bid rejected for auction {auction_id}. "
                    f"Quorum not achieved. Time: {consensus_time:.3f}s"
                )
                
                result = {
                    "consensus_status": "rejected", 
                    "auction_id": auction_id,
                    "proposal_id": proposal.proposal_id,
                    "quorum_achieved": False,
                    "votes_received": len(proposal.votes),
                    "quorum_required": self.quorum_config.bidding_quorum,
                    "consensus_time": consensus_time,
                    "bid_data": bid_data
                }
                
                # Update failure metrics
                self.metrics["quorum_failures"] += 1
                self._update_consensus_metrics(consensus_time, False)
                
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Proposal timeout for auction {auction_id}")
            proposal.status = BidStatus.EXPIRED
            
            return {
                "consensus_status": "timeout",
                "auction_id": auction_id,
                "proposal_id": proposal.proposal_id,
                "quorum_achieved": False,
                "error": "proposal_timeout"
            }
            
        except Exception as e:
            logger.error(f"Proposal failed for auction {auction_id}: {str(e)}")
            proposal.status = BidStatus.REJECTED
            
            return {
                "consensus_status": "error",
                "auction_id": auction_id, 
                "proposal_id": proposal.proposal_id,
                "quorum_achieved": False,
                "error": str(e)
            }
            
        finally:
            # Cleanup
            if auction_id in self.pending_auctions:
                del self.pending_auctions[auction_id]
            completion_event.set()

    async def _broadcast_proposal(self, proposal: BidProposal) -> bool:
        """
        Broadcast proposal đến tất cả nodes và chờ quorum votes
        
        Returns:
            bool: True nếu đạt quorum, False nếu không
        """
        tasks = []
        target_nodes = [node for node in self.all_nodes if node != self.node_id]
        
        logger.debug(f"Broadcasting proposal {proposal.proposal_id} to {len(target_nodes)} nodes")
        
        # Tạo tasks để gửi proposal
        for node in target_nodes:
            task = self._send_proposal_to_node(node, proposal)
            tasks.append(task)
        
        # Chờ responses với timeout
        try:
            done, pending = await asyncio.wait(
                tasks, 
                timeout=self.quorum_config.proposal_timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Đếm successful votes
            successful_votes = 0
            for task in done:
                try:
                    if task.result():
                        successful_votes += 1
                except Exception as e:
                    logger.debug(f"Node vote failed: {e}")
                    continue
            
            total_votes = successful_votes + 1  # +1 for self-vote
            
            # Kiểm tra quorum
            quorum_achieved = total_votes >= self.quorum_config.bidding_quorum
            
            logger.debug(
                f"Proposal {proposal.proposal_id}: {total_votes}/"
                f"{self.quorum_config.bidding_quorum} votes, "
                f"quorum_achieved={quorum_achieved}"
            )
            
            return quorum_achieved
            
        except asyncio.TimeoutError:
            logger.warning(f"Proposal broadcast timeout for {proposal.proposal_id}")
            return False

    async def _send_proposal_to_node(self, node: str, proposal: BidProposal) -> bool:
        """
        Gửi proposal đến một node cụ thể và chờ vote
        
        Args:
            node: Target node address
            proposal: Bid proposal to send
            
        Returns:
            bool: True nếu node vote accept, False nếu reject hoặc error
        """
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.post(
                    f"http://{node}/consensus/propose",
                    json={
                        "proposal_data": proposal.to_dict(),
                        "requester": self.node_id
                    },
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        vote_granted = result.get("vote_granted", False)
                        
                        if vote_granted:
                            # Thêm vote vào proposal
                            self.proposals[proposal.auction_id].votes.add(node)
                            logger.debug(f"Node {node} voted ACCEPT for proposal {proposal.proposal_id}")
                        else:
                            logger.debug(f"Node {node} voted REJECT for proposal {proposal.proposal_id}")
                            
                        return vote_granted
                    else:
                        logger.warning(f"Node {node} returned status {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.debug(f"Timeout when sending proposal to {node}")
            return False
            
        except aiohttp.ClientError as e:
            logger.debug(f"Network error sending to {node}: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error sending to {node}: {e}")
            return False

    async def receive_proposal(self, proposal_data: Dict[str, Any], requester: str) -> bool:
        """
        Nhận proposal từ node khác và vote
        
        Args:
            proposal_data: Proposal data từ node gửi
            requester: Node ID của node gửi
            
        Returns:
            bool: True nếu accept proposal, False nếu reject
        """
        try:
            # Parse proposal
            proposal = BidProposal.from_dict(proposal_data)
            auction_id = proposal.auction_id
            
            # Update metrics
            self.metrics["proposals_received"] += 1
            
            logger.debug(f"Node {self.node_id} received proposal {proposal.proposal_id} from {requester}")
            
            # Validate proposal
            if not self._validate_proposal(proposal):
                logger.warning(f"Proposal validation failed for {proposal.proposal_id}")
                return False
            
            # Conflict detection với vector clock
            if not self._check_vector_clock_conflict(proposal.vector_clock):
                logger.warning(f"Vector clock conflict detected for {proposal.proposal_id}")
                self.metrics["conflicts_detected"] += 1
                return False
            
            # Kiểm tra nếu đã có proposal cho auction này
            if auction_id in self.proposals:
                existing_proposal = self.proposals[auction_id]
                
                # Ưu tiên proposal với timestamp sớm hơn
                if proposal.timestamp < existing_proposal.timestamp:
                    logger.info(f"Replacing existing proposal with newer one for {auction_id}")
                    self.proposals[auction_id] = proposal
                else:
                    logger.debug(f"Keeping existing proposal for {auction_id}")
            else:
                # Store new proposal
                self.proposals[auction_id] = proposal
            
            # Vote accept proposal
            self.proposals[auction_id].votes.add(self.node_id)
            
            # Update vector clock
            self.vector_clock[requester] = max(
                self.vector_clock.get(requester, 0),
                proposal.vector_clock.get(requester, 0)
            )
            self.vector_clock[self.node_id] += 1
            
            logger.debug(f"Node {self.node_id} voted ACCEPT for proposal {proposal.proposal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing proposal from {requester}: {e}")
            return False

    async def _commit_bid(self, proposal: BidProposal):
        """
        Commit bid sau khi đạt quorum
        """
        proposal.status = BidStatus.COMMITTED
        proposal.commit_time = time.time()
        
        # Store in committed bids
        self.committed_bids[proposal.auction_id] = proposal
        
        # Update vector clock
        self.vector_clock[self.node_id] += 1
        
        # Broadcast commit message đến các nodes khác
        asyncio.create_task(self._broadcast_commit(proposal))
        
        # Update metrics
        self.metrics["commits_achieved"] += 1
        
        logger.info(f"Bid committed for auction {proposal.auction_id}")

    async def _broadcast_commit(self, proposal: BidProposal):
        """
        Broadcast commit message đến các nodes khác
        """
        tasks = []
        target_nodes = [node for node in self.all_nodes if node != self.node_id]
        
        for node in target_nodes:
            task = self._send_commit_to_node(node, proposal)
            tasks.append(task)
        
        # Fire and forget - không chờ response
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_commit_to_node(self, node: str, proposal: BidProposal):
        """
        Gửi commit message đến node
        """
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                await session.post(
                    f"http://{node}/consensus/commit",
                    json={
                        "proposal_data": proposal.to_dict(),
                        "committer": self.node_id
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to send commit to {node}: {e}")

    def _validate_proposal(self, proposal: BidProposal) -> bool:
        """
        Validate proposal data
        
        Returns:
            bool: True nếu proposal valid, False nếu không
        """
        try:
            # Check required fields
            if not proposal.auction_id or not proposal.bid_data:
                return False
            
            # Check bid data structure
            bid_data = proposal.bid_data
            required_bid_fields = ["bid_price", "service_id"]
            
            for field in required_bid_fields:
                if field not in bid_data:
                    return False
            
            # Validate bid price
            bid_price = bid_data.get("bid_price")
            if not isinstance(bid_price, (int, float)) or bid_price <= 0:
                return False
            
            # Validate timestamp (không được trong tương lai)
            if proposal.timestamp > time.time() + 60:  # Cho phép 60s clock skew
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Proposal validation error: {e}")
            return False

    def _check_vector_clock_conflict(self, incoming_clock: Dict[str, int]) -> bool:
        """
        Kiểm tra vector clock conflict
        
        Returns:
            bool: True nếu không có conflict, False nếu có conflict
        """
        try:
            for node, timestamp in incoming_clock.items():
                if node in self.vector_clock and timestamp < self.vector_clock[node]:
                    logger.warning(
                        f"Vector clock conflict for node {node}: "
                        f"incoming={timestamp}, local={self.vector_clock[node]}"
                    )
                    return False
            return True
        except Exception as e:
            logger.error(f"Vector clock check error: {e}")
            return False

    def _update_consensus_metrics(self, consensus_time: float, success: bool):
        """
        Update consensus performance metrics
        """
        if success:
            current_avg = self.metrics["average_consensus_time"]
            total_success = self.metrics["commits_achieved"]
            
            self.metrics["average_consensus_time"] = (
                (current_avg * (total_success - 1) + consensus_time) / total_success
            )

    def get_proposal_status(self, auction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status của proposal
        
        Returns:
            Dict với proposal status hoặc None nếu không tìm thấy
        """
        if auction_id not in self.proposals:
            return None
        
        proposal = self.proposals[auction_id]
        
        return {
            "auction_id": auction_id,
            "proposal_id": proposal.proposal_id,
            "status": proposal.status.value,
            "votes": list(proposal.votes),
            "votes_count": len(proposal.votes),
            "quorum_required": self.quorum_config.bidding_quorum,
            "quorum_achieved": len(proposal.votes) >= self.quorum_config.bidding_quorum,
            "timestamp": proposal.timestamp,
            "commit_time": proposal.commit_time
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        """
        return {
            **self.metrics,
            "vector_clock": self.vector_clock.copy(),
            "active_proposals": len(self.proposals),
            "committed_bids": len(self.committed_bids),
            "node_id": self.node_id,
            "total_nodes": len(self.all_nodes),
            "quorum_config": {
                "total_nodes": self.quorum_config.total_nodes,
                "bidding_quorum": self.quorum_config.bidding_quorum,
                "read_quorum": self.quorum_config.read_quorum,
                "write_quorum": self.quorum_config.write_quorum
            }
        }

    async def cleanup_expired_proposals(self, max_age: float = 300.0):
        """
        Clean up expired proposals (older than max_age seconds)
        """
        current_time = time.time()
        expired_auctions = []
        
        for auction_id, proposal in self.proposals.items():
            if (current_time - proposal.timestamp) > max_age:
                expired_auctions.append(auction_id)
        
        for auction_id in expired_auctions:
            del self.proposals[auction_id]
            logger.debug(f"Cleaned up expired proposal for auction {auction_id}")
        
        return len(expired_auctions)

# Utility functions
def create_quorum_manager(node_id: str, node_addresses: List[str]) -> QuorumBiddingManager:
    """
    Factory function để tạo QuorumBiddingManager
    """
    quorum_config = QuorumConfig(total_nodes=len(node_addresses))
    return QuorumBiddingManager(node_id, node_addresses, quorum_config)

if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    async def test_quorum():
        nodes = ["node1", "node2", "node3", "node4", "node5"]
        manager = create_quorum_manager("node1", nodes)
        
        bid_data = {
            "bid_price": 2.5,
            "service_id": 1,
            "creative_id": "test_creative",
            "timestamp": time.time()
        }
        
        result = await manager.propose_bid("test_auction", bid_data)
        print("Test result:", result)
    
    asyncio.run(test_quorum())