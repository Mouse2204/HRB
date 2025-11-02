# raft_consensus.py
# Raft Consensus Protocol Implementation
import enum
import asyncio
import time
import random
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import aiohttp
import logging
from collections import defaultdict
import json

logger = logging.getLogger("raft_consensus")

class NodeState(enum.Enum):
    """Trạng thái của node trong Raft cluster"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class LogEntry:
    """Một entry trong Raft log"""
    
    def __init__(self, term: int, index: int, command: str, data: Dict[str, Any]):
        self.term = term
        self.index = index
        self.command = command
        self.data = data
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
            "data": self.data,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        return cls(
            term=data["term"],
            index=data["index"],
            command=data["command"],
            data=data["data"]
        )

@dataclass
class RaftConfig:
    """Cấu hình Raft parameters"""
    election_timeout_min: float = 1.5  # seconds
    election_timeout_max: float = 3.0  # seconds
    heartbeat_interval: float = 0.5    # seconds
    commit_timeout: float = 2.0        # seconds
    max_log_entries: int = 10000       # maximum log entries to keep
    
    def get_random_election_timeout(self) -> float:
        """Tạo random election timeout"""
        return random.uniform(self.election_timeout_min, self.election_timeout_max)

# consenus_final.py - Completely clean version
# (Copy to replace the current consenus.py)

# ... [giữ nguyên toàn bộ code trước __init__ method] ...

class RaftConsensus:
    def __init__(self, node_id: str, all_nodes: List[str], config: Optional[RaftConfig] = None):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.config = config or RaftConfig()
        
        # 🚨 CRITICAL: Initialize timers FIRST
        self.election_timer = None
        self.heartbeat_timer = None
        
        # Single node mode
        if len(all_nodes) == 1:
            self.quorum_size = 1
            self.state = NodeState.LEADER
            self.leader_id = self.node_id
            self.current_term = 1
            self.next_index = {}
            self.match_index = {}
            self.last_heartbeat = time.time()
            self._start_heartbeat_timer()
            logger.info(f"Single node mode - {node_id} automatically became LEADER")
        else:
            self.quorum_size = (len(all_nodes) // 2) + 1
            self.state = NodeState.FOLLOWER
            self.leader_id = None
            self.current_term = 0
            self._start_election_timer()
        
        # Rest of initialization...
        self.voted_for = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.election_timeout = self.config.get_random_election_timeout()
        self.auction_state: Dict[str, Any] = {}
        self.bid_commit_callbacks = {}
        self.session_timeout = aiohttp.ClientTimeout(total=1.0)
        self.metrics = {
            "state_changes": 0, "elections_started": 0, "logs_appended": 0,
            "heartbeats_sent": 0, "commit_count": 0, "term_changes": 0
        }
        logger.info(f"RaftConsensus initialized for node {node_id}")

    
    def _start_election_timer(self):
        """Bắt đầu election timer"""
        if self.election_timer and not self.election_timer.done():
            self.election_timer.cancel()
        
        self.election_timer = asyncio.create_task(self._election_timeout_handler())
    
    async def _election_timeout_handler(self):
        """Xử lý election timeout - bắt đầu election nếu cần"""
        try:
            await asyncio.sleep(self.election_timeout)
            
            # Nếu không nhận được heartbeat từ leader, bắt đầu election
            if self.state != NodeState.LEADER:
                time_since_heartbeat = time.time() - self.last_heartbeat
                if time_since_heartbeat > self.election_timeout:
                    logger.info(f"Election timeout - starting election. Last heartbeat: {time_since_heartbeat:.2f}s ago")
                    await self.start_election()
            
            # Restart timer
            self._start_election_timer()
            
        except asyncio.CancelledError:
            logger.debug("Election timer cancelled")
        except Exception as e:
            logger.error(f"Election timer error: {e}")
            self._start_election_timer()
    
    async def start_election(self):
        """
        Bắt đầu leader election
        Chuyển sang candidate state và request votes
        """
        if self.state == NodeState.LEADER:
            return
        
        # Chuyển sang candidate state
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id  # Vote cho chính mình
        self.leader_id = None
        
        # Update metrics
        self.metrics["elections_started"] += 1
        self.metrics["term_changes"] += 1
        self.metrics["state_changes"] += 1
        
        logger.info(f"Node {self.node_id} starting election for term {self.current_term}")
        
        # Reset election timeout
        self.election_timeout = self.config.get_random_election_timeout()
        self._start_election_timer()
        
        # Prepare vote request
        last_log_index = len(self.log) - 1
        last_log_term = self.log[-1].term if self.log else 0
        
        vote_request = {
            "term": self.current_term,
            "candidate_id": self.node_id,
            "last_log_index": last_log_index,
            "last_log_term": last_log_term,
            "timestamp": time.time()
        }
        
        # Request votes từ các nodes khác
        vote_count = 1  # Self vote
        
        tasks = []
        target_nodes = [node for node in self.all_nodes if node != self.node_id]
        
        for node in target_nodes:
            task = self._request_vote_from_node(node, vote_request)
            tasks.append(task)
        
        # Chờ vote responses
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, dict) and result.get("vote_granted", False):
                        vote_count += 1
                        
                        # Kiểm tra nếu đã đạt quorum
                        if vote_count >= self.quorum_size:
                            await self._become_leader()
                            return
            except Exception as e:
                logger.error(f"Error gathering votes: {e}")
        
        # Nếu không đạt quorum, trở lại follower
        logger.info(f"Election failed for term {self.current_term}. Votes: {vote_count}/{self.quorum_size}")
        await self._return_to_follower()
    
    async def _request_vote_from_node(self, node: str, vote_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gửi vote request đến node và chờ response
        
        Returns:
            Dict với vote response
        """
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.post(
                    f"http://{node}/raft/request_vote",
                    json=vote_request,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Node {node} returned status {response.status} for vote request")
                        return {"vote_granted": False}
                        
        except asyncio.TimeoutError:
            logger.debug(f"Timeout requesting vote from {node}")
            return {"vote_granted": False}
            
        except Exception as e:
            logger.error(f"Error requesting vote from {node}: {e}")
            return {"vote_granted": False}
    
    async def handle_vote_request(self, vote_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý vote request từ candidate khác
        
        Returns:
            Dict với vote decision
        """
        try:
            candidate_term = vote_request["term"]
            candidate_id = vote_request["candidate_id"]
            candidate_last_log_index = vote_request["last_log_index"]
            candidate_last_log_term = vote_request["last_log_term"]
            
            logger.debug(f"Node {self.node_id} received vote request from {candidate_id} for term {candidate_term}")
            
            # Reply false if term < currentTerm
            if candidate_term < self.current_term:
                return {
                    "term": self.current_term,
                    "vote_granted": False,
                    "reason": "term_too_old"
                }
            
            # Nếu candidate term lớn hơn, update current term và trở thành follower
            if candidate_term > self.current_term:
                self.current_term = candidate_term
                self.voted_for = None
                self.state = NodeState.FOLLOWER
                self.metrics["term_changes"] += 1
            
            # Kiểm tra vote eligibility
            can_vote = (
                self.voted_for is None or 
                self.voted_for == candidate_id
            )
            
            # Kiểm tra log completeness
            log_ok = self._is_candidate_log_up_to_date(
                candidate_last_log_index, 
                candidate_last_log_term
            )
            
            vote_granted = can_vote and log_ok
            
            if vote_granted:
                self.voted_for = candidate_id
                self.last_heartbeat = time.time()
                logger.info(f"Node {self.node_id} granted vote to {candidate_id} for term {candidate_term}")
            else:
                logger.debug(f"Node {self.node_id} rejected vote for {candidate_id}. can_vote={can_vote}, log_ok={log_ok}")
            
            return {
                "term": self.current_term,
                "vote_granted": vote_granted,
                "node_id": self.node_id
            }
            
        except Exception as e:
            logger.error(f"Error handling vote request: {e}")
            return {
                "term": self.current_term,
                "vote_granted": False,
                "error": str(e)
            }
    
    def _is_candidate_log_up_to_date(self, candidate_last_log_index: int, candidate_last_log_term: int) -> bool:
        """
        Kiểm tra xem candidate's log có up-to-date hơn không
        
        Returns:
            bool: True nếu candidate's log ít nhất mới bằng local log
        """
        # So sánh theo Raft paper
        last_log_index = len(self.log) - 1
        last_log_term = self.log[-1].term if self.log else 0
        
        if candidate_last_log_term > last_log_term:
            return True
        elif candidate_last_log_term == last_log_term:
            return candidate_last_log_index >= last_log_index
        else:
            return False
    
    async def _become_leader(self):
        """
        Chuyển sang leader state và bắt đầu gửi heartbeats
        """
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        self.metrics["state_changes"] += 1
        
        # Initialize leader state
        self.next_index = {node: len(self.log) for node in self.all_nodes if node != self.node_id}
        self.match_index = {node: 0 for node in self.all_nodes if node != self.node_id}
        
        # Cancel election timer
        if self.election_timer and not self.election_timer.done():
            self.election_timer.cancel()
        
        # Start heartbeat timer
        self._start_heartbeat_timer()
        
        logger.info(f"Node {self.node_id} became LEADER for term {self.current_term}")
        
        # Gửi immediate heartbeat để establish authority
        await self._send_heartbeats()
    
    def _start_heartbeat_timer(self):
        """Bắt đầu heartbeat timer (chỉ cho leader)"""
        if self.heartbeat_timer and not self.heartbeat_timer.done():
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = asyncio.create_task(self._heartbeat_timer_handler())
    
    async def _heartbeat_timer_handler(self):
        """Xử lý heartbeat timer - gửi heartbeats định kỳ"""
        try:
            while self.state == NodeState.LEADER:
                await asyncio.sleep(self.config.heartbeat_interval)
                await self._send_heartbeats()
        except asyncio.CancelledError:
            logger.debug("Heartbeat timer cancelled")
        except Exception as e:
            logger.error(f"Heartbeat timer error: {e}")
            if self.state == NodeState.LEADER:
                self._start_heartbeat_timer()
    
    async def _send_heartbeats(self):
        """Gửi heartbeat (AppendEntries) đến tất cả followers"""
        if self.state != NodeState.LEADER:
            return
        
        tasks = []
        target_nodes = [node for node in self.all_nodes if node != self.node_id]
        
        for node in target_nodes:
            task = self._send_heartbeat_to_node(node)
            tasks.append(task)
        
        # Fire and forget - không chờ response
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.metrics["heartbeats_sent"] += len(tasks)
            except Exception as e:
                logger.error(f"Error sending heartbeats: {e}")
    
    async def _send_heartbeat_to_node(self, node: str):
        """
        Gửi heartbeat đến một node cụ thể
        """
        try:
            # Tạo AppendEntries message
            prev_log_index = self.next_index[node] - 1
            prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
            
            # Lấy log entries để gửi (trong heartbeat thường là empty)
            entries = []
            if self.next_index[node] < len(self.log):
                entries = [entry.to_dict() for entry in self.log[self.next_index[node]:]]
            
            append_entries_msg = {
                "term": self.current_term,
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": entries,
                "leader_commit": self.commit_index,
                "timestamp": time.time()
            }
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.post(
                    f"http://{node}/raft/append_entries",
                    json=append_entries_msg,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get("success", False):
                            # Update match index và next index
                            if entries:
                                self.match_index[node] = prev_log_index + len(entries)
                                self.next_index[node] = self.match_index[node] + 1
                            
                            # Kiểm tra commit condition
                            await self._update_commit_index()
                        else:
                            # Log inconsistency, decrement next_index
                            if self.next_index[node] > 0:
                                self.next_index[node] -= 1
                    
                    else:
                        logger.warning(f"Node {node} returned status {response.status} for heartbeat")
                        
        except Exception as e:
            logger.debug(f"Error sending heartbeat to {node}: {e}")
    
    async def handle_append_entries(self, append_entries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý AppendEntries message từ leader
        
        Returns:
            Dict với response
        """
        try:
            leader_term = append_entries["term"]
            leader_id = append_entries["leader_id"]
            prev_log_index = append_entries["prev_log_index"]
            prev_log_term = append_entries["prev_log_term"]
            entries = append_entries["entries"]
            leader_commit = append_entries["leader_commit"]
            
            # Reset election timer
            self.last_heartbeat = time.time()
            
            # Reply false if term < currentTerm
            if leader_term < self.current_term:
                return {
                    "term": self.current_term,
                    "success": False,
                    "reason": "term_too_old"
                }
            
            # Nếu leader term lớn hơn, update current term
            if leader_term > self.current_term:
                self.current_term = leader_term
                self.voted_for = None
                self.metrics["term_changes"] += 1
            
            # Trở thành follower nếu chưa phải
            if self.state != NodeState.FOLLOWER:
                self.state = NodeState.FOLLOWER
                self.metrics["state_changes"] += 1
            
            self.leader_id = leader_id
            
            # Kiểm tra log consistency
            log_ok = self._check_log_consistency(prev_log_index, prev_log_term)
            
            if not log_ok:
                return {
                    "term": self.current_term,
                    "success": False,
                    "reason": "log_inconsistent"
                }
            
            # Xử lý log entries
            if entries:
                # Xóa conflicting entries và append new entries
                self.log = self.log[:prev_log_index + 1]
                for entry_data in entries:
                    entry = LogEntry.from_dict(entry_data)
                    self.log.append(entry)
                    self.metrics["logs_appended"] += 1
            
            # Update commit index
            if leader_commit > self.commit_index:
                old_commit_index = self.commit_index
                self.commit_index = min(leader_commit, len(self.log) - 1)
                
                # Apply committed entries
                if self.commit_index > old_commit_index:
                    await self._apply_committed_entries(old_commit_index + 1, self.commit_index)
            
            return {
                "term": self.current_term,
                "success": True,
                "node_id": self.node_id
            }
            
        except Exception as e:
            logger.error(f"Error handling append entries: {e}")
            return {
                "term": self.current_term,
                "success": False,
                "error": str(e)
            }
    
    def _check_log_consistency(self, prev_log_index: int, prev_log_term: int) -> bool:
        """
        Kiểm tra log consistency với previous log entry
        
        Returns:
            bool: True nếu log consistent, False nếu không
        """
        # Nếu prev_log_index nằm ngoài log range
        if prev_log_index >= len(self.log):
            return False
        
        # Nếu prev_log_index là -1 (log empty), luôn consistent
        if prev_log_index == -1:
            return True
        
        # Kiểm tra term khớp
        return self.log[prev_log_index].term == prev_log_term
    
    async def _update_commit_index(self):
        """
        Update commit index dựa trên match_index của followers
        """
        if self.state != NodeState.LEADER:
            return
        
        # Tìm commit index mới
        match_indexes = list(self.match_index.values())
        match_indexes.append(len(self.log) - 1)  # Bao gồm cả leader
        
        match_indexes.sort(reverse=True)
        
        # Commit index là index mà đa số nodes đã replicate
        new_commit_index = match_indexes[self.quorum_size - 1]
        
        # Chỉ commit entries từ current term
        if new_commit_index > self.commit_index and self.log[new_commit_index].term == self.current_term:
            old_commit_index = self.commit_index
            self.commit_index = new_commit_index
            self.metrics["commit_count"] += 1
            
            # Apply committed entries
            if self.commit_index > old_commit_index:
                await self._apply_committed_entries(old_commit_index + 1, self.commit_index)
    
    async def _apply_committed_entries(self, start_index: int, end_index: int):
        """
        Apply committed entries to state machine
        
        Args:
            start_index: Start index (inclusive)
            end_index: End index (inclusive)
        """
        for i in range(start_index, end_index + 1):
            if i < len(self.log):
                entry = self.log[i]
                await self._apply_log_entry(entry)
                self.last_applied = i
    
    async def _apply_log_entry(self, entry: LogEntry):
        """
        Apply một log entry to state machine
        
        Args:
            entry: Log entry to apply
        """
        try:
            if entry.command == "BID_COMMIT":
                # Apply bid commit to auction state
                auction_id = entry.data.get("auction_id")
                bid_data = entry.data.get("bid_data")
                
                if auction_id and bid_data:
                    self.auction_state[auction_id] = {
                        "bid_data": bid_data,
                        "commit_index": entry.index,
                        "commit_term": entry.term,
                        "commit_time": time.time()
                    }
                    
                    # Gọi callback nếu có
                    if auction_id in self.bid_commit_callbacks:
                        callback = self.bid_commit_callbacks[auction_id]
                        if asyncio.iscoroutinefunction(callback):
                            await callback(entry.data)
                        else:
                            callback(entry.data)
                        
                        # Remove callback sau khi gọi
                        del self.bid_commit_callbacks[auction_id]
                    
                    logger.info(f"Applied BID_COMMIT for auction {auction_id} at index {entry.index}")
            
            elif entry.command == "CONFIG_UPDATE":
                # Apply configuration update
                config_data = entry.data.get("config")
                logger.info(f"Applied CONFIG_UPDATE: {config_data}")
            
            else:
                logger.warning(f"Unknown log command: {entry.command}")
                
        except Exception as e:
            logger.error(f"Error applying log entry {entry.index}: {e}")
    
    async def process_bid_through_raft(self, auction_id: str, bid_data: Dict[str, Any]) -> bool:
        # 🚨 SINGLE NODE FIX: Allow processing for single node leader
        if self.state != NodeState.LEADER:
            logger.warning(f"Node {self.node_id} is not leader, cannot process bid")
            return False
        
        # Tạo log entry
        log_entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command="BID_COMMIT",
            data={
                "auction_id": auction_id,
                "bid_data": bid_data,
                "proposer": self.node_id,
                "timestamp": time.time()
            }
        )
        
        # Append to local log
        self.log.append(log_entry)
        self.metrics["logs_appended"] += 1
        
        logger.info(f"Leader {self.node_id} appended BID_COMMIT for auction {auction_id} at index {len(self.log) - 1}")
        
        # Replicate log đến followers (sẽ auto-commit cho single node)
        replication_success = await self._replicate_log_entry(log_entry)
        
        return replication_success
    
    async def _replicate_log_entry(self, log_entry: LogEntry) -> bool:
        # 🚨 SINGLE NODE FIX: Auto-commit for single node
        if len(self.all_nodes) == 1:
            # Single node automatically commits
            self.commit_index = log_entry.index
            await self._apply_committed_entries(self.last_applied + 1, self.commit_index)
            logger.info(f"Single node auto-committed log entry at index {log_entry.index}")
            return True
        
        # Original multi-node code
        tasks = []
        target_nodes = [node for node in self.all_nodes if node != self.node_id]
        
        for node in target_nodes:
            task = self._replicate_to_node(node, log_entry)
            tasks.append(task)
        
        # Chờ replication results
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful_replications = sum(1 for r in results if r is True)
                
                # Kiểm tra quorum (bao gồm cả leader)
                total_successful = successful_replications + 1  # +1 for leader
                quorum_achieved = total_successful >= self.quorum_size
                
                if quorum_achieved:
                    # Update commit index
                    await self._update_commit_index()
                    return True
                else:
                    logger.warning(f"Log replication failed for index {log_entry.index}. Success: {total_successful}/{self.quorum_size}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error in log replication: {e}")
                return False
        
        return True  # Nếu không có followers, chỉ leader cũng đạt quorum
    
    async def _replicate_to_node(self, node: str, log_entry: LogEntry) -> bool:
        """
        Replicate log entry đến một node cụ thể
        
        Returns:
            bool: True nếu replication thành công, False nếu thất bại
        """
        try:
            prev_log_index = self.next_index[node] - 1
            prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
            
            append_entries_msg = {
                "term": self.current_term,
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": [log_entry.to_dict()],
                "leader_commit": self.commit_index,
                "timestamp": time.time()
            }
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.post(
                    f"http://{node}/raft/append_entries",
                    json=append_entries_msg,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        success = result.get("success", False)
                        
                        if success:
                            # Update match index và next index
                            self.match_index[node] = log_entry.index
                            self.next_index[node] = log_entry.index + 1
                        
                        return success
                    else:
                        return False
                        
        except Exception as e:
            logger.debug(f"Error replicating to {node}: {e}")
            return False
    
    async def _return_to_follower(self):
        """Trở lại follower state"""
        if self.state == NodeState.FOLLOWER:
            return
        
        self.state = NodeState.FOLLOWER
        self.leader_id = None
        self.metrics["state_changes"] += 1
        
        # Cancel heartbeat timer
        if self.heartbeat_timer and not self.heartbeat_timer.done():
            self.heartbeat_timer.cancel()
        
        # Restart election timer
        self._start_election_timer()
        
        logger.info(f"Node {self.node_id} returned to FOLLOWER state")
    
    def register_bid_commit_callback(self, auction_id: str, callback):
        """
        Đăng ký callback khi bid được commit
        
        Args:
            auction_id: Auction ID
            callback: Callback function (async or sync)
        """
        self.bid_commit_callbacks[auction_id] = callback
    
    def get_raft_status(self) -> Dict[str, Any]:
        """Get current Raft status"""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_length": len(self.log),
            "voted_for": self.voted_for,
            "quorum_size": self.quorum_size,
            "election_timeout": self.election_timeout,
            "last_heartbeat": self.last_heartbeat,
            "metrics": self.metrics.copy()
        }
    
    def get_log_info(self) -> Dict[str, Any]:
        """Get log information"""
        return {
            "log_length": len(self.log),
            "last_log_index": len(self.log) - 1 if self.log else -1,
            "last_log_term": self.log[-1].term if self.log else 0,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied
        }

# Utility function
def create_raft_consensus(node_id: str, node_addresses: List[str]) -> RaftConsensus:
    """
    Factory function để tạo RaftConsensus instance
    """
    config = RaftConfig()
    return RaftConsensus(node_id, node_addresses, config)

if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    async def test_raft():
        nodes = ["node1:8000", "node2:8000", "node3:8000"]
        raft = create_raft_consensus("node1:8000", nodes)
        
        print("Raft status:", raft.get_raft_status())
        
        # Simulate becoming leader
        await raft._become_leader()
        print("After becoming leader:", raft.get_raft_status())
    
    asyncio.run(test_raft())