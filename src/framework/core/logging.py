import json
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Logging levels for different types of events"""

    RESEARCH = "research"
    AGENT = "agent"
    TOOL = "tool"
    DEBUG = "debug"
    ERROR = "error"


@dataclass
class LogEntry:
    """Individual log entry with metadata"""

    timestamp: str
    level: LogLevel
    agent_id: str
    event_type: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class ReasoningChain:
    """Captures an agent's complete thought process"""

    agent_id: str
    task_id: str
    timestamp: str
    thinking_process: List[str]
    decision_rationale: str
    selected_action: str
    alternatives_considered: List[str]
    confidence_level: float
    expected_outcome: str


class AgentLogger:
    """Centralized logging system for multi-agent research"""

    def __init__(self, verbose: bool = True, log_file: str = None):
        self.verbose = verbose
        self.log_file = log_file
        self.logs: List[LogEntry] = []
        self.reasoning_chains: List[ReasoningChain] = []
        self.active_agents: Dict[str, str] = {}  # agent_id -> current_status

    def log(
        self,
        level: LogLevel,
        agent_id: str,
        event_type: str,
        message: str,
        data: Dict[str, Any] = None,
        reasoning: str = "",
        confidence: float = 0.0,
    ):
        """Add a log entry"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            agent_id=agent_id,
            event_type=event_type,
            message=message,
            data=data or {},
            reasoning=reasoning,
            confidence=confidence,
        )

        self.logs.append(entry)

        if self.verbose:
            self._print_log(entry)

        if self.log_file:
            self._write_to_file(entry)

    def log_reasoning(self, reasoning_chain: ReasoningChain):
        """Log an agent's reasoning process"""
        self.reasoning_chains.append(reasoning_chain)

        if self.verbose:
            self._print_reasoning(reasoning_chain)

    def _print_log(self, entry: LogEntry):
        """Print log entry to console with formatting"""
        timestamp = entry.timestamp.split("T")[1][:8]  # HH:MM:SS
        level_color = {
            LogLevel.RESEARCH: "\033[94m",  # Blue
            LogLevel.AGENT: "\033[92m",  # Green
            LogLevel.TOOL: "\033[93m",  # Yellow
            LogLevel.DEBUG: "\033[90m",  # Gray
        }
        reset_color = "\033[0m"

        print(
            f"{level_color.get(entry.level, '')}{timestamp} [{entry.level.value.upper()}] "
            f"{entry.agent_id}: {entry.message}{reset_color}"
        )

        if entry.reasoning:
            print(f"  💭 Reasoning: {entry.reasoning}")

        if entry.data and entry.level != LogLevel.DEBUG:
            print(f"  📊 Data: {json.dumps(entry.data, indent=2)}")

    def _print_reasoning(self, reasoning: ReasoningChain):
        """Print reasoning chain with visual formatting"""
        print(f"\n🧠 REASONING CHAIN - {reasoning.agent_id} [{reasoning.task_id}]")
        print("=" * 60)

        print("💭 Thinking Process:")
        for i, thought in enumerate(reasoning.thinking_process, 1):
            print(f"  {i}. {thought}")

        print(f"\n🎯 Decision: {reasoning.decision_rationale}")
        print(f"⚡ Action: {reasoning.selected_action}")
        print(f"🔍 Confidence: {reasoning.confidence_level:.1%}")
        print(f"🎲 Expected: {reasoning.expected_outcome}")

        if reasoning.alternatives_considered:
            print(f"🤔 Alternatives: {', '.join(reasoning.alternatives_considered)}")
        print("=" * 60 + "\n")

    def _write_to_file(self, entry: LogEntry):
        """Write log entry to file"""
        with open(self.log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": entry.timestamp,
                        "level": entry.level.value,
                        "agent_id": entry.agent_id,
                        "event_type": entry.event_type,
                        "message": entry.message,
                        "data": entry.data,
                        "reasoning": entry.reasoning,
                        "confidence": entry.confidence,
                    }
                )
                + "\n"
            )

    def get_logs_by_agent(self, agent_id: str) -> List[LogEntry]:
        """Get all logs for specific agent"""
        return [log for log in self.logs if log.agent_id == agent_id]

    def get_logs_by_level(self, level: LogLevel) -> List[LogEntry]:
        """Get all logs of specific level"""
        return [log for log in self.logs if log.level == level]

    def get_research_summary(self) -> Dict[str, Any]:
        """Generate summary of research session"""
        return {
            "total_logs": len(self.logs),
            "agents_involved": list(set(log.agent_id for log in self.logs)),
            "reasoning_chains": len(self.reasoning_chains),
            "log_levels": {
                level.value: len(self.get_logs_by_level(level)) for level in LogLevel
            },
            "timeline": [
                (log.timestamp, log.agent_id, log.event_type) for log in self.logs[-10:]
            ],
        }

    def export_session(self, filename: str = None) -> Dict[str, Any]:
        """Export complete logging session for analysis"""
        if filename is None:
            filename = (
                f"research_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        session_data = {
            "session_summary": self.get_research_summary(),
            "all_logs": [
                {
                    "timestamp": log.timestamp,
                    "level": log.level.value,
                    "agent_id": log.agent_id,
                    "event_type": log.event_type,
                    "message": log.message,
                    "data": log.data,
                    "reasoning": log.reasoning,
                    "confidence": log.confidence,
                }
                for log in self.logs
            ],
            "reasoning_chains": [
                {
                    "agent_id": chain.agent_id,
                    "task_id": chain.task_id,
                    "timestamp": chain.timestamp,
                    "thinking_process": chain.thinking_process,
                    "decision_rationale": chain.decision_rationale,
                    "selected_action": chain.selected_action,
                    "alternatives_considered": chain.alternatives_considered,
                    "confidence_level": chain.confidence_level,
                    "expected_outcome": chain.expected_outcome,
                }
                for chain in self.reasoning_chains
            ],
        }

        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2)

        return session_data
