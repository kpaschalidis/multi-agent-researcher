import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.framework import create_general_research_workflow


load_dotenv()

# Example query
QUERY = "What are the latest developments in AI agent frameworks?"


async def demo_logging_system():
    print("🔍 Multi-Agent Research Framework")
    print("=" * 60)

    # Setup
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("⚠️  Warning: TAVILY_API_KEY not found. Some features may not work.")
        return

    research_system = create_general_research_workflow(
        llm=llm,
        tavily_api_key=tavily_api_key,
        verbose_logging=True,
        log_file=".log",
    )

    print(f"\n🎯 Research Query: '{QUERY}'")
    print("=" * 60)
    print("\n🤖 Watch the agents think and work:")
    print("-" * 40)

    try:
        result = await research_system.research(QUERY, publication_formats=["markdown"])

        # Results
        print("\n" + "=" * 60)
        print("📋 RESEARCH RESULTS")
        print("=" * 60)

        print(f"\n📊 Research Plan:")
        print(
            result["research_plan"][:200] + "..."
            if len(result["research_plan"]) > 200
            else result["research_plan"]
        )

        print(f"\n📝 Final Report (excerpt):")
        report_excerpt = (
            result["final_report"][:300] + "..."
            if len(result["final_report"]) > 300
            else result["final_report"]
        )

        print(report_excerpt)

        print(f"\n📈 Research Metrics:")
        print(f"- Agents spawned: {result['num_subagents']}")
        print(f"- Reasoning chains: {result['reasoning_chains']}")
        print(f"- Log entries: {result['log_entries']}")
        print(f"- Citations: {len(result['citations'])}")
        print(f"- Errors: {len(result['errors'])}")

        # Agent activity summary
        summary = result["research_summary"]
        print(f"\n🎭 Agent Activity:")
        print(f"- Agents involved: {', '.join(summary['agents_involved'])}")
        print(f"- Research logs: {summary['log_levels'].get('research', 0)}")
        print(f"- Agent decision logs: {summary['log_levels'].get('agent', 0)}")
        print(f"- Tool usage logs: {summary['log_levels'].get('tool', 0)}")

        print(f"\n💾 Detailed logs saved to: demo_research.log")

        # Reasoning examples
        if research_system.logger.reasoning_chains:
            print(f"\n🧠 Sample Reasoning Chain:")
            chain = research_system.logger.reasoning_chains[0]
            print(f"Agent: {chain.agent_id}")
            print(f"Decision: {chain.decision_rationale}")
            print(f"Confidence: {chain.confidence_level:.1%}")

        # Write to file
        md_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
        with open(md_filename, "w") as md_file:
            md_file.write(result["final_report"])

    except Exception as e:
        print(f"❌ Research failed: {str(e)}")

        # Even on failure, show what logs we captured
        if hasattr(research_system, "logger") and research_system.logger.logs:
            print(
                f"\n📋 Log entries before failure: {len(research_system.logger.logs)}"
            )
            print("Check the log file for detailed error information.")


def analyze_logs():
    try:
        with open(".log", "r") as f:
            lines = f.readlines()
            print(f"\n📋 Log Analysis: {len(lines)} entries saved")

            levels = {"research": 0, "agent": 0, "tool": 0, "debug": 0}
            for line in lines:
                try:
                    import json

                    log_entry = json.loads(line.strip())
                    level = log_entry.get("level", "unknown")
                    if level in levels:
                        levels[level] += 1
                except:
                    continue

            print("Log distribution:", levels)
    except FileNotFoundError:
        print("No log file found")


if __name__ == "__main__":
    print("Starting basic usage demo...")
    asyncio.run(demo_logging_system())
    analyze_logs()
