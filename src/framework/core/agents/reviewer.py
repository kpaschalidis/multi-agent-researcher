from typing import Any, Dict, List
from ..base import ResearchAgent
from ..logging import LogLevel
from langchain_core.prompts import ChatPromptTemplate
import json5


# TODO: Needs to be improved
class ReviewerAgent(ResearchAgent):
    """Quality assurance and validation agent"""

    def _build_system_prompt(self) -> str:
        return """You are an Expert Research Quality Reviewer with advanced evaluation capabilities.

Your role:
1. Evaluate research quality using multiple criteria
2. Provide specific, actionable feedback
3. Track improvements through revision cycles
4. Generate nuanced quality scores (0.0-1.0)

Evaluation Criteria:
- Content Depth (0-20 points): Comprehensiveness and detail level
- Source Quality (0-20 points): Credibility and authority of sources  
- Accuracy (0-20 points): Factual correctness and verification
- Coverage (0-20 points): How well the research addresses the query
- Clarity (0-20 points): Organization, flow, and readability

Scoring Guidelines:
- 0.9-1.0: Exceptional quality, publication-ready
- 0.8-0.89: High quality, minor improvements needed
- 0.7-0.79: Good quality, moderate improvements needed
- 0.6-0.69: Acceptable quality, significant improvements needed
- 0.5-0.59: Poor quality, major revisions required
- Below 0.5: Unacceptable quality, fundamental issues

Be precise and constructive in your feedback."""

    async def review_research(
        self,
        research_data: List[Dict],
        guidelines: List[str],
        previous_score: float = None,
        revision_count: int = 0,
    ) -> Dict[str, Any]:
        """Review research quality and provide feedback"""

        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "detailed_review_start",
            f"Starting detailed review (revision #{revision_count})",
            data={
                "previous_score": previous_score,
                "guidelines_count": len(guidelines),
            },
        )

        analysis_result = await self._analyze_research_content(research_data)

        llm_assessment = await self._get_llm_quality_assessment(
            research_data, guidelines, revision_count
        )

        final_score = await self._calculate_final_score(
            analysis_result, llm_assessment, previous_score
        )

        needs_revision = self._determine_revision_need(
            final_score, revision_count, previous_score
        )

        specific_feedback = await self._generate_specific_feedback(
            research_data, analysis_result, llm_assessment, revision_count
        )

        result = {
            "overall_score": final_score,
            "needs_revision": needs_revision,
            "previous_score": previous_score,
            "score_improvement": (
                final_score - previous_score if previous_score else 0.0
            ),
            "revision_count": revision_count,
            "analysis_breakdown": analysis_result,
            "llm_assessment": llm_assessment,
            "specific_feedback": specific_feedback,
            "strengths": self._identify_strengths(research_data, analysis_result),
            "weaknesses": self._identify_weaknesses(research_data, analysis_result),
            "approval_status": self._determine_approval_status(
                final_score, revision_count
            ),
            "detailed_reasoning": self._generate_detailed_reasoning(
                final_score, analysis_result, revision_count
            ),
        }

        self.logger.log(
            LogLevel.RESEARCH,
            self.agent_id,
            "review_completed",
            f"Review completed. Score: {final_score:.3f} (improvement: {result['score_improvement']:+.3f})",
            data={
                "score": final_score,
                "needs_revision": needs_revision,
                "revision_count": revision_count,
                "feedback_count": len(specific_feedback),
            },
        )

        return result

    async def _analyze_research_content(
        self, research_data: List[Dict]
    ) -> Dict[str, float]:
        """Quantitative analysis of research content"""

        total_content_length = 0
        total_sources = 0
        unique_sources = set()
        url_sources = 0

        for item in research_data:
            findings = item.get("findings", "")
            sources = item.get("sources", [])

            total_content_length += len(findings)
            total_sources += len(sources)

            for source in sources:
                unique_sources.add(source)
                if "http" in source:
                    url_sources += 1

        # TODO: Should be calculated based on the content length and the number of sources
        # Currently content depth is calculated based on the number of characters in the content
        # and source quality is calculated based on the number of URLs in the content
        # and source diversity is calculated based on the number of unique sources in the content
        content_depth_score = min(20, (total_content_length / 2000) * 20)
        source_quality_score = min(20, (url_sources / 3) * 20)
        source_diversity_score = min(20, (len(unique_sources) / 5) * 20)

        # Basic accuracy check (keyword presence)
        accuracy_indicators = self._check_accuracy_indicators(research_data)
        accuracy_score = accuracy_indicators * 20

        # Coverage assessment (how well distributed the content is)
        coverage_score = min(
            20, len(research_data) * 7
        )  # Up to 3 research items = full score

        return {
            "content_depth": content_depth_score,
            "source_quality": source_quality_score,
            "source_diversity": source_diversity_score,
            "accuracy": accuracy_score,
            "coverage": coverage_score,
            "total_quantitative": (
                content_depth_score
                + source_quality_score
                + source_diversity_score
                + accuracy_score
                + coverage_score
            )
            / 100,
        }

    def _check_accuracy_indicators(self, research_data: List[Dict]) -> float:
        """Check for accuracy indicators in content"""

        all_content = " ".join(
            [item.get("findings", "") for item in research_data]
        ).lower()

        # Positive indicators
        positive_indicators = [
            "according to",
            "research shows",
            "study found",
            "data indicates",
            "published",
            "peer-reviewed",
            "2023",
            "2024",
            "recent",
            "latest",
        ]

        # Negative indicators
        negative_indicators = [
            "might be",
            "could be",
            "possibly",
            "unverified",
            "rumored",
            "allegedly",
            "supposedly",
            "claims without evidence",
        ]

        positive_count = sum(
            1 for indicator in positive_indicators if indicator in all_content
        )
        negative_count = sum(
            1 for indicator in negative_indicators if indicator in all_content
        )

        # Score between 0 and 1
        accuracy_ratio = max(
            0,
            min(
                1, (positive_count - negative_count) / max(1, len(positive_indicators))
            ),
        )
        return accuracy_ratio

    async def _get_llm_quality_assessment(
        self, research_data: List[Dict], guidelines: List[str], revision_count: int
    ) -> Dict[str, Any]:
        """Get LLM-based qualitative assessment"""

        combined_content = "\n\n".join(
            [
                f"Research Item {i+1} ({item.get('task_id', 'unknown')}):\n{item.get('findings', '')}"
                for i, item in enumerate(research_data)
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Evaluate this research content against the quality guidelines and provide a detailed assessment.

Guidelines:
{guidelines}

Research Content:
{content}

Revision Number: {revision_count}

Provide your assessment as JSON:
{{
    "clarity_score": 0.85,
    "depth_score": 0.78,
    "relevance_score": 0.90,
    "organization_score": 0.82,
    "improvement_areas": ["specific area 1", "specific area 2"],
    "strengths": ["strength 1", "strength 2"],
    "overall_impression": "detailed assessment text",
    "recommended_score": 0.84
}}

Be precise and provide scores between 0.0 and 1.0.""",
                ),
            ]
        )

        messages = prompt.format_messages(
            guidelines="\n".join([f"- {g}" for g in guidelines]),
            content=combined_content[:6000],  # Limit for token constraints
            revision_count=revision_count,
        )

        try:
            response = await self.llm.ainvoke(messages)

            return json5.loads(response.content.strip())
        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                self.agent_id,
                "llm_assessment_error",
                f"LLM assessment failed: {str(e)}",
            )
            # Fallback assessment
            return {
                "clarity_score": 0.7,
                "depth_score": 0.7,
                "relevance_score": 0.7,
                "organization_score": 0.7,
                "improvement_areas": ["LLM assessment failed"],
                "strengths": ["Content present"],
                "overall_impression": "Assessment unavailable due to technical issue",
                "recommended_score": 0.7,
            }

    async def _calculate_final_score(
        self,
        analysis_result: Dict[str, float],
        llm_assessment: Dict[str, Any],
        previous_score: float = None,
    ) -> float:
        """Calculate final quality score combining multiple assessments"""

        quantitative_score = analysis_result["total_quantitative"]

        llm_scores = [
            llm_assessment.get("clarity_score", 0.7),
            llm_assessment.get("depth_score", 0.7),
            llm_assessment.get("relevance_score", 0.7),
            llm_assessment.get("organization_score", 0.7),
        ]
        qualitative_score = sum(llm_scores) / len(llm_scores)

        base_score = (quantitative_score * 0.4) + (qualitative_score * 0.6)

        if previous_score is not None:
            # Small bonus for any improvement during revision
            improvement = max(0, base_score - previous_score)
            improvement_bonus = min(0.05, improvement * 0.5)  # Max 5% bonus
            final_score = base_score + improvement_bonus
        else:
            final_score = base_score

        return max(0.0, min(1.0, final_score))

    def _determine_revision_need(
        self, score: float, revision_count: int, previous_score: float = None
    ) -> bool:
        """Determine if revision is needed based on score and improvement"""

        if score < 0.8:
            return True

        if previous_score is not None and revision_count > 0:
            improvement = score - previous_score
            if improvement < 0.02 and revision_count >= 1:
                return False

        return False

    async def _generate_specific_feedback(
        self,
        research_data: List[Dict],
        analysis_result: Dict[str, float],
        llm_assessment: Dict[str, Any],
        revision_count: int,
    ) -> List[Dict[str, str]]:
        """Generate specific, actionable feedback"""

        feedback = []

        if analysis_result["content_depth"] < 15:
            feedback.append(
                {
                    "category": "content_depth",
                    "issue": "Insufficient content depth",
                    "suggestion": f"Expand research findings. Current depth score: {analysis_result['content_depth']:.1f}/20. Need more detailed analysis.",
                    "priority": "high",
                }
            )

        if analysis_result["source_quality"] < 15:
            feedback.append(
                {
                    "category": "source_quality",
                    "issue": "Limited high-quality sources",
                    "suggestion": f"Include more authoritative sources. Current score: {analysis_result['source_quality']:.1f}/20. Target academic papers, official reports.",
                    "priority": "high",
                }
            )

        for area in llm_assessment.get("improvement_areas", []):
            feedback.append(
                {
                    "category": "qualitative",
                    "issue": f"Improvement needed: {area}",
                    "suggestion": f"Address this area based on expert assessment (revision #{revision_count})",
                    "priority": "medium",
                }
            )

        if revision_count > 0:
            feedback.append(
                {
                    "category": "revision",
                    "issue": "Revision cycle feedback",
                    "suggestion": f"This is revision #{revision_count}. Focus on addressing previous feedback systematically.",
                    "priority": "medium",
                }
            )

        return feedback

    def _determine_approval_status(self, score: float, revision_count: int) -> str:
        """Determine approval status based on score and revision history"""

        if score >= 0.9:
            return "approved_excellent"
        elif score >= 0.8:
            return "approved_good"
        elif score >= 0.7 and revision_count >= 2:
            return "approved_conditional"
        elif score >= 0.7:
            return "needs_revision"
        elif score >= 0.6:
            return "needs_major_revision"
        else:
            return "rejected"

    def _identify_strengths(
        self, research_data: List[Dict], analysis_result: Dict[str, float]
    ) -> List[str]:
        """Identify specific strengths in the research"""

        strengths = []

        # Content-based strengths
        total_content_length = sum(
            len(item.get("findings", "")) for item in research_data
        )
        if total_content_length > 3000:
            strengths.append("Comprehensive content depth with detailed findings")

        if analysis_result["content_depth"] >= 18:
            strengths.append("Excellent content depth and thoroughness")
        elif analysis_result["content_depth"] >= 15:
            strengths.append("Good content depth and coverage")

        # Source-based strengths
        total_sources = sum(len(item.get("sources", [])) for item in research_data)
        if total_sources >= 6:
            strengths.append("Strong source diversity with multiple references")
        elif total_sources >= 4:
            strengths.append("Good source coverage")

        if analysis_result["source_quality"] >= 18:
            strengths.append("High-quality, authoritative sources")
        elif analysis_result["source_quality"] >= 15:
            strengths.append("Credible source selection")

        # Coverage and accuracy strengths
        if analysis_result["coverage"] >= 18:
            strengths.append("Comprehensive topic coverage")

        if analysis_result["accuracy"] >= 18:
            strengths.append("Strong accuracy indicators and factual content")

        # Multi-perspective analysis
        if len(research_data) > 1:
            strengths.append("Multi-perspective research approach")

        # Content quality indicators
        all_content = " ".join(
            [item.get("findings", "") for item in research_data]
        ).lower()

        # Check for structured content
        if any(
            indicator in all_content
            for indicator in ["conclusion", "summary", "key findings", "implications"]
        ):
            strengths.append("Well-structured analysis with clear conclusions")

        # Check for current/recent information
        if any(
            indicator in all_content
            for indicator in ["2023", "2024", "2025", "recent", "latest", "current"]
        ):
            strengths.append("Up-to-date information and recent developments")

        # Check for technical depth
        if any(
            indicator in all_content
            for indicator in [
                "framework",
                "architecture",
                "implementation",
                "methodology",
            ]
        ):
            strengths.append("Technical depth and detailed analysis")

        # Fallback if no specific strengths identified
        if not strengths:
            strengths.append(
                "Research demonstrates systematic approach to topic investigation"
            )

        return strengths

    def _identify_weaknesses(
        self, research_data: List[Dict], analysis_result: Dict[str, float]
    ) -> List[str]:
        """Identify specific weaknesses in the research"""

        weaknesses = []

        # Content depth issues
        if analysis_result["content_depth"] < 12:
            weaknesses.append(
                "Insufficient content depth - research findings are too brief"
            )
        elif analysis_result["content_depth"] < 15:
            weaknesses.append(
                "Content depth could be improved with more detailed analysis"
            )

        # Source quality issues
        if analysis_result["source_quality"] < 10:
            weaknesses.append(
                "Limited high-quality sources - need more authoritative references"
            )
        elif analysis_result["source_quality"] < 15:
            weaknesses.append(
                "Source quality could be enhanced with more credible references"
            )

        # Source diversity issues
        if analysis_result["source_diversity"] < 10:
            weaknesses.append(
                "Poor source diversity - research relies on too few unique sources"
            )
        elif analysis_result["source_diversity"] < 15:
            weaknesses.append(
                "Source diversity could be improved with more varied references"
            )

        # Coverage issues
        if analysis_result["coverage"] < 12:
            weaknesses.append("Incomplete topic coverage - key aspects may be missing")
        elif analysis_result["coverage"] < 15:
            weaknesses.append("Topic coverage could be more comprehensive")

        # Accuracy concerns
        if analysis_result["accuracy"] < 12:
            weaknesses.append(
                "Accuracy concerns - limited factual indicators or verification"
            )
        elif analysis_result["accuracy"] < 15:
            weaknesses.append("More factual evidence and verification needed")

        # Content quality issues
        all_content = " ".join(
            [item.get("findings", "") for item in research_data]
        ).lower()

        # Check for vague language
        vague_indicators = [
            "might be",
            "could be",
            "possibly",
            "perhaps",
            "may be",
            "seems to",
        ]
        if sum(1 for indicator in vague_indicators if indicator in all_content) > 3:
            weaknesses.append(
                "Excessive use of uncertain language - more definitive statements needed"
            )

        # Check for lack of specific examples
        if not any(
            indicator in all_content
            for indicator in ["example", "instance", "case study", "demonstration"]
        ):
            weaknesses.append(
                "Lacks specific examples or case studies to illustrate points"
            )

        # Check for missing current information
        if not any(
            indicator in all_content
            for indicator in ["2023", "2024", "2025", "recent", "latest"]
        ):
            weaknesses.append(
                "Limited recent information - research may not reflect latest developments"
            )

        # Check for structural issues
        total_length = len(all_content)
        if total_length < 1000:
            weaknesses.append(
                "Overall content volume is insufficient for comprehensive analysis"
            )

        # Check for repetition or redundancy
        unique_sources = set()
        duplicate_sources = []
        for item in research_data:
            for source in item.get("sources", []):
                if source in unique_sources:
                    duplicate_sources.append(source)
                unique_sources.add(source)

        if len(duplicate_sources) > 2:
            weaknesses.append(
                "Contains duplicate sources - efficiency could be improved"
            )

        # Check for missing critical analysis
        if not any(
            indicator in all_content
            for indicator in ["analysis", "evaluation", "assessment", "critique"]
        ):
            weaknesses.append("Lacks critical analysis - needs more analytical depth")

        # Check for missing context or background
        if not any(
            indicator in all_content
            for indicator in ["background", "context", "history", "evolution"]
        ):
            weaknesses.append("Missing contextual background information")

        return weaknesses

    def _generate_detailed_reasoning(
        self, score: float, analysis_result: Dict[str, float], revision_count: int
    ) -> str:
        """Generate detailed reasoning for the score"""

        reasoning_parts = [
            f"Final Quality Score: {score:.3f}",
            f"Revision Cycle: {revision_count}",
            "",
            "Score Breakdown:",
            f"• Content Depth: {analysis_result['content_depth']:.1f}/20",
            f"• Source Quality: {analysis_result['source_quality']:.1f}/20",
            f"• Source Diversity: {analysis_result['source_diversity']:.1f}/20",
            f"• Accuracy Indicators: {analysis_result['accuracy']:.1f}/20",
            f"• Coverage: {analysis_result['coverage']:.1f}/20",
            "",
            f"Quantitative Assessment: {analysis_result['total_quantitative']:.3f}",
        ]

        if score >= 0.8:
            reasoning_parts.append("✅ Research meets quality standards")
        elif score >= 0.7:
            reasoning_parts.append("⚠️ Research needs improvement but shows promise")
        else:
            reasoning_parts.append("❌ Research requires significant enhancement")

        return "\n".join(reasoning_parts)
