#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from core.rag.retriever import DocumentRetriever
from infrastructure.llm.client import LLMClient
from config.settings import Settings

class ComplianceLevel(Enum):
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"

@dataclass
class ComplianceRule:
    """Represents a compliance rule to be checked."""
    id: str
    name: str
    description: str
    category: str
    severity: str
    requirements: List[str]

@dataclass
class ComplianceViolation:
    """Represents a compliance violation found during analysis."""
    rule_id: str
    location: str  # Document section or line number
    description: str
    severity: str
    suggested_fixes: List[str]

@dataclass
class ComplianceResult:
    """Represents the result of a compliance analysis."""
    document_id: str
    timestamp: datetime
    compliance_level: ComplianceLevel
    violations: List[ComplianceViolation]
    summary: str
    metadata: Dict[str, str]

class ComplianceAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_client = LLMClient(settings)
        self.doc_retriever = DocumentRetriever(settings)
        self.rules: Dict[str, ComplianceRule] = self._load_compliance_rules()

    def _load_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Load compliance rules from configuration.
        
        This method would typically load rules from a database or configuration file.
        For now, we'll return a sample set of rules.
        """
        # TODO: Implement proper rule loading from configuration
        sample_rules = {
            "GDPR-001": ComplianceRule(
                id="GDPR-001",
                name="Personal Data Processing",
                description="Ensure personal data processing is lawful, fair, and transparent",
                category="Data Protection",
                severity="High",
                requirements=[
                    "Clear legal basis for processing",
                    "Transparent processing information",
                    "Fair processing practices"
                ]
            ),
            "SEC-001": ComplianceRule(
                id="SEC-001",
                name="Data Security Measures",
                description="Implement appropriate technical and organizational security measures",
                category="Security",
                severity="High",
                requirements=[
                    "Encryption at rest",
                    "Access controls",
                    "Regular security assessments"
                ]
            )
        }
        return sample_rules

    async def analyze_document(self, document_id: str) -> ComplianceResult:
        """Analyze a document for compliance against all rules."""
        # Retrieve document content
        document_chunks = await self.doc_retriever.get_document_chunks(document_id)
        if not document_chunks:
            raise ValueError(f"Document not found: {document_id}")

        violations: List[ComplianceViolation] = []
        checked_rules: Set[str] = set()

        # Analyze each chunk against relevant rules
        for chunk in document_chunks:
            chunk_violations = await self._analyze_chunk(chunk)
            violations.extend(chunk_violations)
            checked_rules.update(v.rule_id for v in chunk_violations)

        # Determine overall compliance level
        compliance_level = self._determine_compliance_level(violations)

        # Generate summary
        summary = await self._generate_summary(document_id, violations)

        return ComplianceResult(
            document_id=document_id,
            timestamp=datetime.utcnow(),
            compliance_level=compliance_level,
            violations=violations,
            summary=summary,
            metadata={
                "rules_checked": ",".join(sorted(checked_rules)),
                "total_violations": str(len(violations))
            }
        )

    async def _analyze_chunk(self, chunk: str) -> List[ComplianceViolation]:
        """Analyze a document chunk for compliance violations."""
        violations: List[ComplianceViolation] = []

        # Use LLM to analyze the chunk against each rule
        for rule in self.rules.values():
            prompt = self._create_analysis_prompt(chunk, rule)
            response = await self.llm_client.analyze(prompt)
            
            if violation := self._parse_llm_response(response, rule):
                violations.append(violation)

        return violations

    def _create_analysis_prompt(self, chunk: str, rule: ComplianceRule) -> str:
        """Create a prompt for the LLM to analyze compliance."""
        return f"""Analyze the following text for compliance with this rule:

Rule: {rule.name}
Description: {rule.description}
Requirements:
{chr(10).join(f'- {req}' for req in rule.requirements)}

Text to analyze:
{chunk}

Provide a structured analysis including:
1. Whether the text violates the rule
2. Specific location of any violations
3. Detailed description of the violation
4. Suggested fixes
"""

    def _parse_llm_response(self, response: str, rule: ComplianceRule) -> Optional[ComplianceViolation]:
        """Parse LLM response to extract violation information."""
        # TODO: Implement more sophisticated response parsing
        if "violation" in response.lower():
            return ComplianceViolation(
                rule_id=rule.id,
                location="Document section",  # TODO: Extract precise location
                description=response,
                severity=rule.severity,
                suggested_fixes=["TODO: Extract suggested fixes from LLM response"]
            )
        return None

    def _determine_compliance_level(self, violations: List[ComplianceViolation]) -> ComplianceLevel:
        """Determine overall compliance level based on violations."""
        if not violations:
            return ComplianceLevel.COMPLIANT

        high_severity = any(v.severity == "High" for v in violations)
        if high_severity:
            return ComplianceLevel.NON_COMPLIANT
        return ComplianceLevel.PARTIALLY_COMPLIANT

    async def _generate_summary(self, document_id: str, violations: List[ComplianceViolation]) -> str:
        """Generate a human-readable summary of compliance analysis."""
        if not violations:
            return "Document is fully compliant with all checked rules."

        summary_prompt = f"""Summarize the following compliance violations in a clear, actionable format:

Document ID: {document_id}
Violations:
{chr(10).join(f'- Rule {v.rule_id}: {v.description}' for v in violations)}
"""

        return await self.llm_client.summarize(summary_prompt)