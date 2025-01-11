from typing import Dict
import logging

from opik.evaluation.metrics import (
    Contains,
    Equals,
    LevenshteinRatio,
    Hallucination,
    Moderation,
    AnswerRelevance,
    GEval
)
from opik.evaluation.metrics.score_result import ScoreResult

from src.components.evaluation.custom_metric import AnswerCompleteness

logger = logging.getLogger(__name__)

TASK_INTRODUCTION = (
    "You are an expert judge tasked with evaluating the faithfulness of an AI-generated answer to the given context."
)

EVALUATION_CRITERIA = """
- The OUTPUT must not introduce new information beyond what's provided in the CONTEXT.
- The OUTPUT must not contradict any information given in the CONTEXT.
- The OUTPUT should be logically consistent and coherent.
- The OUTPUT should comprehensively address all aspects of the user's query.
"""

class LlmEvaluator:
    def __init__(self):
        """
        Hard-code a set of references and context for your 0704.0001 paper.
        Now we rely on the default opik classes directly.
        """
        self.metrics = {
            "contains_diphoton": Contains(name="contains_diphoton", case_sensitive=False),
            "contains_berger":   Contains(name="contains_berger",   case_sensitive=False),
            "equals_title":      Equals(name="equals_title"),
            "lev_ratio_abstract": LevenshteinRatio(name="lev_ratio_abstract"),
        }

        self.hallucination_metric = Hallucination()
        self.moderation_metric = Moderation()
        self.answer_relevance_metric = AnswerRelevance()
        self.g_eval_metric = GEval(task_introduction=TASK_INTRODUCTION, evaluation_criteria=EVALUATION_CRITERIA)

        # Custom metric
        self.answer_completeness_metric = AnswerCompleteness()

        self.abstract_0704_0001 = (
            "A fully differential calculation in perturbative quantum chromodynamics is\n"
            "presented for the production of massive photon pairs at hadron colliders. All\n"
            "next-to-leading order perturbative contributions from quark-antiquark,\n"
            "gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\n"
            "all-orders resummation of initial-state gluon radiation valid at\n"
            "next-to-next-to-leading logarithmic accuracy. The region of phase space is\n"
            "specified in which the calculation is most reliable. Good agreement is\n"
            "demonstrated with data from the Fermilab Tevatron, and predictions are made for\n"
            "more detailed tests with CDF and DO data. Predictions are shown for\n"
            "distributions of diphoton pairs produced at the energy of the Large Hadron\n"
            "Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\n"
            "boson are contrasted with those produced from QCD processes at the LHC, showing\n"
            "that enhanced sensitivity to the signal can be obtained with judicious\n"
            "selection of events.\n"
        )
        self.context_0704_0001 = [
            "Title: Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies",
            f"Abstract: {self.abstract_0704_0001}",
            "Authors: Balázs C., Berger E. L., Nadolsky P. M., Yuan C. -P.",
            "Submitted on: Mon, 2 Apr 2007 19:18:42 GMT",
            "Updated on: 2008-11-26"
        ]

        self.static_references = {
            "contains_diphoton": "diphoton",
            "contains_berger":   "Berger",
            "equals_title": "Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies",
            "lev_ratio_abstract": self.abstract_0704_0001,
        }

        self.answer_context = (
            "Transformer models are neural network architectures used in NLP "
            "that employ self-attention mechanisms to process entire sentences "
            "simultaneously, capturing long-range dependencies and context. They "
            "use positional encoding to differentiate between words in different "
            "positions within a sentence."
        )


    def evaluate(self, output: str) -> Dict[str, ScoreResult]:
        """
        Evaluate an LLM output with your *static* references for 0704.0001.
        Return a dict of {metric_name -> ScoreResult} objects.
        """
        results = {}
        for metric_name, metric_obj in self.metrics.items():
            # e.g. "contains_diphoton" => reference "diphoton"
            ref = self.static_references[metric_name]
            score_res = metric_obj.score(output=output, reference=ref)
            results[metric_name] = score_res
        return results

    def check_hallucination(self, input_text: str, output_text: str) -> ScoreResult:
        """
        0 = no hallucination, 1 = hallucination found.
        """
        return self.hallucination_metric.score(
            input=input_text,
            output=output_text,
            context=self.context_0704_0001
        )

    def check_moderation(self, output_text: str) -> ScoreResult:
        """
        0.0 => safe, up to 1.0 => extremely unsafe
        """
        return self.moderation_metric.score(output=output_text)

    def check_answer_relevance(
        self,
        input_text: str,
        output_text: str,
    ) -> float:
        """
        Return a float in [0..1], measuring how relevant `output_text` is
        to `input_text` given `context_snippet`.
        """
        score_result = self.answer_relevance_metric.score(
            input=input_text,
            output=output_text,
            context=[self.answer_context]
        )
        return score_result.value

    def check_g_eval(
        self, output_text: str
    ) -> float:
        """
        Evaluate the LLM's output using GEval metric.
        """
        score_result = self.g_eval_metric.score(
            output=output_text
        )
        return score_result.value

    def check_answer_completeness(
            self, input_text: str, output_text: str
    ) -> float:
        """
        Check how complete the LLM's answer is relative to the user's input.

        Args:
            input_text: The user's question or prompt.
            output_text: The LLM-generated answer.

        Returns:
            float: Completeness score between 0.0 and 1.0.
        """
        score_result = self.answer_completeness_metric.score(
            input=input_text,
            output=output_text,
            context=[self.answer_context]
        )
        return score_result.value

