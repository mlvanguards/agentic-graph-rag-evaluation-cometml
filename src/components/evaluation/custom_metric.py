import json
from typing import Any, List, Optional, Union
import pydantic
from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation.models import litellm_chat_model
from opik.evaluation.models import base_model
import logging
LOGGER = logging.getLogger(__name__)


class AnswerCompletenessResponseFormat(pydantic.BaseModel):
    answer_completeness_score: float
    reason: str


class AnswerCompleteness(base_metric.BaseMetric):
    """
    A metric that evaluates the completeness of an answer relative to the user's input.

    This metric uses an LLM to assess whether the generated answer fully addresses all aspects of the user's query.

    Args:
        model: The language model to use for evaluation. Defaults to "gpt-4o".
        name: The name of the metric. Defaults to "answer_completeness_metric".
        few_shot_examples: Optional list of few-shot examples to guide the LLM's evaluation.
        track: Whether to track the metric. Defaults to True.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "answer_completeness_metric",
        few_shot_examples: Optional[List[Any]] = None,  # Replace `Any` with actual type if defined
        track: bool = True,
    ):
        super().__init__(
            name=name,
            track=track,
        )
        self._init_model(model)
        if few_shot_examples is None:
            self._few_shot_examples = []  # Define default few-shot examples if available
        else:
            self._few_shot_examples = few_shot_examples

    def _init_model(
        self, model: Optional[Union[str, base_model.OpikBaseModel]]
    ) -> None:
        if isinstance(model, base_model.OpikBaseModel):
            self._model = model
        else:
            self._model = litellm_chat_model.LiteLLMChatModel(model_name=model or "gpt-4o")

    def score(
        self, input: str, output: str, context: List[str], **ignored_kwargs: Any
    ) -> score_result.ScoreResult:
        """
        Calculate the answer completeness score for the given input-output pair.

        Args:
            input: The user's question or prompt.
            output: The LLM-generated answer.
            context: A list of context strings relevant to the input.

        Returns:
            score_result.ScoreResult: Contains the completeness score and reason.
        """
        llm_query = self._generate_prompt(input, output, context)
        model_output = self._model.generate_string(
            input=llm_query, response_format=AnswerCompletenessResponseFormat
        )
        return self._parse_model_output(model_output)

    async def ascore(
        self, input: str, output: str, context: List[str], **ignored_kwargs: Any
    ) -> score_result.ScoreResult:
        """
        Asynchronously calculate the answer completeness score for the given input-output pair.

        Args:
            input: The user's question or prompt.
            output: The LLM-generated answer.
            context: A list of context strings relevant to the input.

        Returns:
            score_result.ScoreResult: Contains the completeness score and reason.
        """
        llm_query = self._generate_prompt(input, output, context)
        model_output = await self._model.agenerate_string(
            input=llm_query, response_format=AnswerCompletenessResponseFormat
        )
        return self._parse_model_output(model_output)

    def _generate_prompt(self, input_text: str, output_text: str, context: List[str]) -> str:
        """
        Generate the prompt to send to the LLM for evaluation.

        Args:
            input_text: The user's question or prompt.
            output_text: The LLM-generated answer.
            context: Relevant context information.

        Returns:
            str: The complete prompt for the LLM.
        """
        context_combined = " ".join(context)
        prompt = f"""
        YOU ARE AN EXPERT IN NLP EVALUATION METRICS, TRAINED TO ASSESS THE COMPLETENESS OF ANSWERS PROVIDED BY LANGUAGE MODELS.

        ###INSTRUCTIONS###
        - ANALYZE THE GIVEN USER INPUT AND THE GENERATED ANSWER.
        - DETERMINE IF THE ANSWER FULLY ADDRESSES ALL ASPECTS OF THE USER'S QUERY.
        - ASSIGN A COMPLETENESS SCORE BETWEEN 0.0 (COMPLETELY INCOMPLETE) AND 1.0 (FULLY COMPLETE).
        - PROVIDE A BRIEF REASON FOR THE SCORE, HIGHLIGHTING WHICH PARTS WERE ADDRESSED OR MISSING.

        ###EXAMPLE###
        Input: "Explain the concept of transformer models in NLP and provide examples."
        Answer: "Transformer models are neural network architectures used in NLP that employ self-attention mechanisms..."
        Context: "Transformer models are neural network architectures used in NLP that employ self-attention mechanisms to process entire sentences simultaneously, capturing long-range dependencies and context."
        ---
        {
            "answer_completeness_score": 0.9,
            "reason": "The answer thoroughly explains transformer models and provides examples, fully addressing the user's request."
        }

        ###INPUTS:###
        ***
        User input:
        {input_text}
        Answer:
        {output_text}
        Contexts:
        {context_combined}
        ***
        """
        return prompt

    def _parse_model_output(self, content: str) -> score_result.ScoreResult:
        """
        Parse the LLM's JSON response into a ScoreResult object.

        Args:
            content: The JSON string returned by the LLM.

        Returns:
            score_result.ScoreResult: The parsed score and reason.
        """
        dict_content = json.loads(content)
        score: float = dict_content.get("answer_completeness_score", 0.5)
        reason: str = dict_content.get("reason", "No reason provided.")

        # Validate score range
        if not (0.0 <= score <= 1.0):
            LOGGER.warning("Received score out of bounds. Defaulting to 0.5.")
            score = 0.5

        return score_result.ScoreResult(
            name=self.name, value=score, reason=reason
        )

