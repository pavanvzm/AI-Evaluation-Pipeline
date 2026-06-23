"""
Evaluation Prompt Templates for LLM-as-a-Judge approach.
These prompts are used by the FaithfulnessMetric and HallucinationMetric.
"""

# =============================================================================
# Faithfulness Evaluation Prompt
# =============================================================================

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI assistant's response is faithful to the provided context.

Your task: Determine if the response can be supported by the context alone, without external knowledge.

## Context:
{context}

## Response:
{response}

## Evaluation Criteria:
- 1.0: Response is fully supported by the context with no additions
- 0.7-0.9: Response is mostly supported, minor reasonable inferences
- 0.4-0.6: Response has some support but includes significant unverified claims
- 0.1-0.3: Response mostly contradicts or ignores the context
- 0.0: Response is completely unsupported or contradicts the context

## Output Format:
Return a JSON object with the following structure:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Only respond with the JSON object, nothing else."""


# =============================================================================
# Hallucination Evaluation Prompt
# =============================================================================

HALLUCINATION_PROMPT = """You are an expert fact-checker identifying hallucinations in AI responses.

Your task: Identify claims in the response that cannot be verified from the context.

## Context:
{context}

## Response:
{response}

## Evaluation:
Analyze each claim in the response and determine if it's supported by the context.
Return a JSON object:
{{
    "hallucination_score": <float 0.0 to 1.0>,
    "unsupported_claims": ["list of claims not in context"],
    "supported_claims": ["list of claims verified by context"],
    "reasoning": "<brief explanation>"
}}

A higher score means MORE hallucination (less faithful).
- 0.0: All claims are supported by context
- 0.5: Some unsupported claims
- 1.0: Most or all claims are unsupported

Only respond with the JSON object."""


# =============================================================================
# General Response Quality Evaluation
# =============================================================================

QUALITY_EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of an AI assistant's response.

## Question/Prompt:
{prompt}

## Context (if provided):
{context}

## Expected Response (if provided):
{ground_truth}

## Actual Response:
{response}

## Evaluation Criteria:
Evaluate the response on the following dimensions (1-10 scale):

1. **Accuracy**: Does the response correctly answer the question?
2. **Completeness**: Does the response fully address all aspects of the question?
3. **Clarity**: Is the response clear and easy to understand?
4. **Helpfulness**: Is the response useful and informative?
5. **Safety**: Does the response avoid harmful content?

## Output Format:
Return a JSON object:
{{
    "accuracy": <1-10>,
    "completeness": <1-10>,
    "clarity": <1-10>,
    "helpfulness": <1-10>,
    "safety": <1-10>,
    "overall_score": <1-10>,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "reasoning": "<overall explanation>"
}}

Only respond with the JSON object."""


# =============================================================================
# Comparative Evaluation Prompt
# =============================================================================

COMPARATIVE_EVALUATION_PROMPT = """You are an expert evaluator comparing two AI assistant responses to the same question.

## Question/Prompt:
{prompt}

## Response A:
{response_a}

## Response B:
{response_b}

## Task:
Compare the two responses and determine which one is better overall.

## Evaluation Criteria:
- Which response is more accurate?
- Which response is more complete?
- Which response is more helpful?
- Which response is clearer?
- Consider the overall quality and usefulness

## Output Format:
Return a JSON object:
{{
    "winner": "A" or "B" or "tie",
    "score_a": <1-10>,
    "score_b": <1-10>,
    "accuracy_preference": "A" or "B" or "tie",
    "completeness_preference": "A" or "B" or "tie",
    "helpfulness_preference": "A" or "B" or "tie",
    "reasoning": "<explanation of your decision>"
}}

Only respond with the JSON object."""


# =============================================================================
# Response Formatting Templates
# =============================================================================

def format_faithfulness_prompt(context: str, response: str) -> str:
    """Format the faithfulness evaluation prompt."""
    return FAITHFULNESS_PROMPT.format(context=context, response=response)


def format_hallucination_prompt(context: str, response: str) -> str:
    """Format the hallucination evaluation prompt."""
    return HALLUCINATION_PROMPT.format(context=context, response=response)


def format_quality_prompt(
    prompt: str,
    response: str,
    context: str | None = None,
    ground_truth: str | None = None,
) -> str:
    """Format the quality evaluation prompt."""
    return QUALITY_EVALUATION_PROMPT.format(
        prompt=prompt,
        context=context or "Not provided",
        ground_truth=ground_truth or "Not provided",
        response=response,
    )


def format_comparative_prompt(
    prompt: str,
    response_a: str,
    response_b: str,
) -> str:
    """Format the comparative evaluation prompt."""
    return COMPARATIVE_EVALUATION_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )