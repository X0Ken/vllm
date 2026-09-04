# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import RepetitionDetectionParams

DEFAULT_CONFIG = RepetitionDetectionParams(
    min_pattern_size=2,
    max_pattern_size=64,
    min_count=4,
)
REQUEST_CONFIG = RepetitionDetectionParams(
    min_pattern_size=8,
    max_pattern_size=128,
    min_count=5,
)
DEFAULT_SAMPLING_PARAMS = {"repetition_detection": DEFAULT_CONFIG}


def test_chat_repetition_detection_default_and_request_override():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    params = request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == DEFAULT_CONFIG

    request.repetition_detection = REQUEST_CONFIG
    params = request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == REQUEST_CONFIG


def test_completion_repetition_detection_default_and_request_override():
    request = CompletionRequest(prompt="hi", max_tokens=32)
    params = request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == DEFAULT_CONFIG

    request.repetition_detection = REQUEST_CONFIG
    params = request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == REQUEST_CONFIG


def test_responses_repetition_detection_default_and_request_override():
    request = ResponsesRequest(input="hi")
    params = request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == DEFAULT_CONFIG

    request.repetition_detection = REQUEST_CONFIG
    params = request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == REQUEST_CONFIG


def test_anthropic_repetition_detection_is_forwarded_to_chat():
    request = AnthropicMessagesRequest(
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        repetition_detection=REQUEST_CONFIG,
    )
    chat_request = AnthropicServingMessages._build_base_request(
        request,
        [{"role": "user", "content": "hi"}],
    )
    params = chat_request.to_sampling_params(32, DEFAULT_SAMPLING_PARAMS)
    assert params.repetition_detection == REQUEST_CONFIG
