# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
    ]


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer, sample_tools):
    return Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


class TestGlm47ExtractToolCalls:
    def test_no_tool_call(self, glm47_tool_parser, mock_request):
        out = "This is a plain response."
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_args_with_newlines(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_whitespace_preserved_in_arg_values(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "  Beijing  "}

    def test_content_before(self, glm47_tool_parser, mock_request):
        out = "Checking.<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, glm47_tool_parser, mock_request):
        out = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        )
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    def test_empty_content_none(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
        out = "  \n  <tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None


def _reset(parser):
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


class TestGlm47Streaming:
    def test_no_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = ["<tool_call>", "get_current_date", "</tool_call>"]
        current_text = ""
        for chunk in chunks:
            current_text += chunk
            glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
        assert len(glm47_tool_parser.prev_tool_call_arr) >= 1

    def test_with_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
        current_text = ""
        for chunk in chunks:
            current_text += chunk
            glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
        args = json.loads(glm47_tool_parser.prev_tool_call_arr[0]["arguments"])
        assert args["city"] == "Beijing"


class _DummyTokenizer:
    def get_vocab(self):
        return {
            "<think>": 1,
            "</think>": 2,
            "<tool_call>": 3,
            "</tool_call>": 4,
        }


class _Glm47DeepSeekParser(_WrappedParser):
    reasoning_parser_cls = DeepSeekR1ReasoningParser
    tool_parser_cls = Glm47MoeModelToolParser


@pytest.fixture
def write_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="write",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            ),
        ),
    ]


@pytest.fixture
def write_request(write_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = write_tools
    request.tool_choice = "auto"
    return request


def test_thinking_disabled_non_streaming_routes_tool_markup_to_glm47_parser(
    write_tools, write_request
):
    parser = _Glm47DeepSeekParser(_DummyTokenizer(), tools=write_tools)
    raw = (
        "<tool_call>write"
        "<arg_key>file_path</arg_key><arg_value>/tmp/x.txt</arg_value>"
        "<arg_key>content</arg_key><arg_value>hello</arg_value>"
        "</tool_call>"
    )

    reasoning, content, tool_calls = parser.parse(
        raw, write_request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "write"
    assert json.loads(tool_calls[0].arguments) == {
        "file_path": "/tmp/x.txt",
        "content": "hello",
    }


def test_thinking_disabled_streaming_emits_tool_call_delta_not_reasoning(
    write_tools, write_request
):
    parser = _Glm47DeepSeekParser(_DummyTokenizer(), tools=write_tools)
    raw = (
        "<tool_call>write"
        "<arg_key>file_path</arg_key><arg_value>/tmp/x.txt</arg_value>"
        "</tool_call>"
    )

    delta = parser.parse_delta(
        delta_text=raw,
        delta_token_ids=[],
        request=write_request,
        prompt_token_ids=[],
        finished=True,
    )

    assert delta is not None
    assert delta.reasoning is None
    assert delta.content is None
    assert delta.tool_calls
    function_payloads = [tc.function for tc in delta.tool_calls]
    assert any(payload.get("name") == "write" for payload in function_payloads)
    combined_args = "".join(
        payload.get("arguments") or "" for payload in function_payloads
    )
    assert json.loads(combined_args) == {"file_path": "/tmp/x.txt"}


def test_streaming_zero_argument_glm47_call_emits_name(glm47_tool_parser, mock_request):
    _reset(glm47_tool_parser)
    current_text = "<tool_call>get_current_date</tool_call>"

    delta = glm47_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    assert delta is not None
    assert delta.tool_calls
    function_payloads = [tc.function for tc in delta.tool_calls]
    assert any(
        payload.get("name") == "get_current_date" for payload in function_payloads
    )
    combined_args = "".join(
        payload.get("arguments") or "" for payload in function_payloads
    )
    assert json.loads(combined_args) == {}
