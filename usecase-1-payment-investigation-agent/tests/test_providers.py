"""Exercise actual SDK serialization with in-memory HTTP responses."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from anthropic import Anthropic
from openai import OpenAI

from agent.agent import TOOL_SCHEMAS
from agent.providers import Conversation, Settings, read_settings


class ProviderTests(unittest.TestCase):
    def test_chat_completions_preserves_tool_metadata(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            message = {"role": "assistant", "content": "Done"}
            if len(requests) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_payment", "arguments": '{"payment_id":"P50000"}'},
                    "extra_content": {"google": {"thought_signature": "opaque-test-signature"}},
                }]}
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "test", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]})
        with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            conversation = Conversation(client, Settings("openai", "test", "chat_completions"), "system", "user")
            with patch.dict(os.environ, {"LLM_CHAT_TOKEN_FIELD": "max_tokens"}):
                _, calls = conversation.request(TOOL_SCHEMAS)
                conversation.add_results([{"id": calls[0]["id"], "result": {"amount": 12000}}])
                text, calls = conversation.request(TOOL_SCHEMAS)
        self.assertEqual(text, "Done")
        self.assertFalse(calls)
        history = requests[1]["messages"]
        self.assertEqual(history[-1]["role"], "tool")
        self.assertEqual(history[-1]["tool_call_id"], "call_1")
        self.assertEqual(history[-2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "opaque-test-signature")
        self.assertEqual(requests[0]["max_tokens"], 4096)

    def test_responses_preserves_reasoning_and_call_ids(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            output = [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "Done", "annotations": []}]}]
            if len(requests) == 1:
                output = [{"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque-reasoning"},
                          {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_payment",
                           "arguments": '{"payment_id":"P50000"}'}]
            return httpx.Response(200, json={"id": "resp_1", "object": "response", "created_at": 0,
                                           "model": "test", "status": "completed", "output": output})
        with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            conversation = Conversation(client, Settings("openai", "test", "responses"), "system", "user")
            _, calls = conversation.request(TOOL_SCHEMAS)
            conversation.add_results([{"id": calls[0]["id"], "result": {"amount": 12000}}])
            text, calls = conversation.request(TOOL_SCHEMAS)
        self.assertEqual(text, "Done")
        self.assertFalse(calls)
        history = requests[1]["input"]
        self.assertTrue(any(item.get("encrypted_content") == "opaque-reasoning" for item in history))
        self.assertEqual(history[-1]["type"], "function_call_output")
        self.assertEqual(history[-1]["call_id"], "call_1")
        self.assertFalse(requests[1]["store"])
        self.assertEqual(requests[1]["text"]["format"]["type"], "json_schema")
        self.assertTrue(requests[1]["text"]["format"]["strict"])
        self.assertEqual(requests[0]["tools"][0]["name"], "get_payment")

    def test_anthropic_batches_results_immediately_after_tool_use(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            content = [{"type": "text", "text": "Done"}]
            if len(requests) == 1:
                content = [{"type": "tool_use", "id": f"tool_{index}", "name": "get_payment",
                            "input": {"payment_id": "P50000"}} for index in range(2)]
            return httpx.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant",
                                           "model": "test", "content": content,
                                           "stop_reason": "tool_use" if len(requests) == 1 else "end_turn",
                                           "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})
        with Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            conversation = Conversation(client, Settings("anthropic", "test", "messages"), "system", "user")
            _, calls = conversation.request(TOOL_SCHEMAS)
            conversation.add_results([{"id": call["id"], "result": {"amount": 12000}} for call in calls])
            text, calls = conversation.request(TOOL_SCHEMAS)
        self.assertEqual(text, "Done")
        self.assertFalse(calls)
        history = requests[1]["messages"]
        self.assertEqual([item["role"] for item in history], ["user", "assistant", "user"])
        self.assertEqual([block["tool_use_id"] for block in history[-1]["content"]], ["tool_0", "tool_1"])
        self.assertIn("input_schema", requests[0]["tools"][0])

    def test_provider_selection_and_api_overrides(self):
        examples = [
            ({"OPENAI_API_KEY": "test", "OPENAI_MODEL": "test"}, "openai", "responses"),
            ({"OPENAI_API_KEY": "test", "OPENAI_MODEL": "test", "OPENAI_BASE_URL": "http://localhost:11434/v1"}, "openai", "chat_completions"),
            ({"ANTHROPIC_API_KEY": "test", "ANTHROPIC_MODEL": "test"}, "anthropic", "messages"),
            ({"AZURE_OPENAI_API_KEY": "test", "AZURE_OPENAI_DEPLOYMENT": "test",
              "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com", "AZURE_OPENAI_API_VERSION": "test"}, "azure", "chat_completions"),
        ]
        for environment, provider, api in examples:
            with self.subTest(provider=provider, api=api), patch.dict(os.environ, environment, clear=True), patch("dotenv.load_dotenv"):
                settings = read_settings()
                self.assertEqual((settings.provider, settings.api), (provider, api))


if __name__ == "__main__":
    unittest.main()
