"""Normalize Chat Completions and Responses while preserving reasoning state."""
import os
from types import SimpleNamespace


class ModelTransport:
    def __init__(self, client, model, tool_schemas):
        self.client = client
        self.model = model
        self.tools = tool_schemas
        default = 'responses' if model.startswith(('gpt-5.6', 'gpt-6')) else 'chat'
        self.api = os.getenv('OPENAI_API_MODE', default)
        if self.api not in ('chat', 'responses'):
            raise ValueError('OPENAI_API_MODE must be chat or responses')
        self.input = []
        self.cursor = 0

    def next_message(self, messages):
        if self.api == 'chat':
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools,
                tool_choice='auto', response_format={'type': 'json_object'})
            return response.choices[0].message
        # Assistant output is already preserved below, including encrypted reasoning.
        # Append only newly generated user prompts and function results.
        for message in messages[self.cursor:]:
            if message['role'] == 'tool':
                self.input.append({'type': 'function_call_output',
                                   'call_id': message['tool_call_id'], 'output': message['content']})
            elif message['role'] != 'assistant':
                self.input.append(message)
        self.cursor = len(messages)
        response = self.client.responses.create(
            model=self.model, input=self.input,
            tools=[{'type': 'function', **t['function'], 'strict': False} for t in self.tools],
            tool_choice='auto', text={'format': {'type': 'json_object'}},
            store=False, include=['reasoning.encrypted_content'])
        self.input.extend(item.model_dump(exclude_none=True) for item in response.output)
        calls = [SimpleNamespace(id=item.call_id, function=SimpleNamespace(
            name=item.name, arguments=item.arguments)) for item in response.output if item.type == 'function_call']
        return SimpleNamespace(content=response.output_text, tool_calls=calls,
                               model_dump=lambda **kwargs: {'role': 'assistant', 'content': response.output_text})
