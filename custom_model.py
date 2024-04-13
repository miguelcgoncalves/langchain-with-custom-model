import json
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Type, Union
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import BaseTool
from pollinations_ai import PollinationsAIChat

client = PollinationsAIChat()


class CustomModel(BaseChatModel):
    model_name: str = "gpt-4o"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    def _convert_to_oai_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        oai_role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
        }
        formatted_messages = []
        for message in messages:
            if isinstance(message, ToolMessage):
                formatted_messages.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            elif isinstance(message, AIMessage) and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "function": {
                                "arguments": json.dumps(tool_call["args"]),
                                "name": tool_call["name"],
                            },
                            "id": tool_call["id"],
                            "type": "function",
                        }
                    )
                formatted_messages.append(
                    {"role": "assistant", "tool_calls": tool_calls, "content": ""}
                )
            else:
                formatted_messages.append(
                    {"role": oai_role[message.type], "content": message.content}
                )

        return formatted_messages

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = True,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
            else:
                pass
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in (
                    "file_search",
                    "web_search_preview",
                    "computer_use_preview",
                ):
                    tool_choice = {"type": tool_choice}
                elif tool_choice == "any":
                    tool_choice = "required"
                else:
                    pass
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        serialized_messages = self._convert_to_oai_messages(messages)
        payload: Any = {"messages": serialized_messages}
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        response = client.create_completions(**payload)
        choice = response["choices"][0]
        if choice["finish_reason"] == "stop":
            message = AIMessage(
                content=choice["message"]["content"],
                additional_kwargs={},
                response_metadata={},
                usage_metadata={
                    "input_tokens": response["usage"]["prompt_tokens"],
                    "output_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"],
                },
            )
        elif choice["finish_reason"] == "tool_calls":
            tool_calls = [
                ToolCall(
                    name=tool_call["function"]["name"],
                    args=json.loads(tool_call["function"]["arguments"]),
                    id=tool_call["id"],
                )
                for tool_call in choice["message"]["tool_calls"]
            ]
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            raise Exception(f"finish_reason {choice['finish_reason']}")

        generations = [ChatGeneration(message=message)]
        return ChatResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model_name,
        }
