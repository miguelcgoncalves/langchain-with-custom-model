import json
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
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
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from pydantic import BaseModel
import requests, os

COMPLETIONS_API_URL = os.environ["COMPLETIONS_API_URL"]


class CustomModel(BaseChatModel):
    model: str = "gpt-4o"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

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
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model,
        }

    def with_structured_output(
        self,
        schema: Union[Dict, type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        llm = self.bind_tools([schema], tool_choice="any")
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast("TypeBaseModel", schema)], first_tool_only=True
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
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
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
        if tool_choice and isinstance(tool_choice, str):
            if tool_choice in tool_names:
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            elif tool_choice in (
                "file_search",
                "web_search_preview",
                "computer_use_preview",
            ):
                tool_choice = {"type": tool_choice}
            elif tool_choice == "any":
                tool_choice = "required"
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def _convert_to_openai_message(self, message: BaseMessage) -> Dict[str, Any]:
        roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        msg = {"role": roles[message.type], "content": message.content}
        if isinstance(message, AIMessage) and message.tool_calls:
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
            return {**msg, "tool_calls": tool_calls}
        if isinstance(message, ToolMessage):
            return {**msg, "tool_call_id": message.tool_call_id}
        return msg

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: Any = {
            "messages": [self._convert_to_openai_message(m) for m in messages]
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs["tool_choice"]
        response = requests.post(COMPLETIONS_API_URL, json=payload)
        if not response.ok:
            raise Exception(response.text)
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]
        usage = data.get("usage", {})
        if finish_reason == "tool_calls":
            tool_calls = [
                ToolCall(
                    name=tool_call["function"]["name"],
                    args=json.loads(tool_call["function"]["arguments"]),
                    id=tool_call["id"],
                )
                for tool_call in message["tool_calls"]
            ]
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            message = AIMessage(
                content=message["content"],
                additional_kwargs={},
                response_metadata={},
                usage_metadata={
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
            )
        generations = [ChatGeneration(message=message)]
        return ChatResult(generations=generations)
