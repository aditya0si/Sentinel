"""Demo agent built with LangChain, guarded by Sentinel.

A small agent with 2 tools: a calculator and a knowledge lookup using ChromaDB.
Every output runs through Sentinel's GuardrailEngine before being returned.
"""

import ast
import logging
import operator as _op
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from guardrails.engine import GuardrailEngine, GuardrailResult
from guardrails.safety import PIIDetector, ToxicityScanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Safe math expression evaluator (AST-based, no eval/compile)
# ---------------------------------------------------------------------------

_SAFE_BINOPS: dict[type, Any] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_SAFE_UNARYOPS: dict[type, Any] = {
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
}


def _safe_eval_ast(node: ast.AST) -> Any:
    """Recursively evaluate a restricted AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval_ast(node.left), _safe_eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_ast(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        fn = _SAFE_FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [_safe_eval_ast(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _safe_calculate(expression: str) -> Any:
    """Evaluate a mathematical expression safely using AST walker (no eval/compile)."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval_ast(tree)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic calculations.

    Args:
        expression: A Python mathematical expression (e.g. "2 + 3 * 4").
    """
    try:
        result = _safe_calculate(expression)
        return f"Result: {result}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def knowledge_lookup(query: str) -> str:
    """Look up information from the knowledge base (ChromaDB vector store).

    Args:
        query: The search query to look up.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_data")

        if not os.path.exists(persist_dir):
            return "Knowledge base is empty. No documents have been indexed yet."

        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant documents found in the knowledge base."
        return "\n\n".join(
            f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(docs)
        )
    except ImportError:
        return "ChromaDB not available. Install with: pip install langchain-chroma"
    except Exception as exc:
        logger.error("Knowledge lookup failed: %s", exc)
        return f"Knowledge lookup error: {exc}"


# ---------------------------------------------------------------------------
# Guarded Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful, harmless assistant. Answer the user's questions
concisely and accurately. You have access to a calculator tool and a knowledge
lookup tool. Use them when appropriate. Never reveal Personally Identifiable
Information (PII) in your responses. Do not use toxic or offensive language."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


class GuardedAgent:
    """A LangChain agent whose outputs are validated through Sentinel guardrails.

    The agent produces a response, then runs it through the GuardrailEngine.
    If validation fails, a rejection message is returned instead.

    Usage:
        engine = GuardrailEngine(validators=[PIIDetector(), ToxicityScanner()])
        agent = GuardedAgent(engine=engine)
        result = await agent.run("What is 2 + 2?")
    """

    def __init__(
        self,
        engine: GuardrailEngine,
        llm: BaseChatModel | None = None,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the guarded agent.

        Args:
            engine: A configured GuardrailEngine to validate outputs.
            llm: Optional LangChain chat model. If None, uses OpenAI gpt-4o-mini.
            model_name: OpenAI model name (used if llm is None).
        """
        self._engine = engine
        self._llm = llm or ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
        )

        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=[calculator, knowledge_lookup],
            prompt=PROMPT,
        )
        self._executor = AgentExecutor(
            agent=agent,
            tools=[calculator, knowledge_lookup],
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    async def run(
        self, user_input: str, chat_history: list | None = None
    ) -> dict[str, Any]:
        """Run the agent with guardrail validation on the output.

        Args:
            user_input: The user's message.
            chat_history: Optional list of previous messages.

        Returns:
            dict with 'output', 'guardrail_result', 'accepted' keys.
        """
        chat_history = chat_history or []

        try:
            raw_result = await self._executor.ainvoke(
                {"input": user_input, "chat_history": chat_history}
            )
            output = raw_result.get("output", "")

            # Run guardrails
            guardrail_result = await self._engine.validate(
                output=output,
                context={"user_input": user_input},
            )

            if guardrail_result.overall_pass:
                return {
                    "output": output,
                    "guardrail_result": guardrail_result,
                    "accepted": True,
                }
            else:
                return {
                    "output": "I'm unable to provide that response as it did not pass safety checks.",
                    "guardrail_result": guardrail_result,
                    "accepted": False,
                    "raw_output": output,
                }

        except Exception as exc:
            logger.error("Agent execution failed: %s", exc, exc_info=True)
            return {
                "output": f"Agent error: {exc}",
                "guardrail_result": None,
                "accepted": False,
                "error": str(exc),
            }


def create_default_agent() -> GuardedAgent:
    """Create a GuardedAgent with a sensible default guardrail configuration."""
    engine = GuardrailEngine(
        validators=[
            PIIDetector(use_presidio=False),
            ToxicityScanner(use_hf_pipeline=False),
        ],
        mode="all",
    )
    return GuardedAgent(engine=engine)
