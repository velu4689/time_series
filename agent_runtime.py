"""
agent_runtime.py

Standalone agent loop that:
  1. Connects to MCP servers (SSE handshake)
  2. Sends queries to LiteLLM gateway
  3. Executes MCP tool calls when the LLM requests them
  4. Returns final responses

Use this to TEST the agent locally without needing LangFlow running.
Once validated, deploy via deploy_langflow_agent.py.

Usage:
    python agent_runtime.py
    python agent_runtime.py --query "List open JIRA bugs in sprint 42"
"""

import httpx
import json
import argparse
from mcp_sse_client import MCPSSEClient, MCPServer

# ─── Config (match deploy_langflow_agent.py) ──────────────────────────────────

LITELLM_URL    = "http://localhost:4000"
LITELLM_APIKEY = "YOUR_LITELLM_API_KEY"
LITELLM_MODEL  = "gpt-4o"

MCP_SERVERS = [
    MCPServer(
        name="atlassian_mcp",
        base_url="http://atlassian:9000",
        sse_path="/sse",
    ),
]

SYSTEM_PROMPT = """You are a helpful assistant with access to Atlassian tools (JIRA and Confluence).
Use the available tools to answer questions accurately. Always call tools when needed rather than guessing."""

MAX_ITERATIONS = 10   # prevent infinite tool-call loops


# ─── Agent Loop ───────────────────────────────────────────────────────────────

class AgentRuntime:
    def __init__(self):
        self.mcp_clients: dict[str, MCPSSEClient] = {}
        self.tool_to_server: dict[str, MCPServer] = {}
        self.openai_tools: list[dict] = []

    def connect_mcp_servers(self):
        """Runs SSE handshake for all configured MCP servers."""
        for server in MCP_SERVERS:
            client = MCPSSEClient(server)
            mcp_tools = client.connect()
            self.mcp_clients[server.name] = client
            for tool in mcp_tools:
                self.tool_to_server[tool["name"]] = server
            self.openai_tools.extend(MCPSSEClient.to_openai_tools(mcp_tools))

        print(f"\n[Agent] Ready. {len(self.openai_tools)} tools available:")
        for t in self.openai_tools:
            print(f"  • {t['function']['name']}: {t['function']['description'][:70]}")

    def _call_litellm(self, messages: list[dict]) -> dict:
        """Calls LiteLLM proxy (OpenAI-compatible endpoint)."""
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{LITELLM_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {LITELLM_APIKEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LITELLM_MODEL,
                    "messages": messages,
                    "tools": self.openai_tools,
                    "tool_choice": "auto",
                    "temperature": 0.1,
                    "max_tokens": 4096
                }
            )
            resp.raise_for_status()
            return resp.json()

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Finds the right MCP server and executes the tool."""
        server = self.tool_to_server.get(tool_name)
        if not server:
            return f"Error: tool '{tool_name}' not found in any MCP server"

        client = self.mcp_clients[server.name]
        print(f"  [Tool] {tool_name}({json.dumps(arguments, separators=(',',':'))[:100]})")

        try:
            result = client.call_tool(tool_name, arguments)
            print(f"  [Tool] ✅ Result: {result[:120]}...")
            return result
        except Exception as e:
            error = f"Tool execution error: {str(e)}"
            print(f"  [Tool] ❌ {error}")
            return error

    def run(self, user_query: str) -> str:
        """
        Full agent loop:
        1. Send query + tools to LiteLLM
        2. If LLM wants to call tools → execute via MCP → feed results back
        3. Repeat until LLM gives a final text response
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        print(f"\n[Agent] Query: {user_query}")
        print("-" * 50)

        for iteration in range(MAX_ITERATIONS):
            response = self._call_litellm(messages)
            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice["finish_reason"]

            # ── Final answer — no more tool calls ────────────────────────
            if finish_reason == "stop":
                final = message.get("content", "")
                print(f"\n[Agent] Final answer after {iteration + 1} iteration(s)")
                return final

            # ── LLM wants to call tools ───────────────────────────────────
            if finish_reason == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                print(f"\n[Agent] Iteration {iteration + 1}: {len(tool_calls)} tool call(s)")

                # Add assistant message with tool_calls to history
                messages.append({
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls
                })

                # Execute each tool and add results to message history
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    result = self._execute_tool(tool_name, arguments)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result
                    })

                # Loop back — send tool results to LLM for next step
                continue

            # ── Unexpected finish reason ──────────────────────────────────
            print(f"[Agent] Unexpected finish_reason: {finish_reason}")
            return message.get("content", "Unexpected agent termination.")

        return "Agent reached max iterations without a final answer."


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MCP Agent locally")
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    agent = AgentRuntime()
    agent.connect_mcp_servers()

    if args.query:
        answer = agent.run(args.query)
        print(f"\n{'='*50}\nAnswer:\n{answer}\n{'='*50}")

    elif args.interactive:
        print("\n[Interactive Mode] Type 'exit' to quit\n")
        while True:
            query = input("You: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue
            answer = agent.run(query)
            print(f"\nAgent: {answer}\n")

    else:
        # Default: run a set of sample queries
        test_queries = [
            "List all JIRA projects",
            "Find open bugs assigned to me in the current sprint",
            "Search Confluence for onboarding documentation",
        ]
        for q in test_queries:
            answer = agent.run(q)
            print(f"\n{'='*50}")
            print(f"Q: {q}")
            print(f"A: {answer}")
            print("=" * 50)


if __name__ == "__main__":
    main()
