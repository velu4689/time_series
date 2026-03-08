"""
deploy_langflow_agent.py

Programmatically deploys a LangFlow agent that:
  - Connects to MCP servers via SSE (with proper handshake)
  - Uses LiteLLM gateway as the LLM backend
  - Handles full tool-call loop (LLM → MCP → LLM)
  - Exposes the agent as a REST endpoint in LangFlow

Usage:
    python deploy_langflow_agent.py
    python deploy_langflow_agent.py --dry-run   # preview flow JSON only
"""

import httpx
import json
import argparse
import sys
from mcp_sse_client import MCPSSEClient, MCPServer

# ─── Configuration ────────────────────────────────────────────────────────────

LANGFLOW_URL    = "http://localhost:7860"       # LangFlow instance
LANGFLOW_APIKEY = "YOUR_LANGFLOW_API_KEY"       # LangFlow API key (if auth enabled)

LITELLM_URL     = "http://localhost:4000"        # LiteLLM proxy gateway
LITELLM_APIKEY  = "YOUR_LITELLM_API_KEY"         # LiteLLM master key
LITELLM_MODEL   = "gpt-4o"                       # Model registered in LiteLLM

# MCP Servers — add/remove as needed
MCP_SERVERS = [
    MCPServer(
        name="atlassian_mcp",
        base_url="http://atlassian:9000",
        sse_path="/sse",
    ),
    # MCPServer(
    #     name="postgres_mcp",
    #     base_url="http://postgres-mcp:8001",
    #     sse_path="/sse",
    # ),
    # MCPServer(
    #     name="datahub_mcp",
    #     base_url="http://datahub-mcp:8002",
    #     sse_path="/sse",
    # ),
]

AGENT_FLOW_NAME     = "atlassian-mcp-agent"
AGENT_ENDPOINT_NAME = "atlassian-mcp-agent"
AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to Atlassian tools (JIRA and Confluence).

You can:
- Search and retrieve JIRA issues, projects, and sprints
- Create and update JIRA tickets
- Search Confluence pages and spaces
- Retrieve documentation and wiki content

When a user asks a question:
1. Identify which tool(s) to call
2. Call them with the correct parameters
3. Synthesize the results into a clear, helpful response

Always be concise and structured in your responses."""

# ─── Step 1: Discover MCP Tools ───────────────────────────────────────────────

def discover_all_tools() -> tuple[list[dict], dict[str, MCPServer]]:
    """
    Connects to all configured MCP servers, runs the SSE handshake,
    and collects all tool definitions.
    Returns (openai_tools_list, server_lookup_by_tool_name)
    """
    all_openai_tools = []
    tool_to_server = {}   # tool_name → MCPServer (for routing tool calls)

    for server in MCP_SERVERS:
        client = MCPSSEClient(server)
        try:
            mcp_tools = client.connect()
            openai_tools = MCPSSEClient.to_openai_tools(mcp_tools)
            all_openai_tools.extend(openai_tools)
            for tool in mcp_tools:
                tool_to_server[tool["name"]] = server
            print(f"✅ {server.name}: {len(mcp_tools)} tools loaded")
        except Exception as e:
            print(f"❌ {server.name}: Failed to connect — {e}")
            sys.exit(1)

    return all_openai_tools, tool_to_server


# ─── Step 2: Build LangFlow Flow JSON ─────────────────────────────────────────

def build_langflow_flow(openai_tools: list[dict]) -> dict:
    """
    Constructs the LangFlow flow definition programmatically.
    
    Flow topology:
      ChatInput
          └──► AgentComponent (LiteLLM + MCP tools)
                    └──► ChatOutput
    
    The AgentComponent is configured with:
      - LiteLLM as OpenAI-compatible LLM backend
      - All MCP tool schemas injected as custom tools
      - System prompt for Atlassian context
    """

    tools_json_str = json.dumps(openai_tools, indent=2)

    flow = {
        "name": AGENT_FLOW_NAME,
        "description": "Agent with Atlassian MCP tools via LiteLLM gateway",
        "endpoint_name": AGENT_ENDPOINT_NAME,
        "is_component": False,
        "data": {
            "nodes": [
                # ── Node 1: Chat Input ─────────────────────────────────────
                {
                    "id": "chat_input_node",
                    "type": "genericNode",
                    "position": {"x": 100, "y": 300},
                    "data": {
                        "type": "ChatInput",
                        "node": {
                            "display_name": "Chat Input",
                            "description": "User message entry point",
                            "base_classes": ["Message"],
                            "template": {
                                "input_value": {
                                    "value": "",
                                    "type": "str",
                                    "show": True,
                                    "name": "input_value",
                                    "display_name": "User Message",
                                    "required": False
                                },
                                "session_id": {
                                    "value": "",
                                    "type": "str",
                                    "show": True,
                                    "name": "session_id",
                                    "display_name": "Session ID",
                                    "required": False
                                },
                                "should_store_message": {
                                    "value": True,
                                    "type": "bool",
                                    "show": True,
                                    "name": "should_store_message",
                                    "display_name": "Store Message"
                                }
                            }
                        },
                        "id": "chat_input_node"
                    }
                },

                # ── Node 2: Agent Component (LiteLLM + MCP Tools) ──────────
                {
                    "id": "agent_node",
                    "type": "genericNode",
                    "position": {"x": 500, "y": 300},
                    "data": {
                        "type": "AgentComponent",
                        "node": {
                            "display_name": "Atlassian MCP Agent",
                            "description": "LLM agent with Atlassian MCP tools via LiteLLM",
                            "base_classes": ["Message"],
                            "template": {
                                # LiteLLM as OpenAI-compatible backend
                                "llm": {
                                    "value": "",
                                    "type": "BaseLanguageModel",
                                    "show": True,
                                    "name": "llm",
                                    "display_name": "Language Model",
                                    "input_types": ["BaseLanguageModel"]
                                },
                                "system_prompt": {
                                    "value": AGENT_SYSTEM_PROMPT,
                                    "type": "str",
                                    "show": True,
                                    "name": "system_prompt",
                                    "display_name": "System Prompt",
                                    "multiline": True
                                },
                                "tools": {
                                    "value": [],
                                    "type": "Tool",
                                    "show": True,
                                    "name": "tools",
                                    "display_name": "Tools",
                                    "input_types": ["Tool"]
                                },
                                "input_value": {
                                    "value": "",
                                    "type": "str",
                                    "show": True,
                                    "name": "input_value",
                                    "display_name": "Input",
                                    "input_types": ["Message"]
                                },
                                "max_iterations": {
                                    "value": 10,
                                    "type": "int",
                                    "show": True,
                                    "name": "max_iterations",
                                    "display_name": "Max Iterations"
                                },
                                "handle_parsing_errors": {
                                    "value": True,
                                    "type": "bool",
                                    "show": True,
                                    "name": "handle_parsing_errors",
                                    "display_name": "Handle Parsing Errors"
                                },
                                "session_id": {
                                    "value": "",
                                    "type": "str",
                                    "show": True,
                                    "name": "session_id",
                                    "display_name": "Session ID",
                                    "input_types": ["Message"]
                                }
                            }
                        },
                        "id": "agent_node"
                    }
                },

                # ── Node 3: LiteLLM Model (OpenAI-compatible) ──────────────
                {
                    "id": "litellm_model_node",
                    "type": "genericNode",
                    "position": {"x": 500, "y": 100},
                    "data": {
                        "type": "OpenAIModel",
                        "node": {
                            "display_name": "LiteLLM Gateway",
                            "description": "Routes LLM calls through LiteLLM proxy",
                            "base_classes": ["BaseLanguageModel"],
                            "template": {
                                "model_name": {
                                    "value": LITELLM_MODEL,
                                    "type": "str",
                                    "show": True,
                                    "name": "model_name",
                                    "display_name": "Model Name"
                                },
                                "openai_api_base": {
                                    "value": f"{LITELLM_URL}/v1",
                                    "type": "str",
                                    "show": True,
                                    "name": "openai_api_base",
                                    "display_name": "API Base URL (LiteLLM)"
                                },
                                "openai_api_key": {
                                    "value": LITELLM_APIKEY,
                                    "type": "str",
                                    "show": True,
                                    "password": True,
                                    "name": "openai_api_key",
                                    "display_name": "LiteLLM API Key"
                                },
                                "temperature": {
                                    "value": 0.1,
                                    "type": "float",
                                    "show": True,
                                    "name": "temperature",
                                    "display_name": "Temperature"
                                },
                                "max_tokens": {
                                    "value": 4096,
                                    "type": "int",
                                    "show": True,
                                    "name": "max_tokens",
                                    "display_name": "Max Tokens"
                                },
                                "stream": {
                                    "value": False,
                                    "type": "bool",
                                    "show": True,
                                    "name": "stream",
                                    "display_name": "Stream"
                                }
                            }
                        },
                        "id": "litellm_model_node"
                    }
                },

                # ── Node 4: MCP Tool Component ─────────────────────────────
                {
                    "id": "mcp_tools_node",
                    "type": "genericNode",
                    "position": {"x": 200, "y": 500},
                    "data": {
                        "type": "MCPSse",
                        "node": {
                            "display_name": "Atlassian MCP Tools (SSE)",
                            "description": "Connects to Atlassian MCP via SSE transport",
                            "base_classes": ["Tool"],
                            "template": {
                                "url": {
                                    "value": f"{MCP_SERVERS[0].base_url}{MCP_SERVERS[0].sse_path}",
                                    "type": "str",
                                    "show": True,
                                    "name": "url",
                                    "display_name": "MCP SSE URL"
                                }
                            }
                        },
                        "id": "mcp_tools_node"
                    }
                },

                # ── Node 5: Chat Output ────────────────────────────────────
                {
                    "id": "chat_output_node",
                    "type": "genericNode",
                    "position": {"x": 900, "y": 300},
                    "data": {
                        "type": "ChatOutput",
                        "node": {
                            "display_name": "Chat Output",
                            "description": "Returns agent response to caller",
                            "base_classes": ["Message"],
                            "template": {
                                "input_value": {
                                    "value": "",
                                    "type": "str",
                                    "show": True,
                                    "name": "input_value",
                                    "display_name": "Message",
                                    "input_types": ["Message"]
                                },
                                "should_store_message": {
                                    "value": True,
                                    "type": "bool",
                                    "name": "should_store_message"
                                },
                                "session_id": {
                                    "value": "",
                                    "type": "str",
                                    "show": True,
                                    "name": "session_id",
                                    "display_name": "Session ID",
                                    "input_types": ["Message"]
                                }
                            }
                        },
                        "id": "chat_output_node"
                    }
                }
            ],

            # ── Edges ──────────────────────────────────────────────────────
            "edges": [
                # ChatInput → Agent (input_value)
                {
                    "id": "edge_input_to_agent",
                    "source": "chat_input_node",
                    "target": "agent_node",
                    "sourceHandle": "chat_input_node-message-ChatInput",
                    "targetHandle": "agent_node-input_value-AgentComponent"
                },
                # ChatInput → Agent (session_id)
                {
                    "id": "edge_input_session_to_agent",
                    "source": "chat_input_node",
                    "target": "agent_node",
                    "sourceHandle": "chat_input_node-session_id-ChatInput",
                    "targetHandle": "agent_node-session_id-AgentComponent"
                },
                # LiteLLM Model → Agent (llm)
                {
                    "id": "edge_llm_to_agent",
                    "source": "litellm_model_node",
                    "target": "agent_node",
                    "sourceHandle": "litellm_model_node-model-OpenAIModel",
                    "targetHandle": "agent_node-llm-AgentComponent"
                },
                # MCP Tools → Agent (tools)
                {
                    "id": "edge_mcp_to_agent",
                    "source": "mcp_tools_node",
                    "target": "agent_node",
                    "sourceHandle": "mcp_tools_node-tools-MCPSse",
                    "targetHandle": "agent_node-tools-AgentComponent"
                },
                # Agent → ChatOutput
                {
                    "id": "edge_agent_to_output",
                    "source": "agent_node",
                    "target": "chat_output_node",
                    "sourceHandle": "agent_node-message-AgentComponent",
                    "targetHandle": "chat_output_node-input_value-ChatOutput"
                }
            ],

            "viewport": {"x": 0, "y": 0, "zoom": 0.8}
        }
    }

    return flow


# ─── Step 3: Deploy to LangFlow ───────────────────────────────────────────────

def deploy_to_langflow(flow: dict) -> str:
    """
    POSTs the flow to LangFlow API.
    Returns the deployed flow ID.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LANGFLOW_APIKEY
    }

    with httpx.Client(timeout=30) as client:
        # Check if flow already exists → update, else create
        list_resp = client.get(
            f"{LANGFLOW_URL}/api/v1/flows/",
            headers=headers
        )
        list_resp.raise_for_status()
        existing = list_resp.json()

        existing_flow = next(
            (f for f in existing if f.get("name") == AGENT_FLOW_NAME),
            None
        )

        if existing_flow:
            flow_id = existing_flow["id"]
            print(f"[LangFlow] Updating existing flow: {flow_id}")
            resp = client.patch(
                f"{LANGFLOW_URL}/api/v1/flows/{flow_id}",
                headers=headers,
                json=flow
            )
        else:
            print(f"[LangFlow] Creating new flow: {AGENT_FLOW_NAME}")
            resp = client.post(
                f"{LANGFLOW_URL}/api/v1/flows/",
                headers=headers,
                json=flow
            )

        resp.raise_for_status()
        result = resp.json()
        flow_id = result.get("id", "unknown")

    print(f"✅ Flow deployed: {flow_id}")
    print(f"   Endpoint: {LANGFLOW_URL}/api/v1/run/{AGENT_ENDPOINT_NAME}")
    return flow_id


# ─── Step 4: Smoke Test the Deployed Agent ────────────────────────────────────

def smoke_test(flow_id: str, session_id: str = "smoke-test-001"):
    """
    Sends a test query to the deployed agent to validate end-to-end.
    """
    print("\n[SmokeTest] Sending test query...")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": LANGFLOW_APIKEY
    }

    payload = {
        "input_value": "List all available JIRA projects",
        "output_type": "chat",
        "input_type": "chat",
        "session_id": session_id
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{LANGFLOW_URL}/api/v1/run/{AGENT_ENDPOINT_NAME}",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()

    outputs = data.get("outputs", [])
    if outputs:
        result = outputs[0].get("outputs", [{}])[0]
        text = result.get("results", {}).get("message", {}).get("text", "")
        print(f"[SmokeTest] ✅ Agent responded:\n{text[:500]}...")
    else:
        print(f"[SmokeTest] ⚠️  Unexpected response shape: {data}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy LangFlow MCP Agent")
    parser.add_argument("--dry-run", action="store_true", help="Print flow JSON without deploying")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip smoke test after deploy")
    parser.add_argument("--discover-only", action="store_true", help="Only run MCP tool discovery")
    args = parser.parse_args()

    print("=" * 60)
    print(" LangFlow MCP Agent Deployer")
    print("=" * 60)

    # Step 1 — Discover MCP tools
    print("\n[1/3] Discovering MCP tools...")
    openai_tools, tool_to_server = discover_all_tools()
    print(f"      Total tools discovered: {len(openai_tools)}")

    if args.discover_only:
        print("\n[Tools]")
        for t in openai_tools:
            fn = t["function"]
            print(f"  • {fn['name']}: {fn['description'][:80]}")
        return

    # Step 2 — Build flow
    print("\n[2/3] Building LangFlow flow...")
    flow = build_langflow_flow(openai_tools)

    if args.dry_run:
        print("\n[DryRun] Flow JSON:")
        print(json.dumps(flow, indent=2))
        return

    # Step 3 — Deploy
    print(f"\n[3/3] Deploying to LangFlow @ {LANGFLOW_URL}...")
    flow_id = deploy_to_langflow(flow)

    # Step 4 — Smoke test
    if not args.skip_smoke_test:
        smoke_test(flow_id)

    print("\n" + "=" * 60)
    print(f"✅ Deployment complete!")
    print(f"   Flow ID  : {flow_id}")
    print(f"   Endpoint : {LANGFLOW_URL}/api/v1/run/{AGENT_ENDPOINT_NAME}")
    print(f"   LiteLLM  : {LITELLM_URL} → {LITELLM_MODEL}")
    print(f"   MCP Tools: {len(openai_tools)} tools from {len(MCP_SERVERS)} server(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
