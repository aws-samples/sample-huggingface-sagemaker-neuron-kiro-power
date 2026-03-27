"""SageMaker AI + Neuron MCP Server for Hugging Face models."""

from mcp.server.fastmcp import FastMCP

from .tools.deploy import register_deploy_tools
from .tools.training import register_training_tools
from .tools.endpoint import register_endpoint_tools
from .tools.recommend import register_recommend_tools

mcp = FastMCP("sagemaker-neuron")

register_deploy_tools(mcp)
register_training_tools(mcp)
register_endpoint_tools(mcp)
register_recommend_tools(mcp)

if __name__ == "__main__":
    mcp.run()
