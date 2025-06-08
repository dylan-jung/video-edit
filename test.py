from src.server.agent.tools.list_videos_tool import ListVideosTool

tool = ListVideosTool(project_id="test")
print(tool.call())