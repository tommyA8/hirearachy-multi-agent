class SQLTools:
    def __init__(self):
        pass

    # Example: create a predetermined tool call
    def list_tables(self, state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])

        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")

        return {"messages": [tool_call_message, tool_message, response]}


# # Example: force a model to create a tool call
# def call_get_schema(state: MessagesState):
#     # Note that LangChain enforces that all models accept `tool_choice="any"`
#     # as well as `tool_choice=<string name of tool>`.
#     llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
#     response = llm_with_tools.invoke(state["messages"])

#     return {"messages": [response]}