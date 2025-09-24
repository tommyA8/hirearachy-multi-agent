from langchain_core.messages import HumanMessage
from model.user import UserContext
from workflows.chat_cm import chatcm_agent

agent = chatcm_agent()

def main():
  i = 0
  while True:
      question = input("#> ")

      if i == 0:
        for step in agent.stream({"messages": [HumanMessage(content=question)], 
                                  "user": UserContext(user_id=1, company_id=1, project_id=1, tool_permissions=None),
                                  }, 
                                stream_mode="values", 
                                config={"configurable": {"thread_id": "demo-user-002"}}):
            
            step["messages"][-1].pretty_print()
      else:
        for step in agent.stream({"messages": [HumanMessage(content=question)]},
                                  stream_mode="values", 
                                  config={"configurable": {"thread_id": "demo-user-002"}}):

            step["messages"][-1].pretty_print()

      i += 1

if __name__ == "__main__":
    main()