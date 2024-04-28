# Imports
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph

from dotenv import load_dotenv

load_dotenv()

# __________________________________________________________________________________________________
# _____________________________________ Building tools for analysis ________________________________
# __________________________________________________________________________________________________


@tool("Resume_portfolio_of_Ankit", return_direct=True)
def tool_pdf(input: str) -> str:
    """ Return the response for query related to individual named 'Ankit'. """
    print("Entering for PDF Analysis")
    embed = OpenAIEmbeddings()
    index = Pinecone.from_existing_index(index_name=os.environ["PINECONE_INDEX_NAME"],
                                         embedding=embed,
                                         namespace=os.environ["NAMESPACE"])
    docs = index.similarity_search(input, k=5)
    
    pdf_llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)

    chain = load_qa_chain(llm=pdf_llm, chain_type='stuff')

    response = chain.invoke({'question': input, 'input_documents': docs})

    return response['output_text']


@tool("Customer_analysis", return_direct=True)
def tool_csv(input: str) -> str:
    """Return the response for query asking information about the customer dataset. """
    print("Entering for CSV Analysis")
    file_path = 'path/to/your/csv/file'
    input_df = pd.read_csv(file_path)
    csv_llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
    agent = create_pandas_dataframe_agent(llm=csv_llm,
                                          df=input_df,
                                          verbose=False,
                                          agent_type=AgentType.OPENAI_FUNCTIONS,
                                          )
    answer = agent.invoke(input)

    return answer['output']


tools = [tool_csv, tool_pdf]

# __________________________________________________________________________________________________
# _____________________________________ Agent to Execute tools _____________________________________
# __________________________________________________________________________________________________

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm,
                                               tools,
                                               prompt)

tool_executor = ToolExecutor(tools)

# __________________________________________________________________________________________________
# ________________________________________ Defining Nodes __________________________________________
# __________________________________________________________________________________________________


# Define the agent/graph
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}


# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data['agent_outcome']
    # Execute the tool
    tool_output = tool_executor.invoke(agent_action)
    # print(f"The agent action is {agent_action}")
    # print(f"The tool result is: {output}")
    # Return the output
    return {"intermediate_steps": [(agent_action, str(tool_output))]}


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"

# __________________________________________________________________________________________________
# _______________________________________ Defining AgentState ______________________________________
# __________________________________________________________________________________________________


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# __________________________________________________________________________________________________
# ___________________________________ Designing Graph Workflow _____________________________________
# __________________________________________________________________________________________________

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


# __________________________________________________________________________________________________
# _____________________________________ Execution with user query __________________________________
# __________________________________________________________________________________________________

while True:
    # query = 'What are some LLM based project ? and what is the average age of male customers.'
    query = input("Enter your query: ")
    if query.lower() == 'exit':
        break
    else:
        inputs = {"input": query, "chat_history": []}

        # for s in app.stream(inputs):
        #     print(list(s.values())[0])
        #     print("----")
        output = app.invoke(inputs)

        print(output['agent_outcome'].return_values['output'])
