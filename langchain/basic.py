from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
#DatetimeOutputParser

llm = ChatGroq(model='llama3-8b-8192')
parser_list = CommaSeparatedListOutputParser()
prompt_list = PromptTemplate.from_template(
    template='Answer the question.\n{format_instructions}\n{question}',
    partial_variables={'format_instructions': parser_list.get_format_instructions()}
)

prompt_value = prompt_list.invoke({"question": "List 4 chocolate brands"})
response = llm.invoke(prompt_value)
returned_object = parser_list.parse(response.content)
print(returned_object)
print(type(returned_object))

# parser_dateTime = DatetimeOutputParser()
# prompt_dateTime = PromptTemplate.from_template(
#     template="Answer the question.\n{format_instructions}\n{question}",
#     # input_variables="question",
#     partial_variables={"format_instructions": parser_dateTime.get_format_instructions()},
# )
# prompt_value = prompt_dateTime.invoke({"question": "When was the iPhone released"})
# response = llm.invoke(prompt_value)
# returned_object = parser_dateTime.parse(response.content)
# print(returned_object)




# email_template = PromptTemplate.from_template(
#   "Create an invitation email to the recipient that is {recipient_name}\
# for an event that is {event_type}\
# in a language that is {language}\
# Mention the event location that is {event_location}\
# and event date that is {event_date}.\
# Also write few sentences about the event description that is {event_description}\
# in style that is {style}."
# )
#
# details = {
#   "recipient_name":"John",
#   "event_type":"product launch",
#   "language": "Spanish",
#   "event_location":"Grand Ballroom, City Center Hotel",
#   "event_date":"11 AM, January 15, 2024",
#   "event_description":"an exciting unveiling of our latest GenAI product",
#   "style":"enthusiastic tone"
# }
#
# prompt_value = email_template.invoke(details)
# response = llm.invoke(prompt_value)


# messages = [
#   SystemMessage(content="You are a kpop singer"),
#   HumanMessage(content="What is the square of 2?"),
# ]
# response = llm.invoke(messages)
# print(response.content)
# response  = llm.stream("Tell me an interesting fact about the cats")
# for chunk in response:
#     print(chunk.content, end="")

