from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = ChatGroq(model='llama3-8b-8192')

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    number: str = Field(description="The number of books written by the author")
    books: list[str] = Field(description="The list of books written by the author")


structured_llm = llm.with_structured_output(Author)
returned_object = structured_llm.invoke("Generate the books written by Dan Brown")

print(f"{returned_object.name} wrote {returned_object.number} books.")
print(returned_object.books)

# output_parser = PydanticOutputParser(pydantic_object=Author)
# prompt_list = PromptTemplate.from_template(
#     template = "Answer the question.\n{format_instructions}\n{question}",
#     partial_variables = {"format_instructions": output_parser.get_format_instructions()},
# )
#
# prompt_value = prompt_list.invoke({"question": "Generate the books written by Dan Brown"})
# response = llm.invoke(prompt_value)
#
# returned_object = output_parser.parse(response.content)
# print(returned_object)
# print(f"{returned_object.name} wrote {returned_object.number} books.")
# print(returned_object.books)