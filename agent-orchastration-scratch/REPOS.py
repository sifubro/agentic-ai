
'''
https://github.com/sifubro/GenAI_Agents_NirDiamant/blob/main/all_agents_tutorials/Weather_Disaster_Management_AI_AGENT.ipynb





PAGES:
DEEP AGENTS
- https://www.datacamp.com/tutorial/deep-agents
Has helper functions handle the complexity of extracting text from different file formats (PDF, DOCX, TXT) and converting markdown output back to DOCX format for download
- https://www.colinmcnamara.com/blog/deep-agents-part-4-usage-integration-roadmap

- 





Literal, Enum, Field constraints, Validators
@field_validator
Using Pydantic v2 Field Constraints for Even Stronger Validation
https://chatgpt.com/c/6928a49a-e490-832b-8c88-4c5258cca5cc#


TODO:
In the a) You mentioned that The Field(description=...) metadata gets converted into a JSON schema that's added to the prompt:
print(parser.get_format_instructions())
The output should be formatted as a JSON instance that conforms to the JSON schema below....
Please provide a complete example showing the full prompt to understand. Is it concatenated to the previous prompt?

What happens if the models does not conform to the format instructions specified in the prompt?

How to pass the error traceback back to the LLM to correct itself and answer in a structured manner?
How OutputFixingParser works:
from langchain.output_parsers import OutputFixingParser
# Wrap with OutputFixingParser - it will try to fix errors automatically!

fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

retry_parser = RetryWithErrorOutputParser.from_llm

parser = PydanticOutputParser(pydantic_object=Answer)

structured_llm = llm.with_structured_output(Answer)






agent_scratchpad


'''





























