from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

template1 = 'Write a joke about {topic}'
prompt_template1 = ChatPromptTemplate.from_template(template1)
prompt1 = prompt_template1.format(topic='dogs')
print(prompt1)

template2 = 'Write {number} jokes about {topic}'
prompt_template2 = ChatPromptTemplate.from_template(template2)
prompt2 = prompt_template2.format(number=3, topic='dogs')
print(prompt2)



# PART 3: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
    ]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)

# Extra Informoation about Part 3.
# This does work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)


# ---- Prompts with Chat models ----
from langchain_groq import ChatGroq
import os

groq_key = os.environ.get("GROQ_KEY")

# Create a ChatGroq model
model = ChatGroq(model='mixtral-8x7b-32768', api_key=groq_key)

result = model.invoke(prompt)
print("\n----- Result from Chat Model -----\n")
print(result.content)