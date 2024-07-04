from langchain_groq import ChatGroq
import os

groq_key = os.environ.get("GROQ_KEY")

# Create a ChatGroq model
model = ChatGroq(model='mixtral-8x7b-32768', api_key=groq_key)

