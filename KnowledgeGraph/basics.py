import langchain
import langchain_core
import langchain_community
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    temperature=0, model_name="gemma2-9b-it", api_key=os.environ.get("GROQ_KEY")
)
# Hybrid search = keyword search + dense vector search(semantic)
""" 
    with hybrid search and knowledge graphs can give better results
    cyper query language is used to retrieve data from graph database.
    Graph data consist nodes, relationships and properties.

"""

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

from langchain_community.graphs import Neo4jGraph

langchain.verbose = True
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

from langchain_core.documents import Document

text = """
Elon Reeve Musk (born June 28, 1971) is a businessman and investor known for his key roles in space
company SpaceX and automotive company Tesla, Inc. Other involvements include ownership of X Corp.,
formerly Twitter, and his role in the founding of The Boring Company, xAI, Neuralink and OpenAI.
He is one of the wealthiest people in the world; as of July 2024, Forbes estimates his net worth to be
US$221 billion.Musk was born in Pretoria to Maye and engineer Errol Musk, and briefly attended
the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through
his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada.
Musk later transferred to the University of Pennsylvania and receivedBachelor's degrees in economics
 and physics. He moved to California in 1995 to attend Stanford University, but dropped out after
  two days and, with his brother Kimbal, co-founded online city guide software company Zip2.
 """
documents = [Document(page_content=text)]

from langchain_experimental.graph_transformers import LLMGraphTransformer

llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(graph_documents[0].nodes)
