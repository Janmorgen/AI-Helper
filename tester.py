from langchain_core.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import *
import re
class Tester:
    doc_prompt = ChatPromptTemplate.from_messages([("system",
"""
Document Contents: 
{page_content}

Document source: {source} 
""")])
    prompt = ChatPromptTemplate.from_messages(
            [("system", 
              """"
Provided Information:
{context}
Previous Query:
{query}
Previous Response:
{response}
You are a QA tester, you must generate a single prompt to test the information retrieval and synthesis capabilities of a chatbot.
Rules:
    1. Your query must be related to the provided information, but does not need to adhere to the query and previous response
    2. Your query must include hypothetical elements to test for information synthesis between the provided documents
    3. You must respond in the provided response format
    4. You must answer in JSON format
    5. Your query must be as different as possible from the previous query, cover different topics, use different sources, and ask for connections between different subjects
Response format:
"query": string <- your test query,
"explanation": string array <- a list of general AI capabilties your test query aims to test
              """
              )])
    def __init__(self, llm):
        self.llm=llm
    def create_query(self, response):
        output={}

        qa_chain=create_stuff_documents_chain(llm=self.llm,prompt=self.prompt,document_prompt=self.doc_prompt)
        critique = qa_chain.invoke({"query":response["query"], "response":response["response"], "context":response["documents"]})
        print(critique)
        grade=re.search("\"query\": \"([\s\S]+)\"[,|\s]+\"explanation\"",critique)
        crit =re.search("\"explanation\": (\[[\s\S]+\])",critique)
        # successful = re.search("\"successful\": ([a-z]+)",critique)
        if (grade == None or crit==None):
            print(f"Parsing Failed: \n {critique}")
        else:
            output["query"]=grade.group(1)
            output["explanation"]=crit.group(1)
            # self.crit["successful"]=successful.group(1)
        return output