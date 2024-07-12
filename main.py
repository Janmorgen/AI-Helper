
from langchain_community.llms import Ollama


from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


from langchain_core.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
# import pyttsx3

import random

from critic import *
from retrievers import *
from tester import *
# from tester import *

# obtain audio from the microphone
# def listen():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Say something!")
#         audio = r.listen(source)

#     # try:
#         # for testing purposes, we're just using the default API key
#         # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
#         # instead of `r.recognize_google(audio)`
#     result = r.recognize_google(audio)
#     print("Google Speech Recognition thinks you said " + result)
#     # except sr.UnknownValueError:
#     #     print("Google Speech Recognition could not understand audio")
#     # except sr.RequestError as e:
#     #     print("Could not request results from Google Speech Recognition service; {0}".format(e))
#     return result

# def syth_speech(text):
#     engine = pyttsx3.init()
#     # if u'en_US' in voice.languages
#     voices = [voice for voice in engine.getProperty('voices') if b'en' in voice.languages[0]]
#     randomVoice = random.randint(0,len(voices)-1)
#     print(randomVoice)
#     engine.setProperty("voice",voices[randomVoice].id)
#     engine.say(text)
#     engine.runAndWait()

def process_llm_response(llm_response):
    print(f"\n\nQuestion: \n\t{llm_response['query']}\n------------")
    print(f"Result: \n\t{llm_response['result']}\n------------")
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        # print(source)
        # print(source.page_content)
        print(source.metadata['source']+" - Page: "+str(source.metadata['page'])+"\n------------")

llm = Ollama(model="llama3",
            verbose=False,
            base_url="http://localhost:11434",
            temperature=20,
            # callback_maager=CallbackManager([StreamingStdOutCallbackHandler()])
            )


retrievers = Retriever("paraphrase-distilroberta-base-v1",llm)
# retriever = EnsembleRetriever(retrievers=[retriever,prevResponses.as_retriever()], weights=[.8,.2])

# prompt used to answer user questions
prompt = ChatPromptTemplate.from_messages(
    [("system", "[INST] You are a helpful assistent who answers user questions based on information from the provided context. Feel free to extrapolate or hypothesize, but please adhere to the context closely while doing this, do not assume direct relationships between them unless explicitly apparent\n Query:\n{query}[/INST] \nContext:\n{context} "),
     
     ]
)
prompt = ChatPromptTemplate.from_messages(
    [("system", "[INST] You are a helpful assistent who answers user questions. Feel free to extrapolate or hypothesize, but please adhere to the context closely while doing this, do not assume direct relationships between documents unless explicitly provided.\n Query:\n{query}[/INST] \nContext:\n{context} "),
     
     ]
)
# prompt used for each document to format them appropriately
doc_prompt = ChatPromptTemplate.from_messages(
    [("system", "Document Contents: \n{page_content}\n\nDocument source= \n{source} ")]
)



responses=[]
sum_ratings=0
rating=0
docs=None
simularity=0
retrievers.createRetrieversJSON()
tester= Tester(llm)
previousResponse={}
while True:

    # query = listen()
    # if len(responses)==0:
    query = input("\n What is your question?\n  > ")
    # else:
    #     query=tester.create_query(previousResponse)["query"]
    
    # query = "Who is Karl Marx? "
    # else:
    #     query=tester(responses[-1])
    #     print("Generated query: "+query["query"]+"\n\nTest capabilities: "+query["explanation"]+"\n")
    #     query=query["query"]
    # docs =retriever.get_relevant_documents(query,tags=["page_content","query"])
    if docs !=None:
        for i in responses:
            # print(i.page_content)
            for y in query.split(" "):
                simularity=str(i.page_content).count(query)+simularity
        print(f"Query simularity with previous docs {simularity}")

    # if (simularity<10) or docs == None:
    docs=retrievers.FilterDocsMaxRatioColTest(query)
    print (len(docs))
    print (docs)
    
    if len(responses)>0:
    #     # docs[-1]=(prevResponses.as_retriever().get_relevant_documents(query)[0])
        docs.append(responses[-1])

    # print(docs)
    qa_chain=create_stuff_documents_chain(llm=llm,prompt=prompt,document_prompt=doc_prompt)
    # print(len(embedding_function.embed_query(query)))
    # documents = retriever.get_relevant_documents(query)
    llm_response = qa_chain.invoke({"query":query, "context":docs})
    previousResponse["query"]=query
    previousResponse["documents"]=docs
    previousResponse["response"]=llm_response
    
    
    print("================================")
    # print(retriever.get_relevant_documents(query))
    print(llm_response)
    # syth_speech(llm_response)
    print("================================")
    response=Response(query,llm_response,docs,llm)
    # responses.append(response)
    critic = Critic(response,"paraphrase-distilroberta-base-v1",llm)
    successfulParse=critic.critique()
    if (successfulParse!=False):
        critic.print()
        # critic.add()
        
        average_rating=0
        # sum_ratings+=int(critic.crit["grade"])
        # rating=int(critic.crit["grade"])
        # if len(responses)>0:
        #     average_rating=sum_ratings/len(responses)

        # if (int(critic.crit["grade"])>average_rating):
        doc = Document(page_content=critic.content,metadata=critic.context)
        responses.append(doc)
            # prevResponses.add_documents([doc])
        # vectorstore.add_documents(doc)
    # if len(responses)%10==0:
    #     print(
    #     f"""
    #     System processing last 10 responses...
    #     """)
    #     for i in range(len(responses)-10-1,len(responses)):
    #         responses.critique()
    #         responses.print()

    # process_llm_response(llm_response) 
