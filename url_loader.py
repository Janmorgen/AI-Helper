





# raw_text = ''
# for i in arr:
#     path = "./new_articles/"+i
#     doc_reader = PdfReader(path)
#     for i, page in enumerate(doc_reader.pages):
#         text = page.extract_text()
#         if text:
#             raw_text += text
# print(len(raw_text))
from langchain_community.document_loaders import PyPDFDirectoryLoader

ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
unique_ids = list(set(ids))
seen_ids = set()
if os.path.exists("./pdfs/processed.txt"):
    with open('./pdfs/processed.txt') as f:
        lines=f.readlines()
        for l in lines:
            seen_ids.add(l.replace("\n",""))
else:
    fp = open('./pdfs/processed.txt', 'x')
    fp.close()
# print(seen_ids)
new_docs = [doc for doc, id in zip(docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

# print(new_docs)
vectorstore=None
if len(new_docs)>0:
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 400,
            chunk_overlap  = 100, #striding over the text
        length_function = len,
    )
    # text_splitter = SemanticChunker(OllamaEmbeddings(model="mistral:instruct"))
    all_splits = text_splitter.split_documents(new_docs)
    print(f"Split into {len(all_splits)} chunks")

    with open('./pdfs/processed.txt','w') as f:
        for u in seen_ids:
            f.write(u+"\n")
    vectorstore = Chroma.from_documents(persist_directory="./pdfs",
                                        documents=all_splits, 
                                        embedding=embedding_function)
else:
    vectorstore = Chroma(persist_directory="./pdfs",
                        embedding_function=embedding_function)

loader = PyPDFDirectoryLoader("./new_articles/")
docs = loader.load()