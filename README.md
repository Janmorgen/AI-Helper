## NOTE
This is a project I made for recreational use only, it should not be relied on to produce errorless information. As such the output of the program should not be relied on for any important/critical usecases, especially if it affects any other users.

## Information
This "AI Helper" gathers information from documents provided in the code, pdf, and html folders and uses that information to answer user questions. I have also implemented a rudimentary form of chat history, so previous reponses can be referred to by the user.

I made this system to learn about some underlying systems in LangChain and how they might interact with an Ollama model. Although I have implemented various filters and additional systems to improve the responses given by the model, the system often conflates data between different sources and of course is still prone to hallucinations.

#### Improvements
I could improve the system by adding a more comprehensive fact checker to ensure the AI is sticking to data provided to it by the user documents. To reduce conflation between sources, I could encode the data with more contextual information. I have started on the last point by using the AI to summarize main points and suggest a title for every source it processes, but as of now I have elected not to store the processed data due to hallucinations and assumptions made by the model.


## Usage
To use the system you need Ollama installed on your local computer (this can be changed by modifying the agent IP address)
- `# apt install Ollama`
- (Optional) `ollama run llama3` 
- `pip install -r requirements.txt`
- (Optional) Add documents to /html, /pdf, /code to embed them to chromaDB, allowing the AI to access local data
- `python main.py`
