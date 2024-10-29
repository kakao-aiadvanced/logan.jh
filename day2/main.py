if __name__ == '__main__':
    import getpass
    import os
    from langchain_openai import ChatOpenAI

    os.environ["OPENAI_API_KEY"] = getpass.getpass()
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()


    llm = ChatOpenAI(model="gpt-4o-mini")

    import bs4
    from langchain import hub
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter


    loader = WebBaseLoader(
        web_paths=[
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
        model="text-embedding-3-small"
    ))

    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 6}
    )

    user_query = "agent memory"
    relevant_chunks = retriever.invoke(user_query)
    # for chunk in relevant_chunks:
    #     print(chunk)
    # print('-------------------------------------------------')

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel


    class RelevanceResponse(BaseModel):
        document_source: str
        content: str
        relevance: str

    parser = JsonOutputParser(pydantic_object=RelevanceResponse)

    prompt = PromptTemplate(
        template=(
            "Evaluate the relevance of the retrieved chunks to the following user query\n\nAnd include the document_source and content of each chunk, and respond with relevance: yes if it has relevance, or relevance:no if it doesn't."
            "user query: {query}\n"
            "retrieved chunks: {chunk}\n\n"
            "{format_instructions}\n"
        ),
        input_variables=["query", "chunk"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    results = chain.invoke({"query": user_query, "chunk": relevant_chunks})
    for idx, r in enumerate(results):
        print()
        print(f'<{idx+1}>')
        print(f'content: {r["content"]}')
        print(f'relevance: {r["relevance"]}')
        print(f'----------------------')
        print()

    all_no_results = chain.invoke({"query": user_query, "chunk": [
    "Lionel Messi is an Argentine professional soccer player.",
    "He is often regarded as one of the greatest footballers of all time.",
    "Messi spent the majority of his career at FC Barcelona.",
    "In 2021, he joined Paris Saint-Germain (PSG).",
    "Messi has won multiple Ballon d'Or awards for his exceptional performances."
    ]})
    print(f"all_no_results size: {len(all_no_results)}")
    print(f"actual no results size: {len([item for item in all_no_results if item.get('relevance') == 'no'])}")

    all_yes_results = chain.invoke({"query": user_query,
                                   "chunk": [
    "Memory plays a crucial role in enabling LLM agents to understand conversations and maintain consistent context.",
    "An LLM agent's memory stores elements of past interactions to facilitate more natural and coherent exchanges.",
    "The agent's memory utilizes learned information to ensure accurate responses to user inquiries.",
    "The memory system of conversational agents is updated and expanded in real-time, enhancing user experience.",
    "Efficient memory management is essential for LLM agents to respond flexibly and adaptively to a wide range of topics."
    ]})
    print(f"all_yes_results size: {len(all_yes_results)}")
    print(f"actual yes results size: {len([item for item in all_yes_results if item.get('relevance') == 'yes'])}")

    filtered_list = [item for item in results if item.get('relevance') == 'yes']

    system="""
    Kindly reply with reference to context.
    """

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "question: {question}\n\n context: {context} ")])
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"question": user_query, "context": filtered_list})
    print(generation)