from langchain.prompts import PromptTemplate

class RAGChatbot:
    def __init__(self, collection, model, llm, client):
        self.collection = collection
        self.model = model
        self.llm = llm
        self.client = client

    def search(self, query):
        # Search data
        query_text = query
        query_vector = self.model.encode([query_text])[0].astype("float32").tolist()

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=3,  # top 3 matches
        )

        for r in results:
            print(f"Found ID={r.id}, Score={r.score}, Text={r.payload['text']}")

        # End search data
        return results

    def answer(self, query):
        results = self.search(query)

        # Format retrieved docs into context
        context_texts = [r.payload.get("text", "") for r in results]
        context = "\n".join(context_texts)
                
        # Define a RAG prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant.\n"
                "Use the following context to answer the question. and you can modify the answer for a better content.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

        # Build final prompt
        final_prompt = prompt_template.format(context=context, question=query)

        # Call LLM
        response = self.llm.invoke(final_prompt)
        return response.content
