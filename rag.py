from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from detector import chunk_scanner

class RAGEnv:
    def __init__(self, docs, llm, embeddings):
        documents = []
        for i, d in enumerate(docs):
            documents.append(Document(
                page_content=d,
                metadata={"doc_id": i}
            ))
        self.db = FAISS.from_documents(documents, embeddings)
        self.llm = llm
        # Freeze the underlying model parameters
        for p in self.llm.model.parameters():
            p.requires_grad = False

    def query(self, q):
        docs = self.db.similarity_search(q, k=3)
        context = ' '.join([d.page_content for d in docs])
        suspicion_score = chunk_scanner(context)
        prompt = f"""Use the context to answer the question.
        
        {context}
        
        {q}
        
        Answer:
        """
        # Use pipeline for generation instead of manual tokenization
        outputs = self.llm(
            prompt,
            max_new_tokens=512,
            num_return_sequences=1,
            do_sample=False,
            truncation=True,
            return_full_text=False
        )
        return outputs[0]['generated_text'], suspicion_score