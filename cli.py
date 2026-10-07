from rag_engine import RagEngine

engine = RagEngine()
print(f"Data folder: {engine.data_dir}")

ok, message = engine.check_ollama()
if not ok:
    print(message)
    raise SystemExit

engine.load()
print("Type 'reset' to start a new conversation, or 'quit' to exit.")

while True:
    question = input("\nAsk a question: ")

    if question.lower() == "quit":
        break
    if question.lower() == "reset":
        engine.reset_conversation()
        print("Conversation cleared.")
        continue

    print("\nThinking...")
    result = engine.ask(question)

    if result["search_query"] != question:
        print(f"(Searched for: {result['search_query']})")

    print(f"\nAnswer:\n{result['answer']}")
    print("\nRetrieved from:")
    for source in result["sources"]:
        print(
            f"  - {source['source']} | rerank #{source['rerank_rank']} "
            f"({source['rerank_score']:.2f}) | vector #{source['vector_rank']} "
            f"({source['vector_score']:.3f})"
        )