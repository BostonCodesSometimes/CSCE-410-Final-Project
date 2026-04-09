# Implementation skeleton from ChatGPT

class LitePackEvaluator:

    def __init__(self, retriever, packer, generator, metrics):
        self.retriever = retriever
        self.packer = packer
        self.generator = generator
        self.metrics = metrics

    def evaluate(self, dataset, token_budget):

        all_results = []

        for sample in dataset:
            query = sample["question"]
            gold_docs = sample["relevant_docs"]
            reference_answer = sample["answer"]

            # ----------------------
            # 1. Retrieve
            # ----------------------
            candidates = self.retriever.retrieve(query)

            # ----------------------
            # 2. Token-aware packing
            # ----------------------
            context = self.packer.pack(
                candidates,
                budget=token_budget
            )

            tokens_used = self.metrics.token_count(context)

            # ----------------------
            # 3. Generate answer
            # ----------------------
            answer = self.generator.generate(query, context)

            # ----------------------
            # 4. Compute metrics
            # ----------------------
            recall = self.metrics.recall_at_k(
                candidates, gold_docs
            )

            ndcg = self.metrics.ndcg(
                candidates, gold_docs
            )

            quality = self.metrics.answer_similarity(
                answer, reference_answer
            )

            faithfulness = self.metrics.faithfulness(
                answer, context
            )

            qpt = quality / max(tokens_used, 1)

            all_results.append({
                "recall": recall,
                "ndcg": ndcg,
                "quality": quality,
                "faithfulness": faithfulness,
                "tokens": tokens_used,
                "qpt": qpt
            })

        return self.aggregate(all_results)

    def aggregate(self, results):
        import numpy as np

        return {
            key: np.mean([r[key] for r in results])
            for key in results[0]
        }