from rag.evaluation import score_relevance


def test_relevant_context_scores_higher_than_irrelevant():
    question = "How do I reset my password?"
    relevant = ["To reset your password, go to Settings > Account > Reset Password."]
    irrelevant = ["The weather in Paris is usually mild in spring."]

    relevant_score = score_relevance(question, relevant)
    irrelevant_score = score_relevance(question, irrelevant)

    assert relevant_score > irrelevant_score


def test_empty_contexts_score_zero():
    assert score_relevance("any question", []) == 0.0
