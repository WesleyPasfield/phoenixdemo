from evaluation_system import LLMEvaluationSystem, Review

def main():
    system = LLMEvaluationSystem()
    
    # Example response to evaluate
    response = "The capital of France is London."
    
    # Manual review
    manual_review = Review(
        response_id="123",
        score=2,
        correction="The capital of France is Paris.",
        explanation="Incorrect capital city stated.",
        reviewer_id="user_1"
    )
    
    manual_result = system.add_manual_review(manual_review)
    print("Manual review stored:", manual_result)
    
    # LLM evaluation
    llm_results = system.evaluate_with_llms(
        response,
        models=["gpt-4", "claude-3"]
    )
    print("Policy compliance score:", llm_results["gpt-4"]["internal_policy"])
    
    # Get all results
    all_results = system.get_results()
    print("All evaluation results:", all_results)

if __name__ == "__main__":
    main()