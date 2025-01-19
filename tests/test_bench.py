from banditbench import HardCoreBench

def test_calculate_cost():
    bench = HardCoreBench()
    model_to_cost = {}
    cost = bench.calculate_eval_cost([
        'gpt-3.5-turbo',
        'gemini/gemini-1.5-pro',
        'gemini/gemini-1.5-flash',
        'gpt-4o-2024-11-20',
        "gpt-4o-mini-2024-07-18",
        "o1-2024-12-17",
        "o1-mini-2024-09-12",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ])