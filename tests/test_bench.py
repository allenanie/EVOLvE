from banditbench import HardCoreBench, HardCorePlusBench, FullBench

def test_calculate_hardcore_bench_cost():
    bench = HardCoreBench()
    model_to_cost = {}
    cost = bench.calculate_eval_cost([
        'gpt-3.5-turbo',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gpt-4o-2024-11-20',
        "gpt-4o-mini-2024-07-18",
        "o1-2024-12-17",
        "o1-mini-2024-09-12",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ])

def test_calculate_hardcore_plus_bench_cost():
    bench = HardCorePlusBench()
    model_to_cost = {}
    cost = bench.calculate_eval_cost([
        'gpt-3.5-turbo',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gpt-4o-2024-11-20',
        "gpt-4o-mini-2024-07-18",
        "o1-2024-12-17",
        "o1-mini-2024-09-12",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ])

def test_calculate_full_bench_cost():
    bench = FullBench()
    model_to_cost = {}
    cost = bench.calculate_eval_cost([
        'gpt-3.5-turbo',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gpt-4o-2024-11-20',
        "gpt-4o-mini-2024-07-18",
        "o1-2024-12-17",
        "o1-mini-2024-09-12",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ])