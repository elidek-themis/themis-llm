def process_results(doc, results):
    results, _ = zip(*results)  # remove is_greedy
    no_choices = int(len(results) // 2)

    return {
        "democratic": dict(zip(doc["choices"]["pro"], results[:no_choices])),
        "republican": dict(zip(doc["choices"]["contra"], results[no_choices:])),
    }
