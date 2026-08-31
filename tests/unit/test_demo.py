from demo.run_ab import summarize


def test_summarize_by_thirds():
    rows = [{"score": s, "llm_calls": c, "steps": 2, "vetoes": 0, "vetoes_irreversible": 0, "worsens": 0,
             "abs_error_sum": 0.5, "error_count": 1, "episodes_written": 1, "habit_hits": 0,
             "dehabituations": 0, "tokens": 10, "resolved": True}
            for s, c in zip([10, 20, 30, 40, 50, 60], [3, 3, 3, 2, 2, 1])]
    s = summarize(rows)
    assert s["overall"]["score_mean"] == 35 and s["thirds"][0]["score_mean"] == 15
    assert s["thirds"][2]["llm_calls_mean"] == 1.5 and s["overall"]["mean_abs_error"] == 0.5
    assert s["overall"]["resolved_rate"] == 1.0
