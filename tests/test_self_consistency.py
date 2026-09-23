import pytest
from src.inference.self_consistency import select_by_result_vote


def candidate(result=None,success=True):
    return {'execution_success':success,'predicted_result':result}


def test_result_vote_prefers_majority():
    choices=[candidate([[1]]),candidate([[2]]),candidate([[2]])]
    assert select_by_result_vote(choices)==(1,2)


def test_result_vote_tie_prefers_greedy():
    choices=[candidate([[1]]),candidate([[2]])]
    assert select_by_result_vote(choices)==(0,1)


def test_result_vote_ignores_execution_failures():
    choices=[candidate(success=False),candidate([[2]])]
    assert select_by_result_vote(choices)==(1,1)
    assert select_by_result_vote([candidate(success=False)])==(0,0)


def test_result_vote_rejects_empty_input():
    with pytest.raises(ValueError): select_by_result_vote([])
