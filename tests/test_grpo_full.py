import pytest
from src.grpo.full import best_candidate, check_predictions


def test_validation_selection_includes_sft_and_prefers_earlier_tie():
    initial={'step':0,'metrics':{'correct_count':150},'label':'sft'}
    tied={'step':25,'metrics':{'correct_count':150},'label':'grpo'}
    worse={'step':50,'metrics':{'correct_count':149},'label':'grpo'}
    assert best_candidate([worse,tied,initial])==initial
    assert best_candidate([worse,tied])==tied
    improved={'step':50,'metrics':{'correct_count':151},'label':'grpo'}
    assert best_candidate([initial,improved])==improved


def test_rejects_prediction_id_gaps_before_scoring():
    with pytest.raises(ValueError,match='IDs/order'):
        check_predictions([{'id':'b'}],[{'id':'a'}],{})
