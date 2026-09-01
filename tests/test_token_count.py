from src.data.clean import token_count


class MappingTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}


class ListTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3]


def test_token_count_supports_batch_encoding_shape():
    assert token_count(MappingTokenizer(), [], generation_prompt=True) == 4


def test_token_count_supports_legacy_list_shape():
    assert token_count(ListTokenizer(), [], generation_prompt=False) == 3
