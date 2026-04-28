"""
Tests for evidence matching functionality.
"""
import pytest
from environment.grader import evidence_found, normalize_text, tokenize


class TestExactMatch:
    def test_exact_substring_match(self):
        assert evidence_found("fit_transform(X_all)",
                              "scaler.fit_transform(X_all) on line 5")

    def test_exact_match_at_start(self):
        assert evidence_found("scaler", "scaler.fit_transform(X_all)")

    def test_exact_match_at_end(self):
        assert evidence_found("X_all)", "scaler.fit_transform(X_all)")

    def test_exact_match_in_json(self):
        assert evidence_found('"shuffle": true',
                              '{"shuffle": true, "test_size": 0.2}')

    def test_exact_match_multiline(self):
        artifact = "line1\nscaler.fit_transform(X_all)\nline3"
        assert evidence_found("fit_transform(X_all)", artifact)


class TestNormalizedMatch:
    def test_whitespace_normalized_in_artifact(self):
        # Normalization helps when the artifact has extra whitespace
        assert evidence_found("fit_transform(X_all)",
                              "scaler.fit_transform(X_all)  ")  # trailing space in artifact

    def test_case_insensitive(self):
        assert evidence_found("FIT_TRANSFORM",
                              "scaler.fit_transform(X_all)")

    def test_trailing_whitespace(self):
        assert evidence_found("fit_transform(X_all)  ",
                              "scaler.fit_transform(X_all)")

    def test_newline_normalized(self):
        artifact = "scaler.\nfit_transform(X_all)"
        # After normalization: "scaler. fit_transform(x_all)"
        assert evidence_found("scaler. fit_transform(X_all)", artifact)


class TestTokenOverlap:
    def test_token_overlap_80_percent_passes(self):
        # 5 tokens, 4 match = 80%
        assert evidence_found("alpha beta gamma delta epsilon",
                              "alpha beta gamma delta DIFFERENT")

    def test_token_overlap_exact_80(self):
        # 5 tokens, 4 match = 80%
        assert evidence_found("one two three four five",
                              "one two three four OTHER")

    def test_token_overlap_above_80(self):
        # 4 tokens, 4 match = 100%
        assert evidence_found("one two three four",
                              "one two three four and more")

    def test_token_overlap_below_80_fails(self):
        # 5 tokens, 3 match = 60% - should fail
        assert not evidence_found("one two three four five",
                                  "one two three OTHER WORDS")

    def test_min_3_tokens_required(self):
        # With < 3 tokens, fallback to exact/normalized match only
        assert evidence_found("V1", "V1 violation detected")
        assert not evidence_found("XYZ", "V1 violation detected")


class TestFabricatedEvidence:
    def test_completely_fabricated_fails(self):
        assert not evidence_found(
            "this_text_xyz_does_not_exist_anywhere",
            "StandardScaler().fit_transform(X_train)"
        )

    def test_similar_but_wrong_fails(self):
        assert not evidence_found(
            "encoder.fit_transform(X_train)",
            "scaler.fit_transform(X_all)"
        )


class TestEdgeCases:
    def test_empty_quote_fails(self):
        assert not evidence_found("", "some content here")

    def test_empty_artifact_fails(self):
        assert not evidence_found("some quote", "")

    def test_both_empty_fails(self):
        assert not evidence_found("", "")

    def test_none_values_handled(self):
        assert not evidence_found(None, "content")
        assert not evidence_found("quote", None)

    def test_whitespace_only_quote_fails(self):
        assert not evidence_found("   ", "some content")

    def test_very_short_quote(self):
        assert evidence_found("V1", "Found V1 violation")
        assert not evidence_found("V9", "Found V1 violation")


class TestNormalizationHelpers:
    def test_normalize_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_normalize_strips_edges(self):
        assert normalize_text("  hello  ") == "hello"

    def test_normalize_handles_newlines(self):
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_normalize_lowercases(self):
        assert normalize_text("HELLO World") == "hello world"


class TestTokenization:
    def test_tokenize_extracts_words(self):
        tokens = tokenize("hello world foo bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_tokenize_handles_underscores(self):
        tokens = tokenize("foo_bar baz_qux")
        assert "foo_bar" in tokens
        assert "baz_qux" in tokens

    def test_tokenize_extracts_numbers(self):
        tokens = tokenize("value is 123 and 456")
        assert "123" in tokens
        assert "456" in tokens

    def test_tokenize_lowercases(self):
        tokens = tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_empty_string(self):
        assert tokenize("") == set()

    def test_tokenize_none(self):
        assert tokenize(None) == set()


class TestRealWorldScenarios:
    def test_v1_evidence_found(self):
        artifact = """
        # Preprocessing pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        X_train, X_test = train_test_split(X_scaled, test_size=0.2)
        """
        assert evidence_found("scaler.fit_transform(X_all)", artifact)
        assert evidence_found("fit_transform", artifact)

    def test_v5_evidence_found(self):
        artifact = '{"total_runs": 15, "runs": [{"run_id": 1}]}'
        assert evidence_found('"total_runs": 15', artifact)
        assert evidence_found("total_runs", artifact)

    def test_v6_evidence_found(self):
        artifact = '{"metrics_tracked": ["accuracy", "f1", "precision", "recall"]}'
        assert evidence_found("metrics_tracked", artifact)
        assert evidence_found('"metrics_tracked":', artifact)

    def test_cross_artifact_quote_fails(self):
        """Quote from artifact A should not match artifact B."""
        artifact_a = "scaler.fit_transform(X_all)"
        artifact_b = "train_test_split(X, test_size=0.2)"
        assert evidence_found("fit_transform(X_all)", artifact_a)
        assert not evidence_found("fit_transform(X_all)", artifact_b)
