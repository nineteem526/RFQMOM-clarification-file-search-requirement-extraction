import copy
import json
import unittest

from verify_company_chat import SYNTHETIC_EVIDENCE, validate_synthetic


def valid():
    return {"requirements": [
        {"source_id": evidence["id"], "quote": evidence["text"],
         "status": evidence["approval"], "value": value, "unit": "kN/m²"}
        for evidence, value in zip(SYNTHETIC_EVIDENCE, ("3.0", "5.0"))
    ]}


class SyntheticResponseTests(unittest.TestCase):
    def test_exact_evidence_and_status_pass(self):
        validate_synthetic(json.dumps(valid(), ensure_ascii=False))

    def test_wrong_units_quotes_values_status_or_sources_fail(self):
        for field, value in (
            ("unit", "kN"), ("quote", "不存在的原文"), ("value", "50"),
            ("status", "已确认"), ("source_id", "not-found"),
        ):
            response = copy.deepcopy(valid())
            response["requirements"][1][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_synthetic(json.dumps(response))

    def test_missing_duplicate_or_non_json_fail(self):
        missing = valid()
        missing["requirements"].pop()
        duplicate = valid()
        duplicate["requirements"][1] = duplicate["requirements"][0]
        for text in (json.dumps(missing), json.dumps(duplicate), "```json\n{}\n```"):
            with self.assertRaises(ValueError):
                validate_synthetic(text)
