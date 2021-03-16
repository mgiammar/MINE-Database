"""Tests for rules.py using pytest."""


import json
from pathlib import Path

from minedatabase import pickaxe
from minedatabase.rules import (
    BNICE,
    metacyc_generalized,
    metacyc_intermediate,
    metacyc_intermediate_uniprot,
)


pwd = Path(__file__)
pwd = pwd.parent
print(f"{pwd}/data/test_rules/rules_to_assert.json")
with open(f"{pwd}/data/test_rules/rules_to_assert.json", "r") as f:
    rule_assert_dict = json.load(f)


def test_metacyc_generalized_full():
    rule_list, correactant_list, rule_name = metacyc_generalized()
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_generalized"
    assert len(pk.operators) == 1221
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_generalized_rule0001"]
    )


def test_metacyc_generalized_specify_number():
    rule_list, correactant_list, rule_name = metacyc_generalized(n_rules=10)
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_generalized_10_rules"
    assert len(pk.operators) == 10
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_generalized_rule0001"]
    )


def test_metacyc_generalized_specify_fraction():
    rule_list, correactant_list, rule_name = metacyc_generalized(fraction_coverage=0.5)
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_generalized_0,5_fraction_coverage"
    assert len(pk.operators) == 20
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_generalized_rule0001"]
    )


def test_metacyc_intermediate():
    rule_list, correactant_list, rule_name = metacyc_intermediate()
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_intermediate"
    assert len(pk.operators) == 7325
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001_0167"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_intermediate_rule0001_0167"]
    )


def test_metacyc_intermediate_specify_number():
    rule_list, correactant_list, rule_name = metacyc_intermediate(n_rules=20)
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_intermediate_20_rules"
    assert len(pk.operators) == 1947
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001_0167"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_intermediate_rule0001_0167"]
    )


def test_metacyc_intermediate_specify_fraction():
    rule_list, correactant_list, rule_name = metacyc_intermediate(fraction_coverage=0.2)
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_intermediate_0,2_fraction_coverage"
    assert len(pk.operators) == 585
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0006_0028"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_intermediate_rule0006_0028"]
    )


def test_metacyc_intermediate_uniprot():
    rule_list, correactant_list, rule_name = metacyc_intermediate_uniprot()
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_intermediate_uniprot"
    assert len(pk.operators) == 3370
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001_0177"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_intermediate_uniprot_rule0001_0177"]
    )


def test_metacyc_intermediate_uniprot_specify_number():
    rule_list, correactant_list, rule_name = metacyc_intermediate_uniprot(n_rules=20)
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_intermediate_uniprot_20_rules"
    assert len(pk.operators) == 878
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0001_0177"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_intermediate_uniprot_rule0001_0177"]
    )


def test_metacyc_intermediate_uniprot_specify_fraction():
    rule_list, correactant_list, rule_name = metacyc_intermediate_uniprot(
        fraction_coverage=0.2
    )
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "Metacyc_intermediate_uniprot_0,2_fraction_coverage"
    assert len(pk.operators) == 280
    assert len(pk.coreactants) == 45

    assert (
        pk.operators["rule0006_005"][1]["SMARTS"]
        == rule_assert_dict["Metacyc_intermediate_uniprot_rule0006_005"]
    )


def test_BNICE():
    rule_list, correactant_list, rule_name = BNICE()
    pk = pickaxe.Pickaxe(rule_list=rule_list, coreactant_list=correactant_list)

    assert rule_name == "BNICE"
    assert len(pk.operators) == 250
    assert len(pk.coreactants) == 33

    assert pk.operators["1.1.1_01"][1]["SMARTS"] == rule_assert_dict["BNICE_1.1.1_01"]
