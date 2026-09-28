from bank_marketing.plots import fmt_num, fmt_pct, fmt_times


def test_fmt_pct_uses_decimal_comma():
    assert fmt_pct(0.057, 1) == "5,7%"
    assert fmt_pct(0.1170, 0) == "12%"


def test_fmt_num_uses_dot_for_thousands_and_comma_for_decimals():
    assert fmt_num(124956) == "124.956"
    assert fmt_num(1234.5, 1) == "1.234,5"
    assert fmt_num(-8019) == "-8.019"


def test_fmt_times_formats_lifts():
    assert fmt_times(1.6054) == "1,61×"


def test_period_label_in_portuguese():
    from bank_marketing.plots import period_label

    assert period_label("2009-05") == "mai/09"
    assert period_label("2010-12") == "dez/10"
