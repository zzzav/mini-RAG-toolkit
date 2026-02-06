from src.mail_report import build_report


def test_build_report_top_sender():
    mails = [
        {"from": "a", "subject": "", "snippet": "", "date": ""},
        {"from": "b", "subject": "", "snippet": "", "date": ""},
        {"from": "a", "subject": "", "snippet": "", "date": ""},
    ]
    report = build_report(mails, "", False)
    assert report["total"] == 3
    assert report["top_sender"]["from"] == "a"
    assert report["top_sender"]["count"] == 2


def test_domain():
    mails = [
        {"from": "a@c", "subject": "", "snippet": "", "date": ""},
        {"from": "b@c", "subject": "", "snippet": "", "date": ""},
        {"from": "a@d", "subject": "", "snippet": "", "date": ""},
    ]
    report = build_report(mails, "c", True)
    assert report["top_sender"]["from"].endswith("@c")
    assert report["by_sender"][0]["from"] <= report["by_sender"][1]["from"]
    assert report["total"] == 2
