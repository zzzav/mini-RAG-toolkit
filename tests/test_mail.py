from src.mail_report import build_report


def test_build_report_top_sender():
    mails = [
        {"from": "a", "subject": "", "snippet": "", "date": ""},
        {"from": "b", "subject": "", "snippet": "", "date": ""},
        {"from": "a", "subject": "", "snippet": "", "date": ""},
    ]
    report = build_report(mails)
    assert report["total"] == 3
    assert report["top_sender"]["from"] == "a"
    assert report["top_sender"]["count"] == 2
