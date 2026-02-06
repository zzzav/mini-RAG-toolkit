from src.hello import main


def test_smoke(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello" in captured.out
