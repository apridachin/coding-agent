import os

from coding_agent.execution import LocalExecutionEnvironment, ScopedExecutionEnvironment


def test_scoped_environment_writes_into_subdir(tmp_path):
    root = tmp_path
    subdir = root / "subdir"
    subdir.mkdir()

    inner = LocalExecutionEnvironment(str(root))
    scoped = ScopedExecutionEnvironment(inner, str(subdir))

    scoped.write_file("file.txt", "hello")

    assert os.path.exists(subdir / "file.txt")
    assert not os.path.exists(root / "file.txt")
