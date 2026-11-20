"""Use generated fixtures to check installed Office conversion and source safety."""

from pathlib import Path
import subprocess
import sys
import tempfile

from pilot_runtime import prepare_runtime


def main():
    root = prepare_runtime()
    from docx import Document
    from openpyxl import Workbook, load_workbook
    from office_convert import convert_snapshot, file_hash

    with tempfile.TemporaryDirectory(prefix="office-check-", dir=root) as directory:
        base = Path(directory)
        word = Document()
        word.add_paragraph("虚构PLC控制电源24 V DC，待批准。")
        word.save(base / "fixture.docx")
        book = Workbook()
        book.active.append(["对象", "电压", "单位"])
        book.active.append(["PLC-S7-1500", 24, "V DC"])
        book.save(base / "fixture.xlsx")
        book.close()
        for original, legacy in [("fixture.docx", "fixture.doc"), ("fixture.xlsx", "fixture.xls")]:
            job = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).parent / "office_convert.py"),
                 "--worker", str(base / original), str(base / legacy), str(base / "lease.json")],
                capture_output=True, timeout=100,
            )
            if job.returncode:
                raise RuntimeError(job.stderr.decode("utf-8", errors="replace"))
            before = file_hash(base / legacy)
            converted, receipt = convert_snapshot(base / legacy)
            assert file_hash(base / legacy) == before == receipt["source_sha256"]
            if converted.suffix == ".docx":
                assert "24 V DC，待批准" in "\n".join(p.text for p in Document(converted).paragraphs)
            else:
                loaded = load_workbook(converted, data_only=False)
                assert loaded.active["A2"].value == "PLC-S7-1500"
                assert loaded.active["B2"].value == 24
                assert loaded.active["C2"].value == "V DC"
                loaded.close()
            print(legacy + ": conversion, values and original hash passed", flush=True)


if __name__ == "__main__":
    main()
