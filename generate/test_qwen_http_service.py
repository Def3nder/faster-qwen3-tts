"""Offline-Vertragstests mit simuliertem CLI-Generator; kein Modellimport."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from qwen_chunk_worker import generate_article
from http.server import ThreadingHTTPServer
from qwen_http_service import Jobs, handler, LEASE


class Contract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qwen-offline-")
        script = Path(self.temp.name) / "fake.py"
        script.write_text('''import sys, time, json
from pathlib import Path
print(json.dumps({"type": "ready", "seed": 123}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    text = Path(command["input"]).read_text(encoding="utf-8")
    if text == "wait": time.sleep(60)
    if text == "fail": sys.exit(1)
    Path(command["output"]).write_bytes(b"simulated-mp3")
    print(json.dumps({"type": "result", "id": command["id"], "seed": 124 if text == "drift" else 123}), flush=True)
''')
        self.jobs = Jobs(script, script, worker=script)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.jobs, "test-token"))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.jobs.close()
        self.temp.cleanup()

    def request(self, route, method="GET", data=None, token="test-token"):
        if data is not None and set(data) == {"text"}:
            data = {"markdown": data["text"]}
        req = urllib.request.Request(self.url + route, method=method,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()

    def wait(self, job_id):
        for _ in range(100):
            state = json.loads(self.request("jobs/" + job_id))
            if state["status"] != "running":
                return state
            time.sleep(.02)
        self.fail("Generator läuft zu lange")

    def test_success_auth_idempotence_and_download(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("health", token="wrong")
        self.assertEqual(error.exception.code, 401)
        health = json.loads(self.request("health"))
        self.assertEqual(health["protocol"], "webarchiv-qwen-v4")
        self.assertTrue(health["full_markdown"])
        self.assertEqual(health["audio_format"], "mp3")
        job_id = str(uuid.uuid4())
        for _ in range(2):
            self.request("jobs/" + job_id, "PUT", {"text": "Hallo"})
        self.assertEqual(self.wait(job_id)["status"], "succeeded")
        self.assertEqual(self.request(f"jobs/{job_id}/audio"), b"simulated-mp3")
        self.request("jobs/" + job_id, "DELETE")
        self.assertFalse(self.jobs.jobs[job_id]["directory"].exists())

    def test_cancel_busy_and_late_start(self):
        job_id = str(uuid.uuid4())
        self.request("jobs/" + job_id, "PUT", {"text": "wait"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("jobs/" + str(uuid.uuid4()), "PUT", {"text": "Hallo"})
        self.assertEqual(error.exception.code, 409)
        self.request("jobs/" + job_id, "DELETE")
        self.assertIsNotNone(self.jobs.jobs[job_id]["process"].poll())
        other = str(uuid.uuid4())
        self.request("jobs/" + other, "DELETE")
        with self.assertRaises(urllib.error.HTTPError):
            self.request("jobs/" + other, "PUT", {"text": "Hallo"})

    def test_lease_and_failed_generator(self):
        job_id = str(uuid.uuid4())
        self.request("jobs/" + job_id, "PUT", {"text": "wait"})
        self.jobs.jobs[job_id]["touched"] -= LEASE + 1
        self.jobs.sweep()
        self.assertEqual(self.jobs.status(job_id)["status"], "cancelled")
        failed = str(uuid.uuid4())
        self.request("jobs/" + failed, "PUT", {"text": "fail"})
        self.assertEqual(self.wait(failed)["status"], "failed")
        with self.assertRaises(urllib.error.HTTPError):
            self.request(f"jobs/{failed}/audio")

    def test_article_limit_rejected_before_process(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.request("jobs/" + str(uuid.uuid4()), "PUT", {"text": "x" * 1_000_001})
        self.assertEqual(len(self.jobs.jobs), 0)

    def test_original_markdown_bytes_preserved(self):
        job_id = str(uuid.uuid4())
        markdown = "\ufeff# Titel\r\nDatum: 2026-09-17\r\n**Text** 😀\r\n" * 200
        self.request("jobs/" + job_id, "PUT", {"markdown": markdown})
        self.assertEqual(self.wait(job_id)["status"], "succeeded")
        source = self.jobs.jobs[job_id]["directory"] / "input.md"
        self.assertEqual(source.read_bytes(), markdown.encode("utf-8"))
        self.assertEqual(self.jobs.jobs[job_id]["output"].suffix, ".mp3")


class OriginalScript(unittest.TestCase):
    def test_exact_cli_arguments_and_original_main(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script, config, source, output = [root / name for name in ["original.py", "config.json", "input.md", "output.mp3"]]
            config.write_text('{"clear_markdown": true, "seed_strategy": "increment", "mode": "semantic"}')
            markdown = "\ufeff# Original\r\n**Text**\r\n".encode("utf-8")
            source.write_bytes(markdown)
            script.write_text('''import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
for key in ["config", "input", "output"]: parser.add_argument("--"+key, required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
assert config["clear_markdown"] is True
assert config["seed_strategy"] == "increment"
assert config["mode"] == "semantic"
assert __name__ == "__main__"
Path(args.output).write_bytes(Path(args.input).read_bytes())
raise SystemExit(0)
''')
            generate_article(script, config, source, output)
            self.assertEqual(output.read_bytes(), markdown)

    def test_original_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "fail.py"
            script.write_text("raise SystemExit(2)")
            with self.assertRaisesRegex(RuntimeError, "Exit 2"):
                generate_article(script, Path(temp)/"config", Path(temp)/"input", Path(temp)/"output")


if __name__ == "__main__":
    unittest.main()
