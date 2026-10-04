import modal
import subprocess

app = modal.App("extract-cclr")
vol = modal.Volume.from_name("cclr-data")

@app.function(volumes={"/data": vol}, timeout=60 * 60)
def extract():
    subprocess.run(
        ["tar", "-xjf", "/data/ccrl-pgn.tar.bz2", "-C", "/data"],
        check=True,
    )
    vol.commit()  # persist writes

@app.local_entrypoint()
def main():
    extract.remote()