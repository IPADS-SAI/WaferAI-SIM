```bash
# Install dependencies.

# cd docs/EN
pip install -r requirements.txt

# Build the docs.
make clean
make html
sphinx-autobuild source build/html
# bind local port to sercer IP
ssh -L 8000:127.0.0.1:8000 root@server_ip

```