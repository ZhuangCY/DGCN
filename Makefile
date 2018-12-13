install: init
	.venv/bin/pip install -r requirements.txt

init:
	virtualenv -p /usr/bin/python2 .venv
