import json, os, socket, sys
s = socket.socket(socket.AF_UNIX); s.connect(os.path.expanduser("~/.obsidian-cli.sock")); s.settimeout(30)
s.sendall((json.dumps({"argv": sys.argv[1:], "tty": False, "cwd": os.getcwd()}) + "\n").encode())
out = b""
try:
    while (c := s.recv(65536)): out += c
except socket.timeout: pass
print(out.decode())
