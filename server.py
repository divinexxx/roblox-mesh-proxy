"""Proxy that downloads Roblox mesh/texture assets from the CDN,
parses them, and returns JSON the game server can use to reconstruct
EditableMesh + EditableImage for PromptCreateAvatarAssetAsync."""

import struct, math, json, base64, io, gzip, os, re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.request

PORT = int(os.environ.get("PORT", 8090))

ROBLOSECURITY = os.environ.get("ROBLOSECURITY", "")

def download_asset(asset_id):
    url = f"https://assetdelivery.roblox.com/v1/asset/?id={asset_id}"
    headers = {"User-Agent": "Roblox/WinInet"}
    if ROBLOSECURITY:
        headers["Cookie"] = f".ROBLOSECURITY={ROBLOSECURITY}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    if len(data) < 8:
        raise ValueError("Empty asset response")
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    return data

def parse_mesh(data):
    lineend = data.find(b'\n')
    if lineend < 0 or lineend > 40:
        raise ValueError("missing version header")
    version = data[:lineend].replace(b'\r', b'').decode('ascii', 'replace')

    if version in ('version 1.00', 'version 1.01'):
        return parse_ascii_mesh(data, lineend + 1, version)

    start = lineend + 1

    if version in ('version 6.00', 'version 7.00'):
        return parse_chunked_mesh(data, start, version)

    if version == 'version 2.00':
        size, stride, face_stride, nv, nf = struct.unpack_from('<HBBII', data, start)
        p = start + size
    elif version in ('version 3.00', 'version 3.01'):
        size, stride, face_stride, lstride, nl, nv, nf = struct.unpack_from('<HBBHHII', data, start)
        p = start + size
    elif version in ('version 4.00', 'version 4.01', 'version 5.00'):
        size, lt, nv, nf, nl, nb, nn, ns, hq, pad = struct.unpack_from('<HHIIHHIHBB', data, start)
        stride = 40
        face_stride = 12
        if version == 'version 5.00':
            p = start + 32
        else:
            p = start + size
    else:
        raise ValueError(f"unsupported mesh version: {version}")

    if stride not in (36, 40):
        raise ValueError(f"unexpected vertex stride: {stride}")

    positions = []
    normals = []
    uvs = []
    for i in range(nv):
        off = p + i * stride
        px, py, pz = struct.unpack_from('<3f', data, off)
        nx, ny, nz = struct.unpack_from('<3f', data, off + 12)
        u, v = struct.unpack_from('<2f', data, off + 24)
        positions.append([px, py, pz])
        normals.append([nx, ny, nz])
        uvs.append([u, v])

    p += nv * stride
    if version in ('version 4.00', 'version 4.01', 'version 5.00'):
        if nb > 0:
            p += nv * 8

    faces = []
    for i in range(nf):
        a, b, c = struct.unpack_from('<III', data, p + i * 12)
        faces.append([a, b, c])

    return {
        "version": version,
        "vertex_count": nv,
        "face_count": nf,
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "faces": faces,
    }

def parse_ascii_mesh(data, start, version):
    text = data[start:].decode('ascii')
    count_line, body = text.split('\n', 1)
    nf = int(count_line.strip())
    vectors = re.findall(r'\[([^\[\]]+)\]', body)
    mult = 0.5 if version == 'version 1.00' else 1.0

    positions = []
    normals = []
    uvs = []
    faces = []
    vi = 0
    for fi in range(nf):
        base = fi * 9
        tri_verts = []
        for j in range(3):
            px, py, pz = [float(x) * mult for x in vectors[base + j*3].split(',')]
            nx, ny, nz = [float(x) for x in vectors[base + j*3 + 1].split(',')]
            u, v = float(vectors[base + j*3 + 2].split(',')[0]), float(vectors[base + j*3 + 2].split(',')[1])
            v = 1.0 - v
            positions.append([px, py, pz])
            normals.append([nx, ny, nz])
            uvs.append([u, v])
            tri_verts.append(vi)
            vi += 1
        faces.append(tri_verts)

    return {
        "version": version,
        "vertex_count": len(positions),
        "face_count": nf,
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "faces": faces,
    }

def compute_normals(positions, faces):
    normals = [[0.0, 0.0, 0.0] for _ in positions]
    for face in faces:
        a, b, c = face
        p0, p1, p2 = positions[a], positions[b], positions[c]
        e1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]]
        e2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]]
        nx = e1[1]*e2[2] - e1[2]*e2[1]
        ny = e1[2]*e2[0] - e1[0]*e2[2]
        nz = e1[0]*e2[1] - e1[1]*e2[0]
        for vi in (a, b, c):
            normals[vi][0] += nx
            normals[vi][1] += ny
            normals[vi][2] += nz
    for n in normals:
        l = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        if l > 1e-8:
            n[0] /= l; n[1] /= l; n[2] /= l
        else:
            n[1] = 1.0
    return normals

def parse_draco_mesh(draco_data, version):
    import DracoPy
    mesh = DracoPy.decode(draco_data)
    pts = mesh.points
    nv = pts.shape[0]
    positions = [[float(pts[i][0]), float(pts[i][1]), float(pts[i][2])] for i in range(nv)]
    fi = mesh.faces
    nf = fi.shape[0]
    faces = [[int(fi[i][0]), int(fi[i][1]), int(fi[i][2])] for i in range(nf)]
    if mesh.tex_coord is not None and len(mesh.tex_coord) > 0:
        tc = mesh.tex_coord
        uvs = [[float(tc[i][0]), float(tc[i][1])] for i in range(nv)]
    else:
        uvs = [[0.0, 0.0] for _ in range(nv)]
    if mesh.normals is not None and len(mesh.normals) > 0:
        nm = mesh.normals
        normals = [[float(nm[i][0]), float(nm[i][1]), float(nm[i][2])] for i in range(nv)]
    else:
        normals = compute_normals(positions, faces)
    return {
        "version": version,
        "vertex_count": nv,
        "face_count": nf,
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "faces": faces,
    }

def parse_chunked_mesh(data, start, version):
    p = start
    raw_core = None
    draco_payload = None
    while p < len(data):
        if len(data) - p < 16:
            break
        rawname, rev, length = struct.unpack_from('<8sII', data, p)
        p += 16
        name = rawname.rstrip(b'\0').decode('ascii', 'replace')
        buff = data[p:p + length]
        p += length
        if name == 'COREMESH' and rev == 1:
            raw_core = buff
        elif name == 'COREMESH' and rev == 2:
            n = struct.unpack_from('<I', buff, 0)[0]
            draco_payload = buff[4:]

    if draco_payload is not None:
        return parse_draco_mesh(draco_payload, version)

    if raw_core is None:
        raise ValueError("no COREMESH found")

    nv = struct.unpack_from('<I', raw_core, 0)[0]
    stride = 40
    voff = 4

    positions = []
    normals = []
    uvs = []
    for i in range(nv):
        off = voff + i * stride
        px, py, pz = struct.unpack_from('<3f', raw_core, off)
        nx, ny, nz = struct.unpack_from('<3f', raw_core, off + 12)
        u, v = struct.unpack_from('<2f', raw_core, off + 24)
        positions.append([px, py, pz])
        normals.append([nx, ny, nz])
        uvs.append([u, v])

    face_off = voff + nv * stride
    nf = struct.unpack_from('<I', raw_core, face_off)[0]
    face_off += 4

    faces = []
    for i in range(nf):
        a, b, c = struct.unpack_from('<III', raw_core, face_off + i * 12)
        faces.append([a, b, c])

    return {
        "version": version,
        "vertex_count": nv,
        "face_count": nf,
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "faces": faces,
    }

def parse_texture(data):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img = img.convert("RGBA")
    w, h = img.size
    raw = img.tobytes()
    return {
        "width": w,
        "height": h,
        "rgba_base64": base64.b64encode(raw).decode('ascii'),
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/mesh":
            asset_id = params.get("id", [None])[0]
            if not asset_id:
                self.send_error(400, "Missing ?id=")
                return
            try:
                raw = download_asset(int(asset_id))
                result = parse_mesh(raw)
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif path == "/texture":
            asset_id = params.get("id", [None])[0]
            if not asset_id:
                self.send_error(400, "Missing ?id=")
                return
            try:
                raw = download_asset(int(asset_id))
                result = parse_texture(raw)
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif path == "/health":
            self.send_json({"status": "ok"})

        else:
            self.send_error(404)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, separators=(',', ':'))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}" if args else "")

if __name__ == "__main__":
    print(f"[proxy] Starting on port {PORT}")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
