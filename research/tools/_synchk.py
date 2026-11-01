import ast
root = r'H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body'
for f in ['backend/process.py', 'backend/service.py']:
    p = root + '\\' + f.replace('/', '\\')
    try:
        ast.parse(open(p, encoding='utf-8').read())
        print(f, 'SYNTAX OK')
    except SyntaxError as e:
        print(f, 'SYNTAX FAIL', e)