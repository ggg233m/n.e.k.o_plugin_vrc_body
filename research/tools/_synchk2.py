import ast
root = r'H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body'
p = root + '\\tests\\test_policy_body_nudge.py'
src = open(p, encoding='utf-8').read()
tree = ast.parse(src)
print('SYNTAX OK, lines:', len(src.splitlines()))
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        print('class:', node.name, 'methods:', len([n for n in node.body if isinstance(n, ast.FunctionDef)]))