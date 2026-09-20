"""Static check of the Shiny wiring in app.py.

Shiny reports a missing output binding or a misspelled input as a blank panel
rather than an error, and the server cannot be exercised without a browser.
This walks the AST instead and checks the three things that actually break:

  * every ui.output_*("id") has a render function of that name
  * every render function is placed somewhere in the UI
  * every input.x() and input["x"]() read has a matching ui.input_*("x")

    python3 app/wiringcheck.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "app.py"


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def collect(tree):
    outputs, inputs_declared, downloads = set(), set(), set()
    inputs_read, render_fns = set(), set()
    dynamic_inputs = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            first = _const_str(node.args[0]) if node.args else None
            if name.startswith("output_") and first:
                outputs.add(first)
            elif name.startswith("input_") and first:
                # ui.input_action_button is an input like any other.
                inputs_declared.add(first)
            elif name == "download_button" and first:
                downloads.add(first)
            # f-string ids, e.g. ui.input_slider(f"mult_{i}", ...)
            elif (name.startswith("input_") and node.args
                  and isinstance(node.args[0], ast.JoinedStr)):
                dynamic_inputs.append(_prefix(node.args[0]))

        # input.x() and input["x"]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "input":
            inputs_read.add(node.func.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "input":
            key = node.func.slice
            if isinstance(key, ast.JoinedStr):
                dynamic_inputs.append(_prefix(key))
            elif _const_str(key):
                inputs_read.add(_const_str(key))
        # @reactive.event(input.x)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "input":
            inputs_read.add(node.attr)

        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and \
                        isinstance(target.value, ast.Name) and target.value.id == "render":
                    render_fns.add(node.name)
    return outputs, inputs_declared, downloads, inputs_read, render_fns, dynamic_inputs


def _prefix(joined: ast.JoinedStr) -> str:
    parts = []
    for v in joined.values:
        if isinstance(v, ast.Constant):
            parts.append(str(v.value))
        else:
            break
    return "".join(parts)


def main() -> int:
    tree = ast.parse(APP.read_text())
    outputs, declared, downloads, read, renders, dynamic = collect(tree)

    problems = []

    missing_render = sorted(outputs - renders)
    if missing_render:
        problems.append(f"UI outputs with no render function: {missing_render}")

    unplaced = sorted(renders - outputs - downloads)
    if unplaced:
        problems.append(f"render functions never placed in the UI: {unplaced}")

    undeclared = sorted(
        i for i in read
        if i not in declared and not any(i.startswith(p) for p in dynamic if p))
    if undeclared:
        problems.append(f"inputs read but never declared: {undeclared}")

    unread = sorted(declared - read)

    print(f"UI outputs            {len(outputs)}")
    print(f"render functions      {len(renders)}")
    print(f"download buttons      {len(downloads)}")
    print(f"inputs declared       {len(declared)}"
          + (f" (+ dynamic prefixes {sorted(set(dynamic))})" if dynamic else ""))
    print(f"inputs read           {len(read)}")
    if unread:
        print(f"\nDeclared but never read (harmless, but check): {unread}")
    print()
    for p in problems:
        print("FAIL  " + p)
    if not problems:
        print("PASS  every output is rendered, every render is placed, "
              "every input read is declared")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
