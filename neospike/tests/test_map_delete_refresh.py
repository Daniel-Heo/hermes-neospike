"""Map tab: a successful delete leaves the folder sidebar, the member list and the points at once,
without waiting for the /map reload; a failed delete keeps them; an older /map answer never
overwrites a newer one. Runs the dashboard bundle in node on a minimal hook runtime (no DOM)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from neospike.dashboard import plugin_api as api

NODE_TIMEOUT_SECONDS = 60
SCENARIOS = ("memory", "note", "failed", "folder", "folderWithMembers", "order")

# Function components get hook slots by tree path; a state change re-renders from the root,
# then runs the effects whose deps changed. Class components (the error boundary) render as-is.
_RUNTIME_JS = r"""
"use strict";
const path = require("node:path");

function createRuntime() {
  const slots = new Map();
  let visited = new Set();
  let pendingEffects = [];
  let dirty = false;
  let root = null;
  let tree = null;
  let current = null;

  function createElement(type, props) {
    const children = Array.prototype.slice.call(arguments, 2);
    const all = Object.assign({}, props || {});
    if (children.length) all.children = children.length === 1 ? children[0] : children;
    return { type: type, props: all, key: props && props.key !== undefined ? props.key : null };
  }
  class Component {
    constructor(props) { this.props = props; this.state = {}; }
    setState(next) { this.state = Object.assign({}, this.state, typeof next === "function" ? next(this.state) : next); dirty = true; }
  }
  function nextHook(make) {
    const s = current; const index = s.index++;
    if (s.hooks.length <= index) s.hooks.push(make());
    return s.hooks[index];
  }
  function sameDeps(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every(function (x, k) { return Object.is(x, b[k]); });
  }
  function useState(initial) {
    const hook = nextHook(function () { return { value: typeof initial === "function" ? initial() : initial }; });
    if (!hook.set) {
      hook.set = function (next) {
        const value = typeof next === "function" ? next(hook.value) : next;
        if (!Object.is(value, hook.value)) { hook.value = value; dirty = true; }
      };
    }
    return [hook.value, hook.set];
  }
  function useRef(initial) { return nextHook(function () { return { current: initial }; }); }
  function useMemo(make, deps) {
    const hook = nextHook(function () { return { deps: null, value: undefined, ready: false }; });
    if (!hook.ready || !deps || !sameDeps(deps, hook.deps)) { hook.value = make(); hook.deps = deps; hook.ready = true; }
    return hook.value;
  }
  function useCallback(fn, deps) { return useMemo(function () { return fn; }, deps); }
  function useEffect(fn, deps) {
    const hook = nextHook(function () { return { deps: null, ran: false, cleanup: null }; });
    if (!hook.ran || !deps || !sameDeps(deps, hook.deps)) {
      hook.deps = deps; hook.ran = true;
      pendingEffects.push(function () {
        if (hook.cleanup) hook.cleanup();
        const cleanup = fn();
        hook.cleanup = typeof cleanup === "function" ? cleanup : null;
      });
    }
  }
  function useContext() { return null; }
  function createContext() { return {}; }

  function childPath(at, child, index) {
    return at + "/" + (child && child.key !== null && child.key !== undefined ? "k" + child.key : index);
  }
  function renderNode(node, at) {
    if (node === null || node === undefined || node === false || node === true) return null;
    if (typeof node === "string" || typeof node === "number") return String(node);
    if (Array.isArray(node)) return node.map(function (child, index) { return renderNode(child, childPath(at, child, index)); });
    const type = node.type;
    if (typeof type === "string") return { type: type, props: node.props, children: renderNode(node.props.children, at + "/" + type) };
    const id = at + ":" + (type.name || "anon");
    visited.add(id);
    if (!slots.has(id)) slots.set(id, { hooks: [], instance: null, index: 0 });
    const slot = slots.get(id);
    if (type.prototype && type.prototype.render) {
      if (!slot.instance) slot.instance = new type(node.props);
      slot.instance.props = node.props;
      return renderNode(slot.instance.render(), id);
    }
    const saved = current; current = slot; slot.index = 0;
    let out;
    try { out = type(node.props); } finally { current = saved; }
    return renderNode(out, id);
  }
  function dropUnvisited(slot, id) {
    if (visited.has(id)) return;
    slot.hooks.forEach(function (hook) { if (hook.cleanup) hook.cleanup(); });
    slots.delete(id);
  }
  function renderAll() {
    for (let pass = 0; pass < 50; pass++) {
      dirty = false; visited = new Set(); pendingEffects = [];
      tree = renderNode(root, "r");
      slots.forEach(dropUnvisited);
      pendingEffects.forEach(function (run) { run(); });
      if (!dirty) return;
    }
    throw new Error("render loop");
  }
  return {
    React: { createElement: createElement, Component: Component, Fragment: "fragment" },
    hooks: { useState: useState, useEffect: useEffect, useCallback: useCallback, useMemo: useMemo, useRef: useRef,
             useContext: useContext, createContext: createContext },
    mount: function (element) { root = element; renderAll(); },
    rerender: function () { if (dirty) renderAll(); },
    tree: function () { return tree; }
  };
}

function textOf(node) {
  if (node === null || node === undefined) return "";
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(textOf).join("");
  return textOf(node.children);
}

function findAll(node, match, out) {
  out = out || [];
  if (node === null || node === undefined || typeof node === "string") return out;
  if (Array.isArray(node)) { node.forEach(function (child) { findAll(child, match, out); }); return out; }
  if (match(node)) out.push(node);
  findAll(node.children, match, out);
  return out;
}

const HOST_COMPONENTS = ["Card", "CardHeader", "CardTitle", "CardContent", "Badge", "Button", "Input", "Label", "Select",
                         "SelectOption", "TabsList", "TabsTrigger"];

function loadPage(scriptPath, fetchJSON) {
  const runtime = createRuntime();
  let page = null;
  const components = {};
  HOST_COMPONENTS.forEach(function (name) {
    components[name] = function (props) { return { type: name.toLowerCase(), props: props, key: null }; };
  });
  components.ConfirmDialog = function (props) { return { type: "confirm", props: props, key: null }; };
  components.Tabs = function (props) { return props.children("map", function () {}); };
  const classList = { contains: function () { return false; } };
  function noop() {}
  global.window = {
    __HERMES_PLUGIN_SDK__: { React: runtime.React, hooks: runtime.hooks, components: components, fetchJSON: fetchJSON },
    __HERMES_PLUGINS__: { register: function (name, component) { page = component; } },
    requestAnimationFrame: function () { return 0; }, cancelAnimationFrame: noop,
    setTimeout: function () { return 0; }, clearTimeout: noop,
    addEventListener: noop, removeEventListener: noop, devicePixelRatio: 1,
    performance: { now: function () { return 0; } },
    matchMedia: function () { return { matches: false, addEventListener: noop, removeEventListener: noop }; },
    localStorage: { getItem: function () { return null; }, setItem: noop }, console: console
  };
  global.document = { createElement: function () { return {}; }, documentElement: { classList: classList, dataset: {} },
                      body: { classList: classList } };
  require(path.resolve(scriptPath));
  runtime.mount(runtime.React.createElement(page, null));
  return runtime;
}

async function settle(runtime) {
  for (let k = 0; k < 20; k++) {
    await new Promise(function (resolve) { setImmediate(resolve); });
    runtime.rerender();
  }
}

module.exports = { loadPage: loadPage, settle: settle, findAll: findAll, textOf: textOf };
"""

# One scenario per run: argv = bundle path, scenario name. Prints "<name>: ok".
_SCENARIO_JS = r"""
"use strict";
const assert = require("node:assert/strict");
const H = require("./runtime.js");
const scriptPath = process.argv[2];
const scenario = process.argv[3];

function point(key, id, lane, kind, title, folderId) {
  return { key: key, id: id, lane: lane, kind: kind, title: title, folder_id: folderId, x: 1, y: 0, z: 0, updated_at: 100 - id,
           recent: true, superseded: false, neighbors: [], links: [], backlinks: 0 };
}
function folder(id, name, count) {
  return { id: id, name: name, color: "#56B4E9", origin: "human", count: count, x: null, y: null, z: null, recency: "old" };
}
const db = {
  points: [point("mem:1", 1, "memory", "memory", "alpha memory", 7), point("note:5", 5, "note", "human", "beta note", 7)],
  folders: [folder(7, "Garden", 2), folder(8, "Empty", 0)]
};
// hold: reloads after a delete never answer, so only the screen's own update can remove the item.
const control = { hold: false, failDelete: false, manualMap: false, mapQueue: [] };
function copy(value) { return JSON.parse(JSON.stringify(value)); }
function never() { return new Promise(function () {}); }
function mapAnswer() { return copy(db); }

function fetchJSON(url, init) {
  const method = (init && init.method) || "GET";
  const route = url.replace("/api/plugins/neospike", "").split("?")[0];
  if (method === "GET" && (route === "/map" || route === "/folders") && control.hold) return never();
  if (method === "GET" && route === "/map" && control.manualMap) {
    return new Promise(function (resolve) { control.mapQueue.push(resolve); });
  }
  if (method === "GET" && route === "/map") return Promise.resolve(mapAnswer());
  if (method === "GET" && route === "/folders") return Promise.resolve({ folders: copy(db.folders) });
  if (method === "GET" && route === "/memories/1") return Promise.resolve({ id: 1, text: "alpha memory", kind: "durable", editable: true });
  if (method === "GET" && route === "/notes/5") return Promise.resolve({ id: 5, rev: 1, title: "beta note", body: "x", tags: [] });
  if (method === "GET" && route.indexOf("/connections/") === 0) return Promise.resolve({ items: [], is_topic: false });
  if (method === "DELETE") {
    if (control.failDelete) return Promise.reject(new Error("500: {\"detail\": \"refused\"}"));
    if (route === "/memories/1") { db.points = db.points.filter(function (p) { return p.key !== "mem:1"; }); return Promise.resolve({ removed: true, id: 1 }); }
    if (route === "/notes/5") { db.points = db.points.filter(function (p) { return p.key !== "note:5"; }); return Promise.resolve({ id: 5 }); }
    if (route.indexOf("/folders/") === 0) {
      const id = Number(route.slice("/folders/".length));
      db.folders = db.folders.filter(function (f) { return f.id !== id; });
      db.points.forEach(function (p) { if (p.folder_id === id) p.folder_id = null; });
      return Promise.resolve({ id: id, deleted: true });
    }
  }
  return Promise.resolve({});
}

function memberTitles(rt) {
  return H.findAll(rt.tree(), function (n) { return n.type === "ul" && n.props.className === "neospike-member-list"; })
    .map(function (list) {
      return H.findAll(list, function (n) { return n.type === "span" && n.props.className === "neospike-choice-title"; }).map(H.textOf);
    });
}
function folderNames(rt) {
  return H.findAll(rt.tree(), function (n) { return n.type === "span" && n.props.className === "neospike-folder-name"; }).map(H.textOf);
}
function selectedFolders(rt) {
  return H.findAll(rt.tree(), function (n) { return n.type === "li" && / is-selected/.test(n.props.className || ""); }).length;
}
function unfiledCount(rt) {
  const rows = H.findAll(rt.tree(), function (n) { return n.type === "button" && n.props.className === "neospike-folder-button" && H.textOf(n).indexOf("Unfiled") === 0; });
  return H.textOf(rows[0]).slice("Unfiled".length);
}
function panelOpen(rt) {
  return H.findAll(rt.tree(), function (n) { return n.type === "textarea" && n.props.readOnly === true; }).length > 0;
}
function clickButton(rt, className, text) {
  const found = H.findAll(rt.tree(), function (n) { return n.type === "button" && n.props.className === className && H.textOf(n).indexOf(text) >= 0; });
  assert.ok(found.length > 0, "button " + text);
  found[0].props.onClick();
}
function clickText(rt, text) {
  const found = H.findAll(rt.tree(), function (n) { return n.type === "button" && H.textOf(n) === text; });
  assert.ok(found.length > 0, "button " + text);
  found[found.length - 1].props.onClick();
}
async function confirmLastDelete(rt) {
  const deletes = H.findAll(rt.tree(), function (n) { return n.type === "button" && n.props.destructive && H.textOf(n) === "Delete"; });
  deletes[deletes.length - 1].props.onClick();
  await H.settle(rt);
  const dialogs = H.findAll(rt.tree(), function (n) { return n.type === "confirm" && n.props.open; });
  assert.equal(dialogs.length, 1);
  dialogs[0].props.onConfirm();
  await H.settle(rt);
}
async function openGardenItem(rt, title) {
  clickButton(rt, "neospike-folder-button", "Garden");
  await H.settle(rt);
  assert.deepEqual(memberTitles(rt), [["alpha memory", "beta note"]]);
  clickButton(rt, "neospike-note-item", title);
  await H.settle(rt);
  assert.ok(panelOpen(rt));
}

const RUNS = {
  memory: async function (rt) {
    await openGardenItem(rt, "alpha memory");
    control.hold = true;
    await confirmLastDelete(rt);
    assert.deepEqual(memberTitles(rt), [["beta note"]]);
    assert.equal(panelOpen(rt), false);
  },
  note: async function (rt) {
    await openGardenItem(rt, "beta note");
    control.hold = true;
    await confirmLastDelete(rt);
    assert.deepEqual(memberTitles(rt), [["alpha memory"]]);
    assert.equal(panelOpen(rt), false);
  },
  failed: async function (rt) {
    await openGardenItem(rt, "alpha memory");
    control.failDelete = true;
    await confirmLastDelete(rt);
    assert.deepEqual(memberTitles(rt), [["alpha memory", "beta note"]]);
    assert.ok(panelOpen(rt));
  },
  folder: async function (rt) {
    clickButton(rt, "neospike-folder-button", "Empty");
    await H.settle(rt);
    assert.equal(selectedFolders(rt), 1);
    control.hold = true;
    await confirmLastDelete(rt);
    assert.deepEqual(folderNames(rt), ["Garden", "Unfiled"]);
    assert.equal(selectedFolders(rt), 0);
  },
  folderWithMembers: async function (rt) {
    clickButton(rt, "neospike-folder-button", "Garden");
    await H.settle(rt);
    assert.equal(unfiledCount(rt), "0");
    control.hold = true;
    await confirmLastDelete(rt);
    assert.deepEqual(folderNames(rt), ["Empty", "Unfiled"]);
    assert.equal(selectedFolders(rt), 0);
    assert.deepEqual(memberTitles(rt), []);
    assert.equal(unfiledCount(rt), "2");  // the trashed folder's items are unfiled now
  },
  order: async function (rt) {
    clickButton(rt, "neospike-folder-button", "Garden");
    await H.settle(rt);
    control.manualMap = true;
    const stale = mapAnswer();
    clickText(rt, "Refresh");
    clickText(rt, "Refresh");
    await H.settle(rt);
    assert.equal(control.mapQueue.length, 2);
    db.points = db.points.filter(function (p) { return p.key !== "mem:1"; });
    control.mapQueue[1](mapAnswer());
    await H.settle(rt);
    control.mapQueue[0](stale);  // the older request answers last
    await H.settle(rt);
    assert.deepEqual(memberTitles(rt), [["beta note"]]);
  }
};

(async function () {
  const rt = H.loadPage(scriptPath, fetchJSON);
  await H.settle(rt);
  assert.deepEqual(folderNames(rt), ["Garden", "Empty", "Unfiled"]);
  await RUNS[scenario](rt);
  console.log(scenario + ": ok");
})().catch(function (error) { console.error(error && error.stack ? error.stack : error); process.exit(1); });
"""


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return node


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("map_delete")
    (folder / "runtime.js").write_text(_RUNTIME_JS, encoding="utf-8")
    (folder / "scenario.js").write_text(_SCENARIO_JS, encoding="utf-8")
    return folder / "scenario.js"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_map_delete_updates_the_screen(runner: Path, scenario: str) -> None:
    script = Path(api.__file__).resolve().parent / "dist" / "index.js"
    checked = subprocess.run([_node(), str(runner), str(script), scenario], capture_output=True, text=True,
                             timeout=NODE_TIMEOUT_SECONDS)
    assert checked.returncode == 0, checked.stderr
    assert f"{scenario}: ok" in checked.stdout
