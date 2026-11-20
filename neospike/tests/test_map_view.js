// The map's pure view state (folding, expansion, colours). Run: node test_map_view.js
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");

const view = require(path.join(__dirname, "..", "dashboard", "dist", "index.js"));

const FOLDER_COLOR = "#D55E00";
const points = [
  { key: "mem:1", lane: "memory", kind: "memory", title: "TOPIC: looks like one", folder_id: null },
  { key: "note:1", lane: "note", kind: "human", title: "plain", folder_id: null },
  { key: "note:2", lane: "note", kind: "agent", title: "TOPIC: sphere map", folder_id: null },
  { key: "note:3", lane: "note", kind: "task", title: "TASK note", folder_id: null },
  { key: "mem:2", kind: "memory", folder_id: 7 },
  { key: "note:4", kind: "human", folder_id: 7 },
  { key: "mem:3", kind: "memory", folder_id: 9 }
];
const unfiled = [0, 1, 2, 3];
const folderColors = { 7: FOLDER_COLOR, 9: "#000000" };

function keysOf(indices) { return indices.map(function (index) { return points[index].key; }); }

// 1. Folded by default: no filed point is drawn, every unfiled point is.
assert.deepEqual(view.drawnIndices(points, {}), unfiled);

// 1. Expanding folder 7 adds exactly its members; folding again removes them.
const open7 = view.toggledExpansion({}, 7);
assert.deepEqual(keysOf(view.drawnIndices(points, open7)), ["mem:1", "note:1", "note:2", "note:3", "mem:2", "note:4"]);
const both = view.toggledExpansion(open7, 9);
assert.deepEqual(view.drawnIndices(points, both), [0, 1, 2, 3, 4, 5, 6]);
assert.deepEqual(view.drawnIndices(points, view.toggledExpansion(open7, 7)), unfiled);
assert.deepEqual(view.drawnFlags(points, open7), [true, true, true, true, true, true, false]);

// Unfiled colour by class: a doc (human note) orange, other notes green (a "TOPIC:" title is just a note now),
// memory and TASK blue.
const expected = { "mem:1": "#1E88E5", "note:1": "#FB8C00", "note:2": "#B9F6CA", "note:3": "#1E88E5" };
unfiled.forEach(function (index) {
  const color = view.pointColor(points[index], folderColors, false);
  assert.equal(color, expected[points[index].key]);
  assert.notEqual(color, "#9AA0A6");
});
// A note saved without a title takes the first non-empty body line (heading marks dropped, 200 characters).
assert.equal(view.noteTitle("  Typed  ", "body"), "Typed");
assert.equal(view.noteTitle("", "\n\n## First line\nsecond"), "First line");
assert.equal(view.noteTitle("   ", "x".repeat(300)).length, 200);
assert.equal(view.noteTitle("", "  \n \n"), "");
// A topic point (lane "topic") is red.
const topicPoint = { key: "topic:5", id: 5, lane: "topic", kind: "topic", title: "Deploy", folder_id: null };
assert.equal(view.isTopic(topicPoint), true);
assert.equal(view.isTopic(points[2]), false);
assert.equal(view.pointColor(topicPoint, folderColors, false), "#E53935");
assert.equal(view.pointColor(topicPoint, folderColors, true), "#E53935");
// Expanded members take their folder's colour (dark theme swaps black for white).
assert.equal(view.pointColor(points[4], folderColors, false), FOLDER_COLOR);
assert.equal(view.pointColor(points[6], folderColors, true), "#FFFFFF");

// Folder marker brightness by recency.
assert.equal(view.folderMarkerColor({ recency: "recent" }), "#F5C518");
assert.equal(view.folderMarkerColor({ recency: "mid" }), "#B08E11");
assert.equal(view.folderMarkerColor({ recency: "old" }), "#6E590B");

// On-demand drawing with injected frame and timer queues.
function FrameQueue() { this.callbacks = []; }
FrameQueue.prototype.request = function (callback) { this.callbacks.push(callback); return this.callbacks.length; };
FrameQueue.prototype.flush = function () { const due = this.callbacks; this.callbacks = []; due.forEach(function (callback) { callback(); }); };

function Painter() { this.sizes = []; }
Painter.prototype.paint = function (size) { this.sizes.push({ width: size.width, height: size.height, ratio: size.ratio }); };

const frames = new FrameQueue();
const timers = new FrameQueue();
const painter = new Painter();
const scheduler = view.createDrawScheduler(frames.request.bind(frames), painter.paint.bind(painter), timers.request.bind(timers));

// Size 0 (hidden tab): data arrives, nothing is scheduled or drawn, the draw stays owed.
scheduler.setSize(0, 0, 1);
assert.equal(scheduler.redraw(), false);
scheduler.invalidate();
assert.equal(frames.callbacks.length + timers.callbacks.length, 0);
assert.equal(painter.sizes.length, 0);
assert.equal(scheduler.isDirty(), true);

// A size appears: drawn at once, at that size, with no frame needed.
scheduler.setSize(800, 600, 2);
assert.deepEqual(painter.sizes, [{ width: 800, height: 600, ratio: 2 }]);
assert.equal(scheduler.isPending(), false);
assert.equal(frames.callbacks.length + timers.callbacks.length, 0);

// Bursts (drag, wheel, hover): coalesced into one draw; frame and timer both asked, only the first paints.
scheduler.invalidate(); scheduler.invalidate(); scheduler.invalidate();
assert.equal(frames.callbacks.length, 1);
assert.equal(timers.callbacks.length, 1);
frames.flush(); timers.flush();
assert.equal(painter.sizes.length, 2);
// Animation frames paused (background page): the timer alone still paints.
scheduler.invalidate();
timers.flush();
assert.equal(painter.sizes.length, 3);
frames.flush();
assert.equal(painter.sizes.length, 3);

// The same size again paints nothing; a hidden canvas drops a pending burst until a size returns.
scheduler.setSize(800, 600, 2);
assert.equal(painter.sizes.length, 3);
scheduler.invalidate();
scheduler.setSize(0, 600, 2);
frames.flush(); timers.flush();
assert.equal(painter.sizes.length, 3);
scheduler.setSize(640, 480, 1);
assert.equal(painter.sizes.length, 4);
assert.deepEqual(painter.sizes[3], { width: 640, height: 480, ratio: 1 });

// The component wiring, replayed without a DOM in both effect orders.
// FakePanel mirrors MapPanel: state lives in a "latest" ref written on render; the effect after each render calls
// onRender(); the ResizeObserver calls onResize(). Frames and timers are never flushed, so a draw that depends
// on them (as in the real browser) counts as no draw.
function FakePanel() {
  this.box = { width: 0, height: 0, ratio: 1 };
  this.latest = { points: [] };
  this.painted = [];
  this.frames = new FrameQueue();
  this.timers = new FrameQueue();
  this.binding = view.createCanvasBinding(this.readSize.bind(this), this.frames.request.bind(this.frames),
    this.timers.request.bind(this.timers), this.paint.bind(this));
}
FakePanel.prototype.readSize = function () { return { width: this.box.width, height: this.box.height, ratio: this.box.ratio }; };
FakePanel.prototype.paint = function (size) { this.painted.push({ width: size.width, count: this.latest.points.length }); };
FakePanel.prototype.render = function (points) { this.latest = { points: points }; this.binding.onRender(); };
FakePanel.prototype.layout = function (width, height) { this.box = { width: width, height: height, ratio: 1 }; this.binding.onResize(); };
FakePanel.prototype.lastCount = function () { return this.painted.length ? this.painted[this.painted.length - 1].count : -1; };

const items = new Array(137).fill(0).map(function (_, index) { return { key: "mem:" + index }; });

// Size first, data later: tab shown (render + resize), then /map arrives (render).
const sizeFirst = new FakePanel();
sizeFirst.render([]);
sizeFirst.layout(712, 478);
sizeFirst.render(items);
assert.ok(sizeFirst.painted.length >= 1, "size first: never drawn");
assert.equal(sizeFirst.lastCount(), 137);

// Data first, size later: /map arrives while the canvas is 0-sized, then the size arrives (resize only, no render).
const dataFirst = new FakePanel();
dataFirst.render([]);
dataFirst.render(items);
assert.equal(dataFirst.painted.length, 0);
dataFirst.layout(712, 478);
assert.ok(dataFirst.painted.length >= 1, "data first: never drawn");
assert.equal(dataFirst.lastCount(), 137);
assert.equal(dataFirst.painted[0].width, 712);

// Data first, size seen by the next render instead of the observer: still drawn with the newest data.
const renderSees = new FakePanel();
renderSees.render(items);
renderSees.box = { width: 712, height: 478, ratio: 1 };
renderSees.render(items.slice(0, 10));
assert.equal(renderSees.lastCount(), 10);
// No frame was needed in any order.
[sizeFirst, dataFirst, renderSees].forEach(function (panel) { assert.equal(panel.frames.callbacks.length, 0); });

// Labels stay inside the canvas.
const W = 400, H = 300, PAD = 4;
function inside(box, width, height) {
  return box.x >= PAD && box.y >= PAD && box.x + width <= W - PAD && box.y + height <= H - PAD;
}
const cases = [[200, 150], [395, 150], [398, 2], [2, 298], [0, 0], [W, H]];
cases.forEach(function (anchor) {
  const box = view.placeLabel(anchor[0], anchor[1], 60, 12, 16, W, H, PAD);
  assert.ok(inside(box, 60, 12), "label outside canvas for anchor " + anchor.join(",") + ": " + JSON.stringify(box));
});
// Room on the right: label sits right of the anchor; near the right edge it flips left of it.
assert.deepEqual(view.placeLabel(200, 150, 60, 12, 16, W, H, PAD), { x: 216, y: 144 });
assert.equal(view.placeLabel(395, 150, 60, 12, 16, W, H, PAD).x, 395 - 16 - 60);

// A selected folder shows its members without changing the stored expansion; Unfiled adds nothing.
const stored = {};
assert.deepEqual(keysOf(view.drawnIndices(points, view.expandedWithFocus(stored, 9))), ["mem:1", "note:1", "note:2", "note:3", "mem:3"]);
assert.deepEqual(stored, {});
assert.equal(view.expandedWithFocus(stored, "unfiled"), stored);
assert.equal(view.expandedWithFocus(stored, null), stored);
// Click vs drag by slop; dragging a selected point carries the selection, another point only itself.
assert.equal(view.isDragMove(10, 10, 13, 10, 4), false);
assert.equal(view.isDragMove(10, 10, 15, 10, 4), true);
assert.deepEqual(view.dragKeys(["mem:1", "note:2"], "note:2"), ["mem:1", "note:2"]);
assert.deepEqual(view.dragKeys(["mem:1", "note:2"], "note:3"), ["note:3"]);
assert.equal(view.dragLabel(1, "plain"), "plain");
assert.equal(view.dragLabel(3, "plain"), "3 items");
assert.equal(view.dropFolderId("unfiled"), null);
assert.equal(view.dropFolderId("7"), 7);
assert.equal(view.moveMessage(2, "Work"), "Moved 2 item(s) to Work");

// Titles only for front-side points inside the visible canvas.
assert.equal(view.titleVisible(50, 50, 0.2, 400, 300), true);
assert.equal(view.titleVisible(50, 50, -0.01, 400, 300), false);
assert.equal(view.titleVisible(-1, 50, 0.5, 400, 300), false);
assert.equal(view.titleVisible(401, 50, 0.5, 400, 300), false);
assert.equal(view.titleVisible(50, 301, 0.5, 400, 300), false);
assert.equal(view.titleVisible(50, -3, 0.5, 400, 300), false);

// Checked keys of an emptied folder drop out; order of the kept keys is preserved.
assert.deepEqual(view.keptKeys(["mem:2", "note:4"], []), []);
assert.deepEqual(view.keptKeys(["note:4", "mem:9", "mem:2"], [points[4], points[5]]), ["note:4", "mem:2"]);

// A drawn member beats only its own folder's marker; another folder's marker or a point does not yield.
const marker7 = { type: "folder", folder: { id: 7 }, z: 3 };
assert.equal(view.memberBeatsMarker(marker7, points[4]), true);
assert.equal(view.memberBeatsMarker(marker7, points[6]), false);
assert.equal(view.memberBeatsMarker(marker7, points[0]), false);
assert.equal(view.memberBeatsMarker({ type: "point", index: 4, z: 1 }, points[4]), false);
assert.equal(view.memberBeatsMarker(null, points[4]), false);

// Nearest centre wins the click; a tie goes to the point in front.
assert.equal(view.closerHit(null, 9, -1), true);
assert.equal(view.closerHit({ distance: 3, z: 0.9 }, 1, -0.5), true);   // small memory dot near the centre beats the big topic
assert.equal(view.closerHit({ distance: 1, z: -0.5 }, 3, 0.9), false);
assert.equal(view.closerHit({ distance: 2, z: -0.5 }, 2, 0.1), true);
assert.equal(view.closerHit({ distance: 2, z: 0.5 }, 2, 0.1), false);

// Topic drop keys (topic ids) round-trip; folder targets are not topics.
assert.equal(view.dropTopicKey("12"), "topic:12");
assert.equal(view.topicIdOf("topic:12"), 12);
assert.equal(view.topicIdOf("7"), null);
assert.equal(view.topicIdOf(null), null);

// Header line, then the DB text unchanged (newlines, length and [[ kept).
const raw = "line one\n  line [[two]]\n" + "x".repeat(500);
assert.equal(view.connectedText([{ key: "mem:5", text: raw }, { key: "note:9", text: "# T\nbody" }]),
  "- [[mem:5]]\n" + raw + "\n- [[note:9]]\n# T\nbody");
assert.equal(view.connectedText([]), "");

// The UI text tables. Korean has every English key (and no extra one) with the same {placeholders}.
function placeholders(text) { return (String(text).match(/\{\w+\}/g) || []).sort().join(","); }
function missingFrom(table) { return function (key) { return !Object.prototype.hasOwnProperty.call(table, key); }; }
const enKeys = Object.keys(view.STRINGS.en);
const koKeys = Object.keys(view.STRINGS.ko);
assert.deepEqual(enKeys.filter(missingFrom(view.STRINGS.ko)), [], "keys missing from ko");
assert.deepEqual(koKeys.filter(missingFrom(view.STRINGS.en)), [], "ko keys not in en");
enKeys.forEach(function (key) {
  assert.equal(placeholders(view.STRINGS.ko[key]), placeholders(view.STRINGS.en[key]), "placeholders differ: " + key);
  assert.ok(String(view.STRINGS.ko[key]).trim().length > 0, "empty ko text: " + key);
});

// Locale rule: "ko…" is Korean, anything else (or nothing) is English.
assert.equal(view.languageOf("ko"), "ko");
assert.equal(view.languageOf("ko-KR"), "ko");
assert.equal(view.languageOf("KO"), "ko");
assert.equal(view.languageOf("en"), "en");
assert.equal(view.languageOf("ja"), "en");
assert.equal(view.languageOf(undefined), "en");
assert.equal(view.languageOf(null), "en");
assert.equal(view.languageOf(42), "en");
assert.equal(view.languageOf({ locale: "ko" }), "en");

// Numbers fill the text; the language switch changes the same calls' output.
view.setLanguage("ko");
assert.equal(view.getLanguage(), "ko");
assert.equal(view.moveMessage(2, "Work"), "2개 항목을 Work(으)로 옮겼습니다");
assert.equal(view.dragLabel(3, "plain"), "3개 항목");
assert.equal(view.mapStatusText(true, "boom", 0), "지도를 불러오지 못했습니다: boom");
assert.equal(view.t("memory.split", { n: 3, ids: "mem:1, mem:2, mem:3" }), "기억 3개로 나눴습니다: mem:1, mem:2, mem:3");
assert.equal(view.t("class.topic"), "토픽");
assert.equal(view.t("class.doc"), "문서");
assert.equal(view.t("no.such.key"), "no.such.key");
view.setLanguage("fr");
assert.equal(view.getLanguage(), "en");
assert.equal(view.moveMessage(2, "Work"), "Moved 2 item(s) to Work");
assert.equal(view.t("memory.split", { n: 3, ids: "x" }), "Split into 3 memories: x");

// Setup step 5 example line, command split out for <code>.
assert.equal(view.STRINGS.en["setup.step5Example"], "e.g. hermes gateway restart");
assert.equal(view.STRINGS.ko["setup.step5Example"], "예) hermes gateway restart");
assert.deepEqual(view.codeParts(view.STRINGS.en["setup.step5Example"], "hermes gateway restart"), ["e.g. ", "hermes gateway restart", ""]);
assert.deepEqual(view.codeParts(view.STRINGS.ko["setup.step5Example"], "hermes gateway restart"), ["예) ", "hermes gateway restart", ""]);
assert.deepEqual(view.codeParts("no command here", "hermes gateway restart"), ["no command here", "", ""]);

// Export to folder. Checked memory keys are skipped; the button text carries the count in both languages.
assert.deepEqual(view.exportNoteIds(["note:3", "mem:4", "note:12", "bogus"]), [3, 12]);
assert.equal(view.t("notes.export", { n: 5 }), "Export to folder… (5)");
assert.equal(view.t("notes.exported", { n: 2, folder: "vault" }), "Exported 2 notes to vault");
view.setLanguage("ko");
assert.equal(view.t("notes.exported", { n: 2, folder: "vault" }), "노트 2개를 vault에 내보냈습니다");
view.setLanguage("en");
["../x.md", "/abs.md", "C:/x.md", "a\\b.md", "a//b.md", "./a.md", ""].forEach(function (bad) {
  assert.equal(view.unsafeExportPath(bad), true, "unsafe: " + bad);
});
assert.equal(view.unsafeExportPath("topics/TOPIC_ tree.md"), false);

// Fake File System Access directory handle: a tree of { dirs, files }.
function FakeDir(name) { this.name = name; this.dirs = {}; this.files = {}; }
FakeDir.prototype.getDirectoryHandle = function (name, options) {
  if (!this.dirs[name]) {
    if (!options || !options.create) return Promise.reject(new Error("NotFoundError"));
    this.dirs[name] = new FakeDir(name);
  }
  return Promise.resolve(this.dirs[name]);
};
FakeDir.prototype.getFileHandle = function (name, options) {
  const dir = this;
  if (!(name in dir.files) && !(options && options.create)) return Promise.reject(new Error("NotFoundError"));
  return Promise.resolve({
    createWritable: function () {
      let buffer = "";  // a new writable starts empty, like keepExistingData: false
      return Promise.resolve({
        write: function (data) { buffer += data; return Promise.resolve(); },
        close: function () { dir.files[name] = buffer; return Promise.resolve(); }
      });
    }
  });
};

const root = new FakeDir("vault");
root.files["keep.txt"] = "untouched";
root.files["Plan.md"] = "old long content that must be replaced";
const exportFiles = [
  { path: "Plan.md", content: "new" },
  { path: "topics/TOPIC_ tree.md", content: "---\nid: 7\n---\ntree" },
  { path: "Work/a_b.md", content: "slash" },
  { path: "Work/a_b (note-9).md", content: "slash twin" }
];
const fsChecks = view.writeExportFiles(root, exportFiles).then(function (count) {
  assert.equal(count, 4);
  assert.equal(root.files["keep.txt"], "untouched");
  assert.equal(root.files["Plan.md"], "new");
  assert.equal(root.dirs.topics.files["TOPIC_ tree.md"], "---\nid: 7\n---\ntree");
  assert.deepEqual(Object.keys(root.dirs.Work.files).sort(), ["a_b (note-9).md", "a_b.md"]);
  const untouched = new FakeDir("other");
  return view.writeExportFiles(untouched, [{ path: "ok.md", content: "x" }, { path: "../escape.md", content: "x" }])
    .then(function () { assert.fail("unsafe path was written"); },
      function (error) {
        assert.match(String(error.message), /Unsafe export path/);
        assert.deepEqual(Object.keys(untouched.files), [], "nothing is written when one path is unsafe");
      });
});

fsChecks.then(function () { console.log("test_map_view: ok"); }, function (error) { console.error(error); process.exitCode = 1; });
