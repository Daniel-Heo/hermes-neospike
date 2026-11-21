(function () {
  "use strict";
  // Pure map view state. node loads this file as a module to test it (tests/test_map_view.js).
  const MAP_VIEW = buildMapView();
  if (typeof module === "object" && module && module.exports) module.exports = MAP_VIEW;
  if (typeof window === "undefined") return;

  function buildMapView() {
    // The UI text follows the Hermes dashboard language (SDK.useI18n().locale),
    // not the Desktop display.language. A locale starting with "ko" is Korean; any other or missing locale is English.
    // Server error text and user data (note and memory text, titles) are shown as sent. "{name}" takes params.name.
    const DEFAULT_LANGUAGE = "en";
    const KOREAN_PREFIX = "ko";
    const PARAM_PATTERN = /\{(\w+)\}/g;
    const STRINGS = {
      en: {
        "tab.map": "Map", "tab.notes": "Notes", "tab.docs": "Docs", "tab.memory": "Memory",
        "tab.conversations": "Conversations", "tab.config": "Config", "tab.setup": "Setup",
        "tab.preferences": "Preferences",
        "pref.add": "Add", "pref.type": "Kind", "pref.typePreference": "Preference", "pref.typeConstraint": "Constraint",
        "pref.common": "Common", "pref.situational": "Situational", "pref.situation": "Situation",
        "pref.situationHint": "When it applies (empty = every conversation)", "pref.text": "Content",
        "pref.basis": "Basis", "pref.basisUser": "user said", "pref.basisInferred": "agent inferred",
        "pref.updatedAt": "Changed", "pref.empty": "No preferences or constraints.",
        "pref.loadFailed": "Could not load preferences: {error}", "pref.added": "Preference added.",
        "pref.updated": "Preference updated.", "pref.deleted": "Preference deleted.",
        "pref.deleteTitle": "Delete this preference?", "pref.pending": " · vector pending",
        "pref.changed": "This preference changed meanwhile; current text: {text}",
        "common.save": "Save", "common.cancel": "Cancel", "common.delete": "Delete", "common.edit": "Edit", "common.search": "Search",
        "common.refresh": "Refresh", "common.restore": "Restore", "common.revert": "Revert",
        "common.move": "Move", "common.close": "Close", "common.clear": "Clear", "common.loading": "Loading…",
        "common.none": "None.", "common.saved": "Saved.", "common.unfiled": "Unfiled", "common.folderPick": "Folder…",
        "common.moveToFolder": "Move to folder", "common.selectTitle": "Select {title}", "common.unknownError": "Unknown error",
        "common.confidence": "confidence {n}%", "common.revertedAt": " · reverted {time}", "common.deleted": "(deleted)",
        "common.revisions": "Revisions", "common.noEarlierRevisions": "No earlier revisions.", "common.retry": "Retry",
        "common.open": "open", "common.memoryRef": "memory #{id}",
        "map.dragOne": "1 item", "map.dragMany": "{n} items", "map.moved": "Moved {n} item(s) to {folder}",
        "map.loadFailed": "Could not load the map: {error}", "map.empty": "No memories or notes with embeddings yet.",
        "setup.loadFailed": "Could not load setup status: {error}", "setup.opDone": "Operation completed successfully.",
        "setup.starting": "Starting…", "setup.actionDone": "{label} completed.",
        "setup.act.download": "Download models", "setup.act.activate": "Activate Neospike", "setup.act.health": "Health check",
        "setup.checking": "Checking Neospike setup…",
        "setup.heroCopy": "Private, on-device text memory. Complete each step, then run the health check.",
        "setup.refreshStatus": "Refresh status", "setup.status": "Setup status", "setup.chk.hermes": "Hermes environment",
        "setup.found": "Found", "setup.notFound": "Not found", "setup.chk.deps": "Runtime dependencies",
        "setup.chk.textModel": "Text embedding model",
        "setup.chk.config": "Configuration", "setup.chk.provider": "Memory provider active",
        "setup.current": "Current: {name}",
        "setup.step2": "Choose and download models", "setup.modeText": "Text only",
        "setup.modeTextCopy": "BAAI bge-m3 (1024 dimensions, up to 2048 tokens) for messages, summaries, and files.",
        "setup.downloading": "Downloading…",
        "setup.downloadSelected": "Download selected models", "setup.downloadProgress": "Download in progress",
        "config.diagnostics.title": "Diagnostics settings",
        "setup.debug": "Append Neospike diagnostics to chat responses (developer mode)",
        "setup.debugCopy": "Off by default. Shows memory excerpts in this profile's chats; normal chat history may retain them. No extra model calls.",
        "config.extractLang": "Language of extracted memories", "config.extractLang.ko": "한국어", "config.extractLang.en": "English", "config.extractLangCopy": "The language Neospike uses when it extracts memories from conversations. Default: English.",
        "setup.step5": "Apply to memory handling and verify",
        "setup.step5Copy": "Activation updates memory.provider. Restart the gateway afterward (hermes gateway restart).",
        "setup.step5Example": "e.g. hermes gateway restart",
        "setup.ragActive": "Neospike active", "setup.runHealth": "Run health check",
        "config.criteria.title": "Memory creation rules", "config.criteria.default": "Default",
        "config.enabled.title": "Neospike on/off", "config.enabled.label": "Neospike is on",
        "config.enabled.copy": "Off pauses Neospike in this home without uninstalling it: no automatic recall, no conversation saving, no save follow-up, tool calls return an error and the daily job skips this home. Turn it back on to resume from the next turn; no restart is needed.",
        "config.enabled.stopped": "Paused. The embedding service was asked to stop.",
        "config.enabled.not_running": "Paused. The embedding service was not running.",
        "config.enabled.in_use": "Paused. The embedding service keeps running for: {homes}",
        "config.enabled.error": "Paused, but the embedding service could not be reached: {message}",
        "config.prefModel.title": "Preference and constraint cleanup model", "config.prefModel.auto": "Follow the chat default model (auto)",
        "config.prefModel.copy": "The model Neospike uses to merge and tidy saved preferences and constraints. Saved in the Hermes config.yaml (auxiliary.neospike_pref_reconcile), the same setting the Hermes auxiliary model screen shows. \"API\" marks a model reached with an API key.",
        "config.prefModel.current": "{name} (current)", "config.prefModel.listFailed": "Could not load the model list: {message}",
        "config.prefModel.unchanged": "Not changed.",
        "config.criteria.chars": "{n} / {max} characters",
        "config.criteria.editorLabel": "Memory creation rules (MEMORY_RULE.md)",
        "connected.rel.refers": "refers", "connected.rel.referencedBy": "referenced by", "connected.rel.supersedes": "supersedes",
        "connected.rel.supersededBy": "superseded by", "connected.rel.topicMember": "topic member",
        "connected.rel.nextPiece": "next piece", "connected.rel.previousPiece": "previous piece",
        "folder.moved": "Moved {n} item(s).", "folder.target": "Target folder",
        "note.savedRev": "Saved rev {rev}.",
        "note.conflict": "This note changed since you opened it (now rev {rev}). Overwrite or reload.",
        "note.reloaded": "Reloaded the current version.", "note.trashed": "Moved to trash.",
        "note.restored": "Restored rev {target} as rev {rev}.", "note.titlePlaceholder": "Title (blank: first line of the text)", "note.titleLabel": "Note title",
        "note.bodyLabel": "Note body", "note.blocked": "Blocked lines (secret or prompt-injection pattern): {lines}",
        "note.tagsPlaceholder": "Tags, comma separated", "note.tagsLabel": "Note tags", "note.create": "Create",
        "note.currentVersion": "Current version (rev {rev})", "note.overwrite": "Overwrite with my version",
        "note.reloadCurrent": "Reload current version", "note.trashTitle": "Move this note to the trash?",
        "note.trashCopy": "Links to it become broken links. Its history is kept.", "note.related": "Related",
        "note.backlinks": "Backlinks", "note.outgoing": "Outgoing links", "note.broken": "Broken links", "note.similar": "Similar notes",
        "note.openHint": "Open a note to see its links.",
        "notes.loadFailed": "Could not load notes: {error}", "notes.search": "Search notes", "notes.filterTag": "Filter by tag",
        "notes.empty": "No notes.",
        "docs.search": "Search docs", "docs.new": "New doc", "docs.empty": "No docs.",
        "notes.export": "Export to folder… ({n})", "notes.exported": "Exported {n} notes to {folder}",
        "notes.exportedZip": "Downloaded {n} notes as {file}", "notes.exportFailed": "Export failed: {error}",
        "history.loadFailed": "Could not load history: {error}",
        "history.supersedeReverted": "Supersede reverted; the replaced memory is active again.",
        "history.revisionRestored": "Revision {rev} restored; the previous text was kept as a revision.",
        "history.loading": "Loading history…", "history.replaces": "Replaces #{id}: {text}", "history.replacedBy": "Replaced by #{id}: {text}",
        "history.supersedes": "Supersedes", "history.noSupersedes": "No supersedes links.",
        "memory.split": "Split into {n} memories: {ids}", "memory.updated": "Memory item updated.",
        "memory.changed": "This item changed; reloaded. Current text: {text}",
        "memory.archived": "Memory item archived; prefetch leaves it out, search still shows it.",
        "memory.unarchived": "Memory item unarchived; prefetch can use it again.", "memory.deleted": "Memory item deleted.",
        "memory.textLabel": "Memory text", "memory.select": "Select memory #{id}", "memory.supersededBy": " · superseded by #{id}",
        "memory.archivedTag": " · archived", "memory.hideHistory": "Hide history", "memory.history": "History",
        "memory.unarchive": "Unarchive", "memory.archive": "Archive", "memory.deleteTitle": "Delete this memory item?",
        "memory.loadFailed": "Could not load memory: {error}", "memory.search": "Search memory", "memory.empty": "No memory items.",
        "conv.masked": "Masked when stored", "conv.loadFailed": "Could not load conversations: {error}",
        "conv.search": "Search conversations", "conv.searching": "Searching…", "conv.results": "Search results",
        "conv.noMatch": "No matching conversation excerpts.", "conv.sessions": "Sessions", "conv.messages": " · {n} messages",
        "conv.maskedCount": " · {n} masked", "conv.empty": "No conversations stored.", "conv.transcript": "Transcript",
        "conv.openHint": "Open a session to read its transcript (read-only).",
        "class.topic": "Topics", "class.doc": "Docs", "class.note": "Other notes", "class.other": "Memory, TASK",
        "kind.memory": "Memory", "kind.human": "Docs", "kind.agent": "Agent notes", "kind.task": "TASK",
        "kind.formula": "Auto notes", "kind.note": "Note", "kind.topic": "Topic",
        "folder.needName": "Type a folder name first.", "folder.created": "Folder created.", "folder.renamed": "Folder renamed.",
        "folder.recolored": "Folder colour changed.", "folder.trashed": "Folder moved to trash; its items are unfiled.",
        "folder.lastUsed": "Last used: {recency}", "folder.rename": "Rename", "folder.colour": "Folder colour",
        "folder.name": "Folder name", "folder.newName": "New folder name", "folder.new": "New folder",
        "folder.loadFailed": "Could not load folders: {error}", "folder.trashTitle": "Move this folder to the trash?",
        "folder.trashCopy": "Its items stay and become unfiled.", "folder.fallback": "folder {id}",
        "connected.none": "No connections.", "connected.textLabel": "Text of all connected items",
        "topic.moved": "Moved {n} item(s)", "topic.added": "Added {n} item(s)",
        "topic.skipped": "; skipped {n} (not an active memory or not a member)",
        "topic.moveTo": "Move to", "topic.moveToPick": "Move to…", "topic.remove": "Remove from topic", "topic.topics": "Topics",
        "topic.folders": "Folders",
        "connected.select": "Select connected {key}", "connected.notOnMap": " · not on map", "connected.head": "Connected ({n})",
        "connected.dropHint": " · drop memories here to add", "connected.selectAllLabel": "Select all connected",
        "connected.selectAll": "Select all",
        "member.expand": "Expand on map", "member.superseded": " · superseded", "member.empty": "No items.",
        "legend.superseded": "Superseded memory", "legend.recent": "≤ 7 days", "legend.mid": "≤ 14 days", "legend.old": "older",
        "legend.unfiled": "Unfiled · {label}", "legend.folderUsed": "Folder used {label}", "legend.open": "Legend ▾",
        "legend.closed": "Legend ▸",
        "map.selected": "{n} selected", "map.folderGone": "This folder no longer exists.", "map.filter": "Filter by title",
        "map.filterLabel": "Filter map by title", "map.recentOnly": "Recent only",
        "map.help": "{n} items · zoom {zoom}× · click: open, Ctrl/Shift+click: select, drag an item onto a folder: move, drag empty space: rotate, Shift+drag: pan, wheel or +/−: zoom",
        "map.canvasLabel": "Sphere map of memories and notes", "map.closePanel": "Close panel",
        "boundary.title": "This tab hit an error and could not be shown.", "boundary.retry": "Try again"
      },
      ko: {
        "tab.map": "지도", "tab.notes": "노트", "tab.docs": "문서", "tab.memory": "기억",
        "tab.conversations": "대화", "tab.config": "설정", "tab.setup": "설치",
        "tab.preferences": "취향",
        "pref.add": "추가", "pref.type": "종류", "pref.typePreference": "취향", "pref.typeConstraint": "제약",
        "pref.common": "공통", "pref.situational": "선택", "pref.situation": "상황",
        "pref.situationHint": "적용되는 상황 (비우면 모든 대화)", "pref.text": "내용",
        "pref.basis": "근거", "pref.basisUser": "사용자가 말함", "pref.basisInferred": "에이전트 추론",
        "pref.updatedAt": "수정 시각", "pref.empty": "취향이나 제약이 없습니다.",
        "pref.loadFailed": "취향을 불러오지 못했습니다: {error}", "pref.added": "취향을 추가했습니다.",
        "pref.updated": "취향을 고쳤습니다.", "pref.deleted": "취향을 삭제했습니다.",
        "pref.deleteTitle": "이 취향을 삭제할까요?", "pref.pending": " · 벡터 대기",
        "pref.changed": "그 사이 취향이 바뀌었습니다. 현재 내용: {text}",
        "common.save": "저장", "common.cancel": "취소", "common.delete": "삭제", "common.edit": "편집", "common.search": "검색",
        "common.refresh": "새로고침", "common.restore": "복원", "common.revert": "되돌리기",
        "common.move": "옮기기", "common.close": "닫기", "common.clear": "선택 해제", "common.loading": "불러오는 중…",
        "common.none": "없음.", "common.saved": "저장했습니다.", "common.unfiled": "미분류", "common.folderPick": "폴더…",
        "common.moveToFolder": "폴더로 옮기기", "common.selectTitle": "{title} 선택", "common.unknownError": "알 수 없는 오류",
        "common.confidence": "신뢰도 {n}%", "common.revertedAt": " · 되돌림 {time}", "common.deleted": "(삭제됨)",
        "common.revisions": "수정 기록", "common.noEarlierRevisions": "이전 수정본이 없습니다.", "common.retry": "다시 시도",
        "common.open": "진행 중", "common.memoryRef": "기억 #{id}",
        "map.dragOne": "1개 항목", "map.dragMany": "{n}개 항목", "map.moved": "{n}개 항목을 {folder}(으)로 옮겼습니다",
        "map.loadFailed": "지도를 불러오지 못했습니다: {error}", "map.empty": "아직 임베딩된 기억이나 노트가 없습니다.",
        "setup.loadFailed": "설치 상태를 불러오지 못했습니다: {error}", "setup.opDone": "작업을 마쳤습니다.",
        "setup.starting": "시작하는 중…", "setup.actionDone": "{label}: 완료했습니다.",
        "setup.act.download": "모델 내려받기", "setup.act.activate": "Neospike 활성화", "setup.act.health": "상태 점검",
        "setup.checking": "Neospike 설치 상태를 확인하는 중…",
        "setup.heroCopy": "이 기기 안에서만 쓰는 비공개 텍스트 기억입니다. 단계를 차례로 마친 뒤 상태 점검을 실행하세요.",
        "setup.refreshStatus": "상태 새로고침", "setup.status": "설치 상태", "setup.chk.hermes": "Hermes 환경",
        "setup.found": "찾음", "setup.notFound": "없음", "setup.chk.deps": "실행 의존성",
        "setup.chk.textModel": "텍스트 임베딩 모델",
        "setup.chk.config": "설정", "setup.chk.provider": "메모리 제공자 활성",
        "setup.current": "현재: {name}",
        "setup.step2": "모델 고르기와 내려받기", "setup.modeText": "텍스트만",
        "setup.modeTextCopy": "메시지, 요약, 파일에 BAAI bge-m3(1024차원, 최대 2048토큰)를 씁니다.",
        "setup.downloading": "내려받는 중…",
        "setup.downloadSelected": "고른 모델 내려받기", "setup.downloadProgress": "내려받는 중",
        "config.diagnostics.title": "진단 정보 설정",
        "setup.debug": "채팅 답변에 Neospike 진단 정보 붙이기 (개발모드)",
        "setup.debugCopy": "기본은 꺼져 있습니다. 켜면 이 프로필의 채팅에 기억 발췌가 보이고, 채팅 기록에 남을 수 있습니다. 모델 호출은 늘지 않습니다.",
        "config.extractLang": "기억 추출 결과 언어", "config.extractLang.ko": "한국어", "config.extractLang.en": "English", "config.extractLangCopy": "Neospike가 대화에서 기억을 추출할 때 쓰는 언어입니다. 기본은 영어입니다.",
        "setup.step5": "메모리 처리에 적용과 점검",
        "setup.step5Copy": "활성화하면 memory.provider가 바뀝니다. 그다음 게이트웨이를 다시 시작하세요(hermes gateway restart).",
        "setup.step5Example": "예) hermes gateway restart",
        "setup.ragActive": "Neospike 활성", "setup.runHealth": "상태 점검 실행",
        "config.criteria.title": "기억 생성 규칙", "config.criteria.default": "디폴트",
        "config.enabled.title": "Neospike 켜기/끄기", "config.enabled.label": "Neospike 켜짐",
        "config.enabled.copy": "끄면 제거하지 않고 이 홈에서 Neospike를 멈춥니다. 자동 회상, 대화 저장, 저장 후속 작업을 하지 않고, 도구 호출은 오류를 돌려주며, 일별 작업은 이 홈을 건너뜁니다. 다시 켜면 재시작 없이 다음 턴부터 동작합니다.",
        "config.enabled.stopped": "멈췄습니다. 임베딩 서비스에 종료를 요청했습니다.",
        "config.enabled.not_running": "멈췄습니다. 임베딩 서비스는 실행 중이 아니었습니다.",
        "config.enabled.in_use": "멈췄습니다. 임베딩 서비스는 다음 홈이 쓰고 있어 계속 실행됩니다: {homes}",
        "config.enabled.error": "멈췄지만 임베딩 서비스에 연결하지 못했습니다: {message}",
        "config.prefModel.title": "선호·제약 정리 모델", "config.prefModel.auto": "채팅 기본 모델 따라가기(auto)",
        "config.prefModel.copy": "Neospike가 저장된 선호·제약을 합치고 정리할 때 쓰는 모델입니다. Hermes config.yaml(auxiliary.neospike_pref_reconcile)에 저장되며, Hermes 보조 모델 화면과 같은 설정입니다. \"API\"는 API 키로 쓰는 모델입니다.",
        "config.prefModel.current": "{name} (현재)", "config.prefModel.listFailed": "모델 목록을 불러오지 못했습니다: {message}",
        "config.prefModel.unchanged": "변경하지 않았습니다.",
        "config.criteria.chars": "글자 수: {n} / {max}",
        "config.criteria.editorLabel": "기억 생성 규칙 (MEMORY_RULE.md)",
        "connected.rel.refers": "참조", "connected.rel.referencedBy": "참조됨", "connected.rel.supersedes": "대체함",
        "connected.rel.supersededBy": "대체됨", "connected.rel.topicMember": "주제 멤버",
        "connected.rel.nextPiece": "다음 조각", "connected.rel.previousPiece": "앞 조각",
        "folder.moved": "{n}개 항목을 옮겼습니다.", "folder.target": "대상 폴더",
        "note.savedRev": "rev {rev}로 저장했습니다.",
        "note.conflict": "연 뒤에 이 노트가 바뀌었습니다(현재 rev {rev}). 덮어쓰거나 다시 불러오세요.",
        "note.reloaded": "현재 버전을 다시 불러왔습니다.", "note.trashed": "휴지통으로 옮겼습니다.",
        "note.restored": "rev {target}을(를) rev {rev}로 복원했습니다.", "note.titlePlaceholder": "제목 (비우면 본문 첫 줄)", "note.titleLabel": "노트 제목",
        "note.bodyLabel": "노트 본문", "note.blocked": "막힌 줄(비밀 정보나 프롬프트 주입 패턴): {lines}",
        "note.tagsPlaceholder": "태그 (쉼표로 구분)", "note.tagsLabel": "노트 태그", "note.create": "만들기",
        "note.currentVersion": "현재 버전 (rev {rev})", "note.overwrite": "내 버전으로 덮어쓰기",
        "note.reloadCurrent": "현재 버전 다시 불러오기", "note.trashTitle": "이 노트를 휴지통으로 옮길까요?",
        "note.trashCopy": "이 노트로 가는 링크는 끊어진 링크가 됩니다. 수정 기록은 남습니다.", "note.related": "관련 항목",
        "note.backlinks": "들어오는 링크", "note.outgoing": "나가는 링크", "note.broken": "끊어진 링크", "note.similar": "비슷한 노트",
        "note.openHint": "노트를 열면 링크가 보입니다.",
        "notes.loadFailed": "노트를 불러오지 못했습니다: {error}", "notes.search": "노트 검색", "notes.filterTag": "태그로 거르기",
        "notes.empty": "노트가 없습니다.",
        "docs.search": "문서 검색", "docs.new": "새 문서", "docs.empty": "문서가 없습니다.",
        "notes.export": "폴더로 내보내기… ({n})", "notes.exported": "노트 {n}개를 {folder}에 내보냈습니다",
        "notes.exportedZip": "노트 {n}개를 {file}로 내려받았습니다", "notes.exportFailed": "내보내지 못했습니다: {error}",
        "history.loadFailed": "기록을 불러오지 못했습니다: {error}",
        "history.supersedeReverted": "대체를 되돌렸습니다. 대체됐던 기억이 다시 쓰입니다.",
        "history.revisionRestored": "수정본 {rev}을(를) 복원했습니다. 직전 글은 수정본으로 남겼습니다.",
        "history.loading": "기록을 불러오는 중…", "history.replaces": "#{id} 대체: {text}", "history.replacedBy": "#{id}(으)로 대체됨: {text}",
        "history.supersedes": "대체 관계", "history.noSupersedes": "대체 관계가 없습니다.",
        "memory.split": "기억 {n}개로 나눴습니다: {ids}", "memory.updated": "기억을 고쳤습니다.",
        "memory.changed": "이 항목이 바뀌어 다시 불러왔습니다. 현재 글: {text}",
        "memory.archived": "기억을 보관했습니다. 자동 불러오기에서는 빠지고 검색에는 나옵니다.",
        "memory.unarchived": "보관을 풀었습니다. 자동 불러오기에 다시 쓰입니다.", "memory.deleted": "기억을 삭제했습니다.",
        "memory.textLabel": "기억 내용", "memory.select": "기억 #{id} 선택", "memory.supersededBy": " · #{id}(으)로 대체됨",
        "memory.archivedTag": " · 보관됨", "memory.hideHistory": "기록 숨기기", "memory.history": "기록",
        "memory.unarchive": "보관 해제", "memory.archive": "보관", "memory.deleteTitle": "이 기억을 삭제할까요?",
        "memory.loadFailed": "기억을 불러오지 못했습니다: {error}", "memory.search": "기억 검색", "memory.empty": "기억이 없습니다.",
        "conv.masked": "저장할 때 가림", "conv.loadFailed": "대화를 불러오지 못했습니다: {error}",
        "conv.search": "대화 검색", "conv.searching": "검색 중…", "conv.results": "검색 결과",
        "conv.noMatch": "맞는 대화 발췌가 없습니다.", "conv.sessions": "세션", "conv.messages": " · 메시지 {n}개",
        "conv.maskedCount": " · {n}곳 가림", "conv.empty": "저장된 대화가 없습니다.", "conv.transcript": "대화 기록",
        "conv.openHint": "세션을 열면 대화 기록을 읽을 수 있습니다 (읽기 전용).",
        "class.topic": "토픽", "class.doc": "문서", "class.note": "기타 노트", "class.other": "기억, TASK",
        "kind.memory": "기억", "kind.human": "문서", "kind.agent": "에이전트 노트", "kind.task": "TASK",
        "kind.formula": "자동 노트", "kind.note": "노트", "kind.topic": "토픽",
        "folder.needName": "먼저 폴더 이름을 입력하세요.", "folder.created": "폴더를 만들었습니다.", "folder.renamed": "폴더 이름을 바꿨습니다.",
        "folder.recolored": "폴더 색을 바꿨습니다.", "folder.trashed": "폴더를 휴지통으로 옮겼습니다. 안의 항목은 미분류가 됩니다.",
        "folder.lastUsed": "마지막 사용: {recency}", "folder.rename": "이름 바꾸기", "folder.colour": "폴더 색",
        "folder.name": "폴더 이름", "folder.newName": "새 폴더 이름", "folder.new": "새 폴더",
        "folder.loadFailed": "폴더를 불러오지 못했습니다: {error}", "folder.trashTitle": "이 폴더를 휴지통으로 옮길까요?",
        "folder.trashCopy": "안의 항목은 남고 미분류가 됩니다.", "folder.fallback": "폴더 {id}",
        "connected.none": "연결된 항목이 없습니다.", "connected.textLabel": "연결된 모든 항목의 글",
        "topic.moved": "{n}개 항목을 옮겼습니다", "topic.added": "{n}개 항목을 추가했습니다",
        "topic.skipped": "; {n}개는 건너뜀 (활성 기억이 아니거나 구성원이 아님)",
        "topic.moveTo": "옮길 곳", "topic.moveToPick": "옮길 곳…", "topic.remove": "토픽에서 빼기", "topic.topics": "토픽",
        "topic.folders": "폴더",
        "connected.select": "연결 항목 {key} 선택", "connected.notOnMap": " · 지도에 없음", "connected.head": "연결 ({n})",
        "connected.dropHint": " · 기억을 여기에 끌어 놓으면 추가", "connected.selectAllLabel": "연결 항목 모두 선택",
        "connected.selectAll": "모두 선택",
        "member.expand": "지도에서 펼치기", "member.superseded": " · 대체됨", "member.empty": "항목이 없습니다.",
        "legend.superseded": "대체된 기억", "legend.recent": "7일 이내", "legend.mid": "14일 이내", "legend.old": "그보다 오래됨",
        "legend.unfiled": "미분류 · {label}", "legend.folderUsed": "폴더 사용: {label}", "legend.open": "범례 ▾",
        "legend.closed": "범례 ▸",
        "map.selected": "{n}개 선택", "map.folderGone": "이 폴더는 더 이상 없습니다.", "map.filter": "제목으로 거르기",
        "map.filterLabel": "지도를 제목으로 거르기", "map.recentOnly": "최근 것만",
        "map.help": "{n}개 · 확대 {zoom}× · 클릭: 열기, Ctrl/Shift+클릭: 선택, 항목을 폴더로 끌기: 옮기기, 빈 곳 끌기: 회전, Shift+끌기: 이동, 휠 또는 +/−: 확대·축소",
        "map.canvasLabel": "기억과 노트의 구 지도", "map.closePanel": "패널 닫기",
        "boundary.title": "이 탭에서 오류가 나서 화면을 보여 주지 못했습니다.", "boundary.retry": "다시 시도"
      }
    };
    let language = DEFAULT_LANGUAGE;

    function languageOf(locale) {
      return typeof locale === "string" && locale.toLowerCase().indexOf(KOREAN_PREFIX) === 0 ? "ko" : DEFAULT_LANGUAGE;
    }

    function setLanguage(next) { language = Object.prototype.hasOwnProperty.call(STRINGS, next) ? next : DEFAULT_LANGUAGE; }

    function getLanguage() { return language; }

    // A key missing from the current table falls back to English, then to the key itself.
    function t(key, params) {
      const table = STRINGS[language];
      const text = Object.prototype.hasOwnProperty.call(table, key) ? table[key] : STRINGS[DEFAULT_LANGUAGE][key];
      if (text === undefined) return key;
      if (!params) return text;
      return text.replace(PARAM_PATTERN, function (match, name) { return params[name] === undefined ? match : String(params[name]); });
    }

    // Topic points red, docs (notes the user wrote) orange, other notes green, everything else blue.
    const CLASS_COLORS = { topic: "#E53935", doc: "#FB8C00", note: "#B9F6CA", other: "#1E88E5" };
    // Formula = automatic notes (author "formula"); coloured like the other notes.
    const KIND_CLASSES = { memory: "other", human: "doc", agent: "note", formula: "note", task: "other" };
    const TOPIC_LANE = "topic";
    const RECENCY_COLORS = { recent: "#F5C518", mid: "#B08E11", old: "#6E590B" };
    // Dark theme: the last palette colour (black) is shown as white.
    const DARK_SWAPS = { "#000000": "#FFFFFF" };

    function shownColor(color, dark) {
      const upper = String(color || "").toUpperCase();
      return dark && DARK_SWAPS[upper] ? DARK_SWAPS[upper] : color;
    }

    // The title a note is saved under: the typed one, else the first non-empty body line without its # marks.
    const NOTE_TITLE_MAX = 200;
    function noteTitle(title, body) {
      const typed = String(title || "").trim();
      if (typed) return typed;
      const lines = String(body || "").split("\n");
      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index].replace(/^\s*#+\s*/, "").trim();
        if (line) return line.slice(0, NOTE_TITLE_MAX).trim();
      }
      return "";
    }

    function isInFolder(point) { return point.folder_id !== null && point.folder_id !== undefined; }

    // A topic is its own point (lane "topic"); a note's colour does not depend on its title.
    function isTopic(point) { return !!point && point.lane === TOPIC_LANE; }

    function colorClass(point) {
      if (isTopic(point)) return "topic";
      return KIND_CLASSES[point.kind] || "other";
    }

    function classColor(point) { return CLASS_COLORS[colorClass(point)]; }

    function folderMarkerColor(folder) { return RECENCY_COLORS[folder.recency] || RECENCY_COLORS.old; }

    function isExpanded(expanded, folderId) { return expanded[folderId] === true; }

    // Folded by default: a filed point is drawn only while its folder is expanded; every unfiled point is drawn.
    function drawnIndices(points, expanded) {
      const out = [];
      for (let index = 0; index < points.length; index += 1) {
        if (!isInFolder(points[index]) || isExpanded(expanded, points[index].folder_id)) out.push(index);
      }
      return out;
    }

    function drawnFlags(points, expanded) {
      const flags = new Array(points.length).fill(false);
      drawnIndices(points, expanded).forEach(function (index) { flags[index] = true; });
      return flags;
    }

    // Unfiled: colour by class (topic / note / other). Expanded member: its folder's colour.
    function pointColor(point, folderPalette, dark) {
      if (!isInFolder(point) || !folderPalette[point.folder_id]) return classColor(point);
      return shownColor(folderPalette[point.folder_id], dark);
    }

    function toggledExpansion(expanded, folderId) {
      const next = Object.assign({}, expanded);
      if (isExpanded(next, folderId)) delete next[folderId]; else next[folderId] = true;
      return next;
    }

    // A selected folder (a number id) shows its members as if expanded; Unfiled or none adds nothing.
    function expandedWithFocus(expanded, focus) {
      if (typeof focus !== "number" || isExpanded(expanded, focus)) return expanded;
      const next = Object.assign({}, expanded);
      next[focus] = true;
      return next;
    }

    // A press becomes a drag once it leaves the click slop (pixels from the press point).
    function isDragMove(startX, startY, x, y, slop) { return Math.hypot(x - startX, y - startY) > slop; }

    // Dragging a selected point drags the whole selection; dragging any other point drags only that point.
    function dragKeys(selection, key) { return selection.indexOf(key) >= 0 ? selection.slice() : [key]; }

    function dragLabel(count, title) { return count === 1 ? String(title || t("map.dragOne")) : t("map.dragMany", { n: count }); }

    // Drop target ids: "unfiled" or a folder id as a string. Returns the /folders/move folder_id.
    function dropFolderId(target) { return target === "unfiled" ? null : Number(target); }

    function moveMessage(moved, binName) { return t("map.moved", { n: moved, folder: binName }); }

    // Checked keys that are still among the shown points. A move from any path drops the rest,
    // so an emptied folder never keeps "Move to folder (N)".
    function keptKeys(keys, points) {
      const shown = {};
      points.forEach(function (point) { shown[point.key] = true; });
      return keys.filter(function (key) { return shown[key] === true; });
    }

    // The map payload without a deleted item, so the list and the sphere drop it before /map answers.
    function withoutPoint(data, key) {
      return Object.assign({}, data, { points: data.points.filter(function (point) { return point.key !== key; }) });
    }

    function foldersWithout(folders, folderId) { return folders.filter(function (folder) { return folder.id !== folderId; }); }

    // The map payload without a trashed folder; its items become unfiled, as the server releases them.
    function withoutFolder(data, folderId) {
      return Object.assign({}, data, {
        folders: foldersWithout(data.folders, folderId),
        points: data.points.map(function (point) { return point.folder_id === folderId ? Object.assign({}, point, { folder_id: null }) : point; })
      });
    }

    // A drawn member point beats its own folder marker. A one-member folder's marker sits exactly
    // on that member, so without this the last item could not be dragged out on the map.
    // The click goes to the point whose centre is nearest; at equal distance the one in front.
    function closerHit(best, distance, z) { return best === null || distance < best.distance || (distance === best.distance && z > best.z); }

    // A drag can end on a topic (its point on the sphere, or its panel's Connected list): "topic:<topic id>".
    const TOPIC_DROP_PREFIX = "topic:";
    function dropTopicKey(topicId) { return TOPIC_DROP_PREFIX + Number(topicId); }
    function topicIdOf(target) {
      return typeof target === "string" && target.indexOf(TOPIC_DROP_PREFIX) === 0 ? Number(target.slice(TOPIC_DROP_PREFIX.length)) : null;
    }

    // Every connected item as "- [[mem:N]]" / "- [[note:N]]", then its DB text as stored
    // from the next line (no unfolding, clipping or [[ changes).
    function connectedText(items) {
      return items.map(function (item) { return "- [[" + item.key + "]]\n" + (item.text === undefined || item.text === null ? "" : String(item.text)); }).join("\n");
    }

    function memberBeatsMarker(hit, point) { return hit !== null && hit.type === "folder" && hit.folder.id === point.folder_id; }

    // A point's title is drawn only for a front-side point (z >= 0) whose anchor is inside
    // the visible canvas. Without this, placeLabel clamps an off-screen point's title onto the canvas edge.
    function titleVisible(x, y, z, width, height) { return z >= 0 && x >= 0 && x <= width && y >= 0 && y <= height; }

    // The canvas line. A failed /map shows its error text, never the empty-map notice.
    function mapStatusText(loaded, failureMessage, pointCount) {
      if (!loaded) return null;
      if (failureMessage) return t("map.loadFailed", { error: failureMessage });
      return pointCount ? null : t("map.empty");
    }

    // Draw on demand. Nothing is scheduled while the view is still, so the frame count is 0.
    // - redraw(): state changed (data, filter, expansion, focus, size). Paints now; a 0-sized canvas defers until a size.
    // - setSize(): a new non-zero size paints now (a deferred draw is never lost).
    // - invalidate(): bursts (drag, wheel, hover, animation) coalesce into one frame. The frame is asked of both
    //   scheduleFrame and scheduleTimer, first one wins, so a page whose animation frames are paused still paints.
    // scheduleFrame(callback), scheduleTimer(callback) and paint(size) are injected (window APIs in the page, queues in tests).
    function createDrawScheduler(scheduleFrame, paint, scheduleTimer) {
      const size = { width: 0, height: 0, ratio: 1 };
      let pending = false;
      let dirty = true;

      function hasArea() { return size.width > 0 && size.height > 0; }

      function repaintImmediately() {
        pending = false;
        if (!hasArea()) return false;
        dirty = false;
        paint(size);
        return true;
      }

      // Both the frame and the timer land here; the later one finds nothing pending.
      function onFrame() { if (pending) repaintImmediately(); }

      function invalidate() {
        dirty = true;
        if (pending || !hasArea()) return;
        pending = true;
        scheduleFrame(onFrame);
        if (scheduleTimer) scheduleTimer(onFrame);
      }

      function redraw() {
        dirty = true;
        return repaintImmediately();
      }

      // Stores the size; true when it changed. Paints nothing.
      function resize(width, height, ratio) {
        const nextRatio = ratio > 0 ? ratio : 1;
        if (width === size.width && height === size.height && nextRatio === size.ratio) return false;
        size.width = width; size.height = height; size.ratio = nextRatio;
        return true;
      }

      function setSize(width, height, ratio) {
        if (resize(width, height, ratio) && hasArea()) redraw();
      }

      return { invalidate: invalidate, redraw: redraw, resize: resize, setSize: setSize,
               isPending: function () { return pending; }, isDirty: function () { return dirty; }, size: size };
    }

    // The component's wiring, kept out of React so node can replay effect orders.
    // readSize() -> { width, height, ratio } of the canvas box (0 while hidden). The component calls
    // onRender() from an effect after every render, onResize() from its ResizeObserver, onInteraction() from input.
    // paint(size) must read the newest state itself (a ref), never a value captured at creation.
    function createCanvasBinding(readSize, scheduleFrame, scheduleTimer, paint) {
      const scheduler = createDrawScheduler(scheduleFrame, paint, scheduleTimer);

      function onRender() {
        const box = readSize();
        scheduler.resize(box.width, box.height, box.ratio);
        scheduler.redraw();
      }

      function onResize() {
        const box = readSize();
        scheduler.setSize(box.width, box.height, box.ratio);
      }

      return { onRender: onRender, onResize: onResize, onInteraction: scheduler.invalidate, scheduler: scheduler };
    }

    // Label box placement. Prefers the right of the anchor, flips to its left when it would overflow,
    // then clamps into [pad, canvas - pad]. Returns the box's top-left corner.
    function placeLabel(anchorX, anchorY, boxWidth, boxHeight, offset, canvasWidth, canvasHeight, pad) {
      let x = anchorX + offset;
      if (x + boxWidth > canvasWidth - pad) x = anchorX - offset - boxWidth;
      x = Math.max(pad, Math.min(x, canvasWidth - pad - boxWidth));
      let y = anchorY - boxHeight / 2;
      y = Math.max(pad, Math.min(y, canvasHeight - pad - boxHeight));
      return { x: x, y: y };
    }

    // Split a translated line around a command so the command can be shown as <code>.
    // Returns [before, command, after]; a line without the command comes back whole as [text, "", ""].
    function codeParts(text, command) {
      const at = command ? text.indexOf(command) : -1;
      if (at < 0) return [text, "", ""];
      return [text.slice(0, at), command, text.slice(at + command.length)];
    }

    // Note ids among checked item keys ("note:12"); memory keys are skipped.
    function exportNoteIds(keys) {
      const ids = [];
      (keys || []).forEach(function (key) { const match = /^note:(\d+)$/.exec(String(key)); if (match) ids.push(Number(match[1])); });
      return ids;
    }

    // Paths are re-checked in the browser so a bad reply can never leave the chosen folder.
    function unsafeExportPath(path) {
      const text = String(path || "");
      return !text || text.indexOf("\\") >= 0 || text.indexOf(":") >= 0 ||
        text.split("/").some(function (part) { return part === "" || part === "." || part === ".."; });
    }

    // Write [{path, content}] into a File System Access directory handle, creating sub-folders.
    // A file of the same name is overwritten; other files in the folder are left alone. Resolves to the count written.
    function writeExportFiles(root, files) {
      const list = files || [];
      const folders = {};
      function folderOf(parts) {
        if (!parts.length) return Promise.resolve(root);
        const key = parts.join("/");
        if (!folders[key]) {
          folders[key] = folderOf(parts.slice(0, -1)).then(function (parent) {
            return parent.getDirectoryHandle(parts[parts.length - 1], { create: true });
          });
        }
        return folders[key];
      }
      function writeOne(file) {
        if (unsafeExportPath(file.path)) return Promise.reject(new Error("Unsafe export path: " + file.path));
        const parts = file.path.split("/");
        return folderOf(parts.slice(0, -1))
          .then(function (folder) { return folder.getFileHandle(parts[parts.length - 1], { create: true }); })
          .then(function (handle) { return handle.createWritable(); })
          .then(function (writable) { return writable.write(file.content).then(function () { return writable.close(); }); });
      }
      // Check every path before writing anything.
      for (let index = 0; index < list.length; index += 1) {
        if (unsafeExportPath(list[index].path)) return Promise.reject(new Error("Unsafe export path: " + list[index].path));
      }
      let chain = Promise.resolve();
      list.forEach(function (file) { chain = chain.then(function () { return writeOne(file); }); });
      return chain.then(function () { return list.length; });
    }

    return { CLASS_COLORS: CLASS_COLORS, RECENCY_COLORS: RECENCY_COLORS, shownColor: shownColor, colorClass: colorClass, classColor: classColor,
             isTopic: isTopic, TOPIC_LANE: TOPIC_LANE, noteTitle: noteTitle,
             folderMarkerColor: folderMarkerColor, drawnIndices: drawnIndices, drawnFlags: drawnFlags,
             pointColor: pointColor, toggledExpansion: toggledExpansion, isExpanded: isExpanded,
             mapStatusText: mapStatusText, createDrawScheduler: createDrawScheduler, createCanvasBinding: createCanvasBinding,
             placeLabel: placeLabel, expandedWithFocus: expandedWithFocus, isDragMove: isDragMove, dragKeys: dragKeys,
             dragLabel: dragLabel, dropFolderId: dropFolderId, moveMessage: moveMessage, titleVisible: titleVisible,
             keptKeys: keptKeys, withoutPoint: withoutPoint, foldersWithout: foldersWithout, withoutFolder: withoutFolder,
             memberBeatsMarker: memberBeatsMarker, closerHit: closerHit, dropTopicKey: dropTopicKey,
             topicIdOf: topicIdOf, connectedText: connectedText, STRINGS: STRINGS, languageOf: languageOf, setLanguage: setLanguage,
             getLanguage: getLanguage, t: t, codeParts: codeParts, exportNoteIds: exportNoteIds, unsafeExportPath: unsafeExportPath,
             writeExportFiles: writeExportFiles };
  }

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!(SDK && window.__HERMES_PLUGINS__)) return;
  const React = SDK.React;
  const h = React.createElement.bind(React);
  const t = MAP_VIEW.t;
  const hostHooks = SDK.hooks;
  const useState = hostHooks.useState, useEffect = hostHooks.useEffect;
  const useCallback = hostHooks.useCallback, useRef = hostHooks.useRef;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input, Label, Select, SelectOption, ConfirmDialog, Tabs, TabsList, TabsTrigger } = SDK.components;
  const API = "/api/plugins/neospike";

  function failureMessage(error) {
    const raw = String(error && error.message ? error.message : error || t("common.unknownError"));
    const statusPrefixed = /^\d{3}:\s*(.*)$/s.exec(raw);
    const inner = statusPrefixed ? statusPrefixed[1] : raw;
    try { return JSON.parse(inner).detail || raw; } catch (_) { return inner; }
  }

  function StatusRow(props) {
    const dotClass = "neospike-check-dot " + (props.ok ? "is-ok" : "is-pending");
    return h("li", { className: "neospike-check" },
      h("span", { className: dotClass, "aria-hidden": true }, props.ok ? "✓" : "·"),
      h("span", null, props.label),
      props.note ? h("span", { className: "neospike-check-note" }, props.note) : null
    );
  }

  function OptionRadio(props) {
    return h("label", { className: "neospike-radio-choice" + (props.checked ? " is-selected" : "") },
      h("input", { onChange: props.onChange, checked: props.checked, value: props.value, name: props.name, type: "radio" }),
      h("span", { className: "neospike-radio-content" },
        h("span", { className: "neospike-choice-title" }, props.title),
        props.copy ? h("span", { className: "neospike-choice-copy" }, props.copy) : null
      )
    );
  }

  function InstallView() {
    const [setupInfo, setSetupInfo] = useState(null);
    const [busyKey, setBusyKey] = useState(null);
    const [runJob, setRunJob] = useState(null);
    const [banner, setBanner] = useState(null);

    const reloadStatus = useCallback(function () {
      return SDK.fetchJSON(API + "/setup/status").then(function (data) {
        setSetupInfo(data);
        return data;
      }).catch(function (e) { setBanner({ ok: false, text: t("setup.loadFailed", { error: failureMessage(e) }) }); });
    }, []);

    useEffect(function () { reloadStatus(); }, [reloadStatus]);
    useEffect(function () {
      if (!runJob || runJob.state !== "running") return undefined;
      const poller = window.setInterval(function () {
        SDK.fetchJSON(API + "/setup/progress?job_id=" + encodeURIComponent(runJob.id)).then(function (latest) {
          setRunJob(latest);
          if (latest.state !== "running") {
            setBusyKey(null);
            setBanner({ ok: latest.state === "complete", text: latest.state === "complete" ? t("setup.opDone") : latest.detail });
            reloadStatus();
          }
        }).catch(function (e) { setBusyKey(null); setBanner({ ok: false, text: failureMessage(e) }); });
      }, 1500);
      return function () { window.clearInterval(poller); };
    }, [runJob && runJob.id, runJob && runJob.state, reloadStatus]);

    function action(actionKey, apiPath, fetchInit) {
      setBusyKey(actionKey); setBanner(null);
      return SDK.fetchJSON(API + apiPath, fetchInit || { method: "POST" }).then(function (result) {
        if (result.job_id) setRunJob({ id: result.job_id, state: "running", detail: t("setup.starting"), log: "" });
        else { setBusyKey(null); setBanner({ ok: true, text: t("setup.actionDone", { label: t(actionKey) }) }); reloadStatus(); }
      }).catch(function (e) { setBusyKey(null); setBanner({ ok: false, text: failureMessage(e) }); });
    }

    if (!setupInfo) return h("div", { className: "neospike-page" },
      h("p", { className: "text-sm text-muted-foreground" }, banner ? banner.text : t("setup.checking")),
      banner && h(Button, { variant: "outline", onClick: reloadStatus }, t("common.retry"))
    );
    const isDownloading = busyKey === "setup.act.download";
    const hasConfig = Boolean(setupInfo.config);

    return h("div", { className: "neospike-page" },
      h("div", { className: "neospike-hero" },
        h("div", null, h("h1", null, "Neospike"), h("p", null, t("setup.heroCopy"))),
        h(Button, { variant: "outline", onClick: reloadStatus, disabled: !!busyKey }, t("setup.refreshStatus"))
      ),
      banner ? h("div", { className: "neospike-message " + (banner.ok ? "is-ok" : "is-error"), role: "status" }, banner.text) : null,
      h("div", { className: "neospike-layout" },
        h(Card, { className: "neospike-status" },
          h(CardHeader, null, h(CardTitle, null, t("setup.status"))),
          h(CardContent, null,
            h("ul", null,
              h(StatusRow, { ok: setupInfo.hermes_found && setupInfo.python_found, label: t("setup.chk.hermes"), note: setupInfo.python_found ? t("setup.found") : t("setup.notFound") }),
              h(StatusRow, { ok: setupInfo.dependencies_installed, label: t("setup.chk.deps") }),
              h(StatusRow, { ok: setupInfo.text_model_downloaded, label: t("setup.chk.textModel") }),
              h(StatusRow, { ok: hasConfig, label: t("setup.chk.config") }),
              h(StatusRow, { ok: setupInfo.active, label: t("setup.chk.provider"), note: setupInfo.provider ? t("setup.current", { name: setupInfo.provider }) : null })
            )
          )
        ),
        h("div", { className: "neospike-steps" },
          h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "1"), h(CardTitle, null, t("setup.step2")))),
            h(CardContent, { className: "neospike-stack" },
              h("div", { className: "neospike-choices" },
                h(OptionRadio, { name: "neospike-model-mode", value: "text", checked: true, title: t("setup.modeText"), copy: t("setup.modeTextCopy"), onChange: function () {} })
              ),
              h(Button, { onClick: function () { action("setup.act.download", "/setup/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }); }, disabled: !!busyKey || setupInfo.text_model_downloaded }, isDownloading ? t("setup.downloading") : t("setup.downloadSelected")),
              runJob && runJob.kind === "download" ? h("div", { className: "neospike-progress" },
                h("div", { className: "neospike-progress-bar", "aria-label": t("setup.downloadProgress") }, h("span", { className: runJob.state === "running" ? "is-running" : "" })),
                h("p", null, runJob.detail), runJob.log ? h("pre", null, runJob.log) : null
              ) : null
            )
          ),
          h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "2"), h(CardTitle, null, t("setup.step5")))),
            h(CardContent, { className: "neospike-stack" },
              h("p", { className: "text-sm text-muted-foreground" }, t("setup.step5Copy")),
              h(CmdSample, { text: t("setup.step5Example"), command: GATEWAY_RESTART_COMMAND }),
              h("div", { className: "neospike-actions" },
                h(Button, { disabled: !!busyKey || setupInfo.active, onClick: function () { action("setup.act.activate", "/setup/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) }); } }, setupInfo.active ? t("setup.ragActive") : t("setup.act.activate")),
                h(Button, { variant: "outline", disabled: !!busyKey, onClick: function () { action("setup.act.health", "/health"); } }, t("setup.runHealth"))
              )
            )
          ),
          h(EnabledSwitchView, null)
        )
      )
    );
  }

  // ---- Config: operating settings; Setup stays for installation. The save criteria list is titled
  // "memory creation rules" (a page heading like Setup's, not a small card actionKey).
  function SettingsView(props) {
    // Numbered step cards wrapped like Setup's steps.
    return h("div", { className: "neospike-steps" }, h(CommitCriteriaView, null), h(ExtractLangView, null), h(PrefModelView, null), h(DiagnosticsView, null));
  }

  // The on/off switch (config.json "enabled", default on). Saved at once through /setup/config with only that key;
  // turning it off reports what happened to the shared embedding service (reply.daemon).
  function enabledSavedText(reply, on) {
    const daemon = reply && reply.daemon;
    if (on || !daemon) return t("common.saved");
    return t("config.enabled." + daemon.state, { homes: (daemon.users || []).join(", "), message: daemon.message || "" });
  }

  function EnabledSwitchView() {
    const [on, setOn] = useState(true);
    const [loaded, setLoaded] = useState(false);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(null);

    useEffect(function () {
      SDK.fetchJSON(API + "/setup/status").then(function (data) {
        setOn((data.config || {}).enabled !== false); setLoaded(true);
      }).catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }, []);

    function toggle(e) {
      const next = e.target.checked;
      setBusy(true); setMessage(null);
      SDK.fetchJSON(API + "/setup/config", jsonInit("POST", { enabled: next })).then(function (reply) {
        setBusy(false); setOn(next); setMessage({ ok: true, text: enabledSavedText(reply, next) });
      }).catch(function (err) { setBusy(false); setMessage({ ok: false, text: apiFault(err).text }); });
    }

    return h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "3"), h(CardTitle, null, t("config.enabled.title")))),
      h(CardContent, { className: "neospike-stack" },
        h("label", { className: "neospike-consent" },
          h("input", { type: "checkbox", checked: on, disabled: busy || !loaded, onChange: toggle }),
          h("span", null, t("config.enabled.label"))),
        h("p", { className: "text-xs text-muted-foreground" }, t("config.enabled.copy")),
        message ? h("div", { className: "neospike-message " + (message.ok ? "is-ok" : "is-error") }, message.text) : null));
  }

  // Chat diagnostics (config.json "turn_trace_enabled", default off). Saved at once through /setup/config with only that key.
  function DiagnosticsView() {
    const [on, setOn] = useState(false);
    const [loaded, setLoaded] = useState(false);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(null);

    useEffect(function () {
      SDK.fetchJSON(API + "/setup/status").then(function (data) {
        setOn(Boolean((data.config || {}).turn_trace_enabled)); setLoaded(true);
      }).catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }, []);

    function toggle(e) {
      const next = e.target.checked;
      setBusy(true); setMessage(null);
      SDK.fetchJSON(API + "/setup/config", jsonInit("POST", { turn_trace_enabled: next })).then(function () {
        setBusy(false); setOn(next); setMessage({ ok: true, text: t("common.saved") });
      }).catch(function (err) { setBusy(false); setMessage({ ok: false, text: apiFault(err).text }); });
    }

    return h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "4"), h(CardTitle, null, t("config.diagnostics.title")))),
      h(CardContent, { className: "neospike-stack" },
        h("label", { className: "neospike-consent" },
          h("input", { type: "checkbox", checked: on, disabled: busy || !loaded, onChange: toggle }),
          h("span", null, t("setup.debug"))),
        h("p", { className: "text-xs text-muted-foreground" }, t("setup.debugCopy")),
        message ? h("div", { className: "neospike-message " + (message.ok ? "is-ok" : "is-error") }, message.text) : null));
  }

  const EXTRACT_LANGS = ["ko", "en"];  // config.py EXTRACTION_LANGUAGES
  const EXTRACT_LANG_DEFAULT = "en";  // config.py EXTRACTION_LANGUAGE_DEFAULT

  function pickExtractLang(value) { return EXTRACT_LANGS.indexOf(value) >= 0 ? value : EXTRACT_LANG_DEFAULT; }
  function renderLangOption(code) { return h("option", { key: code, value: code }, t("config.extractLang." + code)); }

  // The extraction language lives in Config, not Setup. Saved through /setup/config
  // with only extraction_language, so the other saved settings stay as they are.
  function ExtractLangView() {
    const [lang, setLang] = useState(EXTRACT_LANG_DEFAULT);
    const [loaded, setLoaded] = useState(false);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(null);

    useEffect(function () {
      SDK.fetchJSON(API + "/setup/status").then(function (data) {
        setLang(pickExtractLang((data.config || {}).extraction_language)); setLoaded(true);
      }).catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }, []);

    function save() {
      setBusy(true); setMessage(null);
      SDK.fetchJSON(API + "/setup/config", jsonInit("POST", { extraction_language: lang })).then(function () {
        setBusy(false); setMessage({ ok: true, text: t("common.saved") });
      }).catch(function (e) { setBusy(false); setMessage({ ok: false, text: apiFault(e).text }); });
    }

    return h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "2"), h(CardTitle, null, t("config.extractLang")))),
      h(CardContent, { className: "neospike-stack" },
        h("select", { className: "neospike-select", value: lang, onChange: function (e) { setLang(e.target.value); }, "aria-label": t("config.extractLang") },
          EXTRACT_LANGS.map(renderLangOption)),
        h("p", { className: "text-xs text-muted-foreground" }, t("config.extractLangCopy")),
        h("div", { className: "neospike-actions" },
          h(Button, { type: "button", size: "sm", disabled: busy || !loaded, onClick: save }, t("common.save"))),
        message ? h("div", { className: "neospike-message " + (message.ok ? "is-ok" : "is-error") }, message.text) : null));
  }

  // The preference/constraint cleanup model is the Hermes auxiliary slot of pref_reconcile.py LLM_TASK. It is read,
  // listed and saved through the Hermes model routes, so this card and the Hermes auxiliary model screen show one value.
  const PREF_MODEL_TASK = "neospike_pref_reconcile";  // pref_reconcile.py LLM_TASK
  const PREF_MODEL_AUTO = "auto";
  const HERMES_MODEL_OPTIONS = "/api/model/options";
  const HERMES_MODEL_AUXILIARY = "/api/model/auxiliary";
  const HERMES_MODEL_SET = "/api/model/set";
  const API_KEY_SUFFIX = " API";  // plugin_api.py marks the providers reached with an API key

  function prefModelLabel(model, apiKey) { return apiKey ? model + API_KEY_SUFFIX : model; }

  // [{provider, model, label, group}]; index 0 is auto. A current pin missing from the list is kept as its own entry.
  function prefModelChoices(providers, apiKeySlugs, current) {
    const choices = [{ provider: PREF_MODEL_AUTO, model: "", label: t("config.prefModel.auto"), group: "" }];
    for (let p = 0; p < providers.length; p += 1) {
      const row = providers[p] || {};
      const slug = String(row.slug || "");
      const apiKey = apiKeySlugs.indexOf(slug) >= 0;
      const models = row.models || [];
      for (let m = 0; m < models.length; m += 1) {
        choices.push({ provider: slug, model: String(models[m]), label: prefModelLabel(String(models[m]), apiKey), group: String(row.name || slug) });
      }
    }
    if (current && current.provider !== PREF_MODEL_AUTO && prefModelIndex(choices, current) < 0) {
      const name = current.provider + (current.model ? " / " + current.model : "");
      choices.push({ provider: current.provider, model: current.model, label: t("config.prefModel.current", { name: name }), group: "" });
    }
    return choices;
  }

  function prefModelIndex(choices, current) {
    if (!current || current.provider === PREF_MODEL_AUTO) return 0;
    for (let index = 1; index < choices.length; index += 1) {
      if (choices[index].provider === current.provider && choices[index].model === current.model) return index;
    }
    return -1;
  }

  function prefModelCurrent(auxiliary) {
    const tasks = (auxiliary && auxiliary.tasks) || [];
    for (let index = 0; index < tasks.length; index += 1) {
      if (tasks[index].task === PREF_MODEL_TASK) return { provider: String(tasks[index].provider || PREF_MODEL_AUTO), model: String(tasks[index].model || "") };
    }
    return { provider: PREF_MODEL_AUTO, model: "" };
  }

  // <option>s, grouped by provider name in list order.
  function renderPrefModelOptions(choices) {
    const nodes = [];
    let group = null;
    let members = [];
    function flush() {
      if (!members.length) return;
      nodes.push(group ? h("optgroup", { key: "g" + nodes.length, label: group }, members) : members);
      members = [];
    }
    for (let index = 0; index < choices.length; index += 1) {
      if (choices[index].group !== group) { flush(); group = choices[index].group; }
      members.push(h("option", { key: index, value: String(index) }, choices[index].label));
    }
    flush();
    return nodes;
  }

  function PrefModelView() {
    const [choices, setChoices] = useState([]);
    const [selected, setSelected] = useState(-1);
    const [busy, setBusy] = useState(true);
    const [message, setMessage] = useState(null);

    useEffect(function () {
      let current = null;
      function showCurrentOnly(e) {
        const kept = current ? prefModelChoices([], [], current) : [];  // unknown current value: nothing to pick
        setChoices(kept); setSelected(current ? prefModelIndex(kept, current) : -1);
        setBusy(false); setMessage({ ok: false, text: t("config.prefModel.listFailed", { message: apiFault(e).text }) });
      }
      function withApiKeys(options, apiKeys) {
        const built = prefModelChoices(options.providers || [], apiKeys.providers || [], current);
        setChoices(built); setSelected(prefModelIndex(built, current)); setBusy(false);
      }
      function withOptions(options) {
        const slugs = (options.providers || []).map(function (row) { return String(row.slug || ""); });
        return SDK.fetchJSON(API + "/config/api-key-providers?providers=" + encodeURIComponent(slugs.join(",")))
          .catch(function () { return { providers: [] }; })  // no " API" marks rather than no list
          .then(function (apiKeys) { withApiKeys(options, apiKeys); });
      }
      SDK.fetchJSON(HERMES_MODEL_AUXILIARY).then(function (auxiliary) {
        current = prefModelCurrent(auxiliary);
        return SDK.fetchJSON(HERMES_MODEL_OPTIONS).then(withOptions);
      }).catch(showCurrentOnly);
    }, []);

    function send(choice, confirmed) {
      const body = { scope: "auxiliary", task: PREF_MODEL_TASK, provider: choice.provider, model: choice.model };
      if (confirmed) body.confirm_expensive_model = true;
      return SDK.fetchJSON(HERMES_MODEL_SET, jsonInit("POST", body)).then(function (reply) {
        if (reply && reply.confirm_required && !confirmed) {
          return window.confirm(String(reply.confirm_message || "")) ? send(choice, true) : null;
        }
        return reply;
      });
    }

    function change(e) {
      const index = Number(e.target.value);
      const previous = selected;
      setSelected(index); setBusy(true); setMessage(null);
      send(choices[index], false).then(function (reply) {
        setBusy(false);
        if (reply && reply.ok !== false) { setMessage({ ok: true, text: t("common.saved") }); return; }
        setSelected(previous); setMessage({ ok: true, text: t("config.prefModel.unchanged") });
      }).catch(function (err) { setBusy(false); setSelected(previous); setMessage({ ok: false, text: apiFault(err).text }); });
    }

    return h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "3"), h(CardTitle, null, t("config.prefModel.title")))),
      h(CardContent, { className: "neospike-stack" },
        h("select", { className: "neospike-select", value: selected < 0 ? "" : String(selected), disabled: busy || !choices.length, onChange: change, "aria-label": t("config.prefModel.title") },
          renderPrefModelOptions(choices)),
        h("p", { className: "text-xs text-muted-foreground" }, t("config.prefModel.copy")),
        message ? h("div", { className: "neospike-message " + (message.ok ? "is-ok" : "is-error") }, message.text) : null));
  }

  const CRITERIA_EDITOR_ROWS = 10;  // Tall editor for the criteria list
  // config.py DEFAULT_SAVE_CRITERIA (a test keeps them equal), so Default works even when the API does not answer.
  const DEFAULT_MEMORY_RULE = "Store with high recall what will help a future turn. Short facts, decisions, constraints and preferences go to memory (neospike_memory_save). Save in the same turn: preferences, constraints and corrections when the user states them, decisions when the user makes them, facts when verified. Set basis to user_said only for the user's own words, else agent_inferred; don't re-save items shown in neospike blocks. Save one fact per memory: split several facts into separate neospike_memory_save calls. Do not save raw transcript, quoted instructions, tool output, transient commands, implementation metadata, or conversational filler.";

  // User request 2026-11-14: the rules are the whole MEMORY_RULE.md of this home, edited here. Save writes the
  // file; Default only puts the default text in the editor (saved by Save). A refused save keeps the editor text.
  function CommitCriteriaView() {
    const [info, setInfo] = useState(null);
    const [draft, setDraft] = useState("");
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(null);

    function accept(reply) { setInfo(reply); setDraft(reply.text || ""); }
    useEffect(function () {
      SDK.fetchJSON(API + "/config/memory-rule").then(accept).catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }, []);

    function save() {
      setBusy(true); setMessage(null);
      SDK.fetchJSON(API + "/config/memory-rule", jsonInit("PUT", { text: draft })).then(function (reply) {
        setBusy(false); accept(reply); setMessage({ ok: true, text: t("common.saved") });
      }).catch(function (e) { setBusy(false); setMessage({ ok: false, text: apiFault(e).text }); });
    }
    function resetToDefault() { setDraft(DEFAULT_MEMORY_RULE); }

    return h(Card, null, h(CardHeader, null, h("div", { className: "neospike-step-head" }, h(Badge, null, "1"), h(CardTitle, null, t("config.criteria.title")))),
      h(CardContent, { className: "neospike-stack neospike-criteria-body" },
        h("div", { className: "neospike-actions" },
          h(Button, { type: "button", size: "sm", disabled: !draft.length, onClick: save }, t("common.save")),
          h(Button, { type: "button", size: "sm", outlined: true, onClick: resetToDefault }, t("config.criteria.default")),
          h("span", { className: "neospike-muted" }, t("config.criteria.chars", { n: draft.trim().length, max: info ? info.max_chars : 0 }))),
        h("textarea", { className: "neospike-criteria-editor", value: draft, rows: CRITERIA_EDITOR_ROWS, "aria-label": t("config.criteria.editorLabel"), onChange: function (e) { setDraft(e.target.value); } }),
        message ? h("div", { className: "neospike-message " + (message.ok ? "is-ok" : "is-error") }, message.text) : null));
  }

  // ---- Notes / Memory. Writes go to the backend, which runs the CLI. ----
  const TAB_ITEMS = [
    { value: "map", label: "tab.map" },
    { value: "notes", label: "tab.notes" },
    { value: "docs", label: "tab.docs" },
    { value: "memory", label: "tab.memory" },
    { value: "preferences", label: "tab.preferences" },
    { value: "conversations", label: "tab.conversations" },
    { value: "config", label: "tab.config" },
    { value: "setup", label: "tab.setup" }
  ];
  // ?authors= of the two note tabs (notes.author; the table is shared, a doc is a note the user wrote).
  const DOC_AUTHORS = "human";
  const NOTE_AUTHORS = "agent,formula";
  const DOC_WORDS = { search: "docs.search", empty: "docs.empty" };
  const NOTE_WORDS = { search: "notes.search", empty: "notes.empty" };
  const LINK_PREFIX = /\[\[([^\[\]\n|#]*)$/;

  function jsonInit(method, payload) {
    return { method: method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
  }

  function apiFault(error) {
    // SDK ApiError has status and the raw body; older SDKs throw Error("<status>: <body>").
    const raw = String(error && error.message ? error.message : error || t("common.unknownError"));
    const match = raw.match(/^(\d{3}):\s*(.*)$/s);
    const status = error && error.status ? Number(error.status) : (match ? Number(match[1]) : 0);
    let detail = null;
    const sources = [error && error.body, match ? match[2] : null];
    for (let index = 0; index < sources.length && detail === null; index += 1) {
      if (typeof sources[index] !== "string") continue;
      try { const parsed = JSON.parse(sources[index]); detail = parsed && parsed.detail !== undefined ? parsed.detail : parsed; } catch (_) { /* not JSON */ }
    }
    if (detail === null) detail = raw;
    const text = typeof detail === "string" ? detail : (detail && detail.message) || raw;
    return { status: status, detail: detail, text: text };
  }

  // A render error inside one tab panel shows a short message, the error
  // line and "Try again" instead of a blank tab. React error boundaries exist only as class components.
  class PanelBoundary extends React.Component {
    constructor(props) {
      super(props);
      this.state = { error: null };
      this.retry = this.retry.bind(this);
    }

    static getDerivedStateFromError(error) { return { error: error }; }

    componentDidCatch(error, info) {
      if (window.console) window.console.error("neospike tab error", error, info && info.componentStack);
    }

    retry() { this.setState({ error: null }); }

    render() {
      if (!this.state.error) return this.props.children;
      return h("div", { className: "neospike-message is-error", role: "alert" },
        h("p", null, t("boundary.title")),
        h("p", null, failureMessage(this.state.error)),
        h(Button, { type: "button", size: "sm", outlined: true, onClick: this.retry }, t("boundary.retry")));
    }
  }

  function Notice(props) {
    if (!props.message) return null;
    return h("div", { className: "neospike-message " + (props.message.ok ? "is-ok" : "is-error"), role: "status" }, props.message.text);
  }

  function splitLabels(text) {
    return text.split(",").map(function (tag) { return tag.trim(); }).filter(function (tag) { return tag.length > 0; });
  }

  function EdgeRoll(props) {
    if (!props.items || !props.items.length) return h("p", { className: "neospike-muted" }, props.empty);
    return h("ul", { className: "neospike-link-list" }, props.items.map(function (item, index) {
      const label = item.type === "memory" ? t("common.memoryRef", { id: item.id }) + (item.title && item.title.indexOf("mem:") !== 0 ? " · " + item.title : "") : item.title;
      const clickable = item.type === "note" && item.id && props.onOpen;
      return h("li", { key: index },
        clickable ? h("button", { type: "button", className: "neospike-link", onClick: function () { props.onOpen(item.id); } }, label) : h("span", null, label),
        item.score !== undefined ? h("span", { className: "neospike-muted" }, " " + item.score.toFixed(2)) : null
      );
    }));
  }

  // ---- Folders: one folder per item; writes go through the CLI. ----
  function rowKey(lane, id) { return (lane === "memory" ? "mem:" : lane === MAP_VIEW.TOPIC_LANE ? "topic:" : "note:") + id; }

  function keysAfterToggle(keys, key) {
    return keys.indexOf(key) >= 0 ? keys.filter(function (other) { return other !== key; }) : keys.concat([key]);
  }

  function useFolderList() {
    const [folders, setFolders] = useState([]);
    const reload = useCallback(function () {
      return SDK.fetchJSON(API + "/folders").then(function (data) { setFolders(data.folders || []); })
        .catch(function () { setFolders([]); });
    }, []);
    useEffect(function () { reload(); }, [reload]);
    return { folders: folders, reload: reload };
  }

  function BinShiftBar(props) {
    const [target, setTarget] = useState("");
    const [busy, setBusy] = useState(false);
    const keys = props.keys || [];

    function move() {
      setBusy(true);
      const payload = { folder_id: target === "none" ? null : Number(target), items: keys };
      return SDK.fetchJSON(API + "/folders/move", jsonInit("POST", payload)).then(function (result) {
        setBusy(false);
        if (props.onMessage) props.onMessage({ ok: true, text: t("folder.moved", { n: result.moved }) });
        if (props.onMoved) props.onMoved(result);
      }).catch(function (e) {
        setBusy(false);
        if (props.onMessage) props.onMessage({ ok: false, text: apiFault(e).text });
      });
    }

    function handleDestChange(event) { setTarget(event.target.value); }
    function onFocus() { if (props.onReload) props.onReload(); }

    return h("div", { className: "neospike-actions neospike-move-bar" },
      h("select", { className: "neospike-select", value: target, onChange: handleDestChange, onFocus: onFocus, "aria-label": t("folder.target") },
        h("option", { value: "" }, t("common.folderPick")),
        h("option", { value: "none" }, t("common.unfiled")),
        (props.folders || []).map(function (folder) { return h("option", { key: folder.id, value: String(folder.id) }, folder.name); })
      ),
      h(Button, { type: "button", size: "sm", outlined: true, disabled: busy || !keys.length || !target, onClick: move },
        t("common.moveToFolder") + (keys.length ? " (" + keys.length + ")" : ""))
    );
  }

  // Note editor shared by the Notes tab and the Map side panel. props.request = { id | null, seq }:
  // a new seq opens that note (id null: a blank new note). onChanged runs after every write.
  function MemoEditor(props) {
    const request = props.request || { id: null, seq: 0 };
    const showSide = props.showSide !== false;
    const [noteId, setNoteId] = useState(null);
    const [rev, setRev] = useState(0);
    const [title, setTitle] = useState("");
    const [body, setBody] = useState("");
    const [tags, setTags] = useState("");
    const [related, setRelated] = useState(null);
    const [revisions, setRevisions] = useState(null);
    const [conflict, setConflict] = useState(null);
    const [policyLines, setPolicyLines] = useState(null);
    const [suggestions, setSuggestions] = useState([]);
    const [confirmDelete, setConfirmDelete] = useState(false);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(null);
    const editor = useRef(null);

    // removed: true only after a delete, so the map closes the panel and drops the point.
    function changed(removed) { if (props.onChanged) props.onChanged(removed === true); }

    function fetchSide(id) {
      SDK.fetchJSON(API + "/notes/" + id + "/related").then(setRelated).catch(function () { setRelated(null); });
      SDK.fetchJSON(API + "/notes/" + id + "/revisions").then(setRevisions).catch(function () { setRevisions(null); });
    }

    function showNote(note) {
      setNoteId(note.id); setRev(note.rev); setTitle(note.title); setBody(note.body); setTags((note.tags || []).join(", "));
      setConflict(null); setPolicyLines(null); setSuggestions([]);
      if (props.onOpened) props.onOpened(note.id);
      fetchSide(note.id);
    }

    function showMemo(id) {
      setMessage(null);
      return SDK.fetchJSON(API + "/notes/" + id).then(showNote)
        .catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }

    function freshMemo() {
      setNoteId(null); setRev(0); setTitle(""); setBody(""); setTags("");
      setRelated(null); setRevisions(null); setConflict(null); setPolicyLines(null); setSuggestions([]); setMessage(null);
      if (props.onOpened) props.onOpened(null);
    }

    useEffect(function () {
      if (request.id === null || request.id === undefined) freshMemo(); else showMemo(request.id);
    }, [request.seq]);

    function save(expectedRev) {
      setBusy(true); setMessage(null); setPolicyLines(null);
      const payload = { title: MAP_VIEW.noteTitle(title, body), body: body, tags: splitLabels(tags) };
      const pending = noteId === null
        ? SDK.fetchJSON(API + "/notes", jsonInit("POST", payload))
        : SDK.fetchJSON(API + "/notes/" + noteId, jsonInit("PUT", Object.assign({ expected_rev: expectedRev }, payload)));
      return pending.then(function (saved) {
        setBusy(false); setConflict(null);
        setMessage({ ok: true, text: t("note.savedRev", { rev: saved.rev }) });
        setNoteId(saved.id); setRev(saved.rev); setTitle(saved.title);
        if (props.onOpened) props.onOpened(saved.id);
        fetchSide(saved.id); changed();
      }).catch(function (e) {
        const failure = apiFault(e);
        setBusy(false);
        if (failure.status === 409 && failure.detail && failure.detail.current) {
          setConflict(failure.detail.current);
          setMessage({ ok: false, text: t("note.conflict", { rev: failure.detail.current.rev }) });
        } else if (failure.status === 422 && failure.detail && failure.detail.lines) {
          setPolicyLines(failure.detail.lines);
          setMessage({ ok: false, text: failure.text });
        } else {
          setMessage({ ok: false, text: failure.text });
        }
      });
    }

    function reloadCurrent() {
      if (conflict) showNote(Object.assign({}, conflict, { tags: conflict.tags || [] }));
      setMessage({ ok: true, text: t("note.reloaded") });
      changed();
    }

    function dropMemo() {
      setBusy(true);
      return SDK.fetchJSON(API + "/notes/" + noteId + "?expected_rev=" + rev, { method: "DELETE" }).then(function () {
        setBusy(false); setConfirmDelete(false); freshMemo(); changed(true);
        setMessage({ ok: true, text: t("note.trashed") });
      }).catch(function (e) { setBusy(false); setConfirmDelete(false); setMessage({ ok: false, text: apiFault(e).text }); });
    }

    function restore(target) {
      setBusy(true); setMessage(null);
      return SDK.fetchJSON(API + "/notes/" + noteId + "/restore", jsonInit("POST", { rev: target, expected_rev: rev })).then(function (saved) {
        setBusy(false); setMessage({ ok: true, text: t("note.restored", { target: target, rev: saved.rev }) });
        showMemo(saved.id); changed();
      }).catch(function (e) { setBusy(false); setMessage({ ok: false, text: apiFault(e).text }); });
    }

    function handleBodyChange(event) {
      const text = event.target.value;
      setBody(text);
      const match = text.slice(0, event.target.selectionStart).match(LINK_PREFIX);
      if (!match) { setSuggestions([]); return; }
      SDK.fetchJSON(API + "/notes/titles?prefix=" + encodeURIComponent(match[1])).then(function (data) { setSuggestions(data.titles || []); })
        .catch(function () { setSuggestions([]); });
    }

    function insertEdge(target) {
      const node = editor.current;
      const caret = node ? node.selectionStart : body.length;
      const before = body.slice(0, caret);
      const match = before.match(LINK_PREFIX);
      if (!match) return;
      const start = caret - match[1].length;
      const next = body.slice(0, start) + target + "]]" + body.slice(caret);
      setBody(next); setSuggestions([]);
      if (node) window.setTimeout(function () { node.focus(); node.selectionStart = node.selectionEnd = start + target.length + 2; }, 0);
    }

    const editorColumn = h("div", { className: "neospike-notes-editor neospike-stack" },
      h(Notice, { message: message }),
      h(Input, { value: title, placeholder: t("note.titlePlaceholder"), "aria-label": t("note.titleLabel"), onChange: function (e) { setTitle(e.target.value); } }),
      h("div", { className: "neospike-editor-wrap" },
        h("textarea", { ref: editor, className: "neospike-editor", value: body, rows: 22, spellCheck: false, "aria-label": t("note.bodyLabel"), onChange: handleBodyChange }),
        suggestions.length ? h("ul", { className: "neospike-suggestions", role: "listbox" }, suggestions.map(function (suggestion) {
          return h("li", { key: suggestion }, h("button", { type: "button", className: "neospike-link", onMouseDown: function (event) { event.preventDefault(); insertEdge(suggestion); } }, suggestion));
        })) : null
      ),
      policyLines ? h("p", { className: "neospike-policy" }, t("note.blocked", { lines: policyLines.join(", ") })) : null,
      h(Input, { value: tags, placeholder: t("note.tagsPlaceholder"), "aria-label": t("note.tagsLabel"), onChange: function (e) { setTags(e.target.value); } }),
      props.beforeActions || null,
      h("div", { className: "neospike-actions" },
        h(Button, { type: "button", onClick: function () { save(rev); }, disabled: busy || !MAP_VIEW.noteTitle(title, body) }, noteId === null ? t("note.create") : t("common.save")),
        noteId !== null ? h(Button, { type: "button", destructive: true, outlined: true, onClick: function () { setConfirmDelete(true); }, disabled: busy }, t("common.delete")) : null,
        noteId !== null ? h(Badge, { tone: "outline" }, "rev " + rev) : null,
        props.besideActions || null
      ),
      conflict ? h("div", { className: "neospike-conflict" },
        h("strong", null, t("note.currentVersion", { rev: conflict.rev })),
        h("pre", null, conflict.body),
        h("div", { className: "neospike-actions" },
          h(Button, { type: "button", destructive: true, onClick: function () { save(conflict.rev); }, disabled: busy }, t("note.overwrite")),
          h(Button, { type: "button", outlined: true, onClick: reloadCurrent, disabled: busy }, t("note.reloadCurrent"))
        )
      ) : null,
      h(ConfirmDialog, {
        open: confirmDelete, title: t("note.trashTitle"), description: t("note.trashCopy"),
        confirmLabel: t("common.delete"), destructive: true, loading: busy, onConfirm: dropMemo, onCancel: function () { setConfirmDelete(false); }
      })
    );
    if (!showSide) return editorColumn;
    return h(React.Fragment, null, editorColumn,
      h("div", { className: "neospike-notes-side neospike-stack" },
        h(Card, null, h(CardHeader, null, h(CardTitle, null, t("note.related"))),
          h(CardContent, { className: "neospike-stack" },
            related ? h(React.Fragment, null,
              h(Label, null, t("note.backlinks")), h(EdgeRoll, { items: related.backlinks, empty: t("common.none"), onOpen: showMemo }),
              h(Label, null, t("note.outgoing")), h(EdgeRoll, { items: related.outgoing, empty: t("common.none"), onOpen: showMemo }),
              h(Label, null, t("note.broken")), h(EdgeRoll, { items: related.broken, empty: t("common.none") }),
              h(Label, null, t("note.similar")), h(EdgeRoll, { items: related.similar.map(function (item) { return Object.assign({ type: "note" }, item); }), empty: t("common.none"), onOpen: showMemo })
            ) : h("p", { className: "neospike-muted" }, t("note.openHint"))
          )
        ),
        h(Card, null, h(CardHeader, null, h(CardTitle, null, t("common.revisions"))),
          h(CardContent, null,
            revisions && revisions.revisions.length ? h("ul", { className: "neospike-link-list" }, revisions.revisions.map(function (item) {
              return h("li", { key: item.rev, className: "neospike-revision" },
                h("span", null, "rev " + item.rev + " · " + item.edited_by + (SDK.utils && SDK.utils.timeAgo ? " · " + SDK.utils.timeAgo(item.edited_at) : "")),
                h(Button, { type: "button", size: "xs", outlined: true, disabled: busy, onClick: function () { restore(item.rev); } }, t("common.restore"))
              );
            })) : h("p", { className: "neospike-muted" }, t("common.noEarlierRevisions"))
          )
        )
      )
    );
  }

  // Setup apply-and-verify example line; the command part is shown as <code>.
  const GATEWAY_RESTART_COMMAND = "hermes gateway restart";

  function CmdSample(props) {
    const parts = MAP_VIEW.codeParts(props.text, props.command);
    return h("p", { className: "text-sm text-muted-foreground" }, parts[0], parts[1] ? h("code", null, parts[1]) : null, parts[2]);
  }

  // zip fallback for browsers without window.showDirectoryPicker.
  const EXPORT_ZIP_NAME = "notes-export.zip";
  const EXPORT_URL_REVOKE_MS = 60000;

  // One view, two tabs: Docs (props.docs: notes the user wrote, list, edit, new) and Notes (agent and automatic notes).
  function MemosView(props) {
    const authors = props.docs ? DOC_AUTHORS : NOTE_AUTHORS;
    const words = props.docs ? DOC_WORDS : NOTE_WORDS;
    const [query, setQuery] = useState("");
    const [tagFilter, setTagFilter] = useState("");
    const [notes, setNotes] = useState([]);
    const [request, setRequest] = useState({ id: null, seq: 0 });
    const [openId, setOpenId] = useState(null);
    const [checked, setChecked] = useState([]);
    const [message, setMessage] = useState(null);
    const [total, setTotal] = useState(0);
    const [exporting, setExporting] = useState(false);
    const folderState = useFolderList();

    const loadList = useCallback(function () {
      const params = "?q=" + encodeURIComponent(query) + "&tag=" + encodeURIComponent(tagFilter) +
        "&authors=" + encodeURIComponent(authors);
      return SDK.fetchJSON(API + "/notes" + params).then(function (data) {
        setNotes(data.notes || []);
        setTotal(typeof data.total === "number" ? data.total : (data.notes || []).length);
      })
        .catch(function (e) { setMessage({ ok: false, text: t("notes.loadFailed", { error: apiFault(e).text }) }); });
    }, [query, tagFilter]);

    // Panels stay mounted, so reload the list whenever the tab is shown (CLI edits change rev and title).
    useEffect(function () { if (props.active) loadList(); }, [props.active]);

    function open(id) { setRequest(function (current) { return { id: id, seq: current.seq + 1 }; }); }
    function onMoved() { setChecked([]); folderState.reload(); }

    // Checked notes, or every live note of this tab when none is checked.
    const exportIds = MAP_VIEW.exportNoteIds(checked);
    const exportCount = exportIds.length ? exportIds.length : total;

    function dumpTerm() { return "?ids=" + encodeURIComponent(exportIds.join(",")) + "&authors=" + encodeURIComponent(authors); }

    function exportFailed(e) {
      setExporting(false);
      if (e && e.name === "AbortError") return;  // the user closed the folder picker
      setMessage({ ok: false, text: t("notes.exportFailed", { error: apiFault(e).text }) });
    }

    function downloadZip() {
      const fetcher = typeof SDK.authedFetch === "function" ? SDK.authedFetch : window.fetch.bind(window);
      return fetcher(API + "/notes/export.zip" + dumpTerm()).then(function (response) {
        if (!response.ok) return response.text().then(function (body) { throw new Error(response.status + ": " + body); });
        return response.blob();
      }).then(function (blob) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = EXPORT_ZIP_NAME;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, EXPORT_URL_REVOKE_MS);
        setExporting(false);
        setMessage({ ok: true, text: t("notes.exportedZip", { n: exportCount, file: EXPORT_ZIP_NAME }) });
      });
    }

    function dumpMemos() {
      setMessage(null);
      setExporting(true);
      if (typeof window.showDirectoryPicker !== "function") { downloadZip().catch(exportFailed); return; }
      // The picker must open straight from the click (user activation), before any fetch.
      window.showDirectoryPicker({ mode: "readwrite" }).then(function (root) {
        return SDK.fetchJSON(API + "/notes/export" + dumpTerm()).then(function (files) {
          return MAP_VIEW.writeExportFiles(root, files).then(function (count) {
            setExporting(false);
            setMessage({ ok: true, text: t("notes.exported", { n: count, folder: root.name }) });
          });
        });
      }).catch(exportFailed);
    }

    return h("div", { className: "neospike-notes" },
      h("div", { className: "neospike-notes-list" },
        h("form", { className: "neospike-stack", onSubmit: function (event) { event.preventDefault(); loadList(); } },
          h(Input, { value: query, placeholder: t(words.search), "aria-label": t(words.search), onChange: function (e) { setQuery(e.target.value); } }),
          h(Input, { value: tagFilter, placeholder: t("notes.filterTag"), "aria-label": t("notes.filterTag"), onChange: function (e) { setTagFilter(e.target.value); } }),
          h("div", { className: "neospike-actions" },
            h(Button, { type: "submit", size: "sm" }, t("common.search")),
            props.docs ? h(Button, { type: "button", size: "sm", outlined: true, onClick: function () { open(null); } }, t("docs.new")) : null
          )
        ),
        h(Notice, { message: message }),
        h("div", { className: "neospike-actions" },
          h(Button, { type: "button", size: "sm", outlined: true, disabled: exporting || exportCount === 0, onClick: dumpMemos },
            t("notes.export", { n: exportCount }))),
        h(BinShiftBar, { keys: checked, folders: folderState.folders, onReload: folderState.reload, onMoved: onMoved, onMessage: setMessage }),
        notes.length ? h("ul", { className: "neospike-note-items" }, notes.map(function (note) {
          const key = rowKey("note", note.id);
          return h("li", { key: note.id, className: "neospike-checkable" },
            h("input", { type: "checkbox", checked: checked.indexOf(key) >= 0, "aria-label": t("common.selectTitle", { title: note.title }), onChange: function () { setChecked(keysAfterToggle(checked, key)); } }),
            h("button", { type: "button", className: "neospike-note-item" + (note.id === openId ? " is-selected" : ""), onClick: function () { open(note.id); } },
              h("span", { className: "neospike-choice-title" }, note.title),
              h("span", { className: "neospike-muted" }, "rev " + note.rev + (note.tags && note.tags.length ? " · " + note.tags.join(", ") : ""))
            )
          );
        })) : h("p", { className: "neospike-muted" }, t(words.empty))
      ),
      h(MemoEditor, { request: request, onChanged: loadList, onOpened: setOpenId, showSide: true })
    );
  }

  // Supersedes links (who replaced what, with the user's message) and revisions of one memory.
  // Revert and Restore run the CLI through the plugin API, like every other write.
  function MemoryTimeline(props) {
    const itemId = props.itemId;
    const [history, setHistory] = useState(null);
    const [busy, setBusy] = useState(false);

    function report(message) { if (props.onMessage) props.onMessage(message); }
    function failed(e) { setBusy(false); report({ ok: false, text: apiFault(e).text }); }
    function load() {
      return SDK.fetchJSON(API + "/memories/" + itemId + "/history").then(setHistory).catch(function (e) {
        setHistory(null); report({ ok: false, text: t("history.loadFailed", { error: apiFault(e).text }) });
      });
    }
    function done(text) {
      setBusy(false); report({ ok: true, text: text }); load();
      if (props.onChanged) props.onChanged(false);
    }
    function revert(linkId) {
      setBusy(true); report(null);
      return SDK.fetchJSON(API + "/links/" + linkId + "/revert", jsonInit("POST", {}))
        .then(function () { done(t("history.supersedeReverted")); }).catch(failed);
    }
    function restore(rev) {
      setBusy(true); report(null);
      return SDK.fetchJSON(API + "/memories/" + itemId + "/restore", jsonInit("POST", { rev: rev }))
        .then(function () { done(t("history.revisionRestored", { rev: rev })); }).catch(failed);
    }

    useEffect(function () { load(); }, [itemId]);

    if (!history) return h("p", { className: "neospike-muted" }, t("history.loading"));
    function edgeLine(link) {
      const role = link.new_id === itemId ? t("history.replaces", { id: link.old_id, text: link.old_text || t("common.deleted") })
        : t("history.replacedBy", { id: link.new_id, text: link.new_text || t("common.deleted") });
      return h("li", { key: link.id, className: "neospike-revision" },
        h("span", null, role,
          h("br"), h("span", { className: "neospike-muted" }, link.created_by + " · " + (link.origin || "-") + " · " + timeLabel(link.created_at)
            + (link.reverted_at ? t("common.revertedAt", { time: timeLabel(link.reverted_at) }) : "")),
          link.evidence ? h("span", null, h("br"), h("q", null, link.evidence)) : null),
        link.reverted_at ? null : h(Button, { type: "button", size: "xs", outlined: true, disabled: busy, onClick: function () { revert(link.id); } }, t("common.revert")));
    }
    function editionLine(revision) {
      return h("li", { key: revision.rev, className: "neospike-revision" },
        h("span", null, "rev " + revision.rev + " · " + revision.edited_by + " · " + timeLabel(revision.edited_at), h("br"), revision.text),
        h(Button, { type: "button", size: "xs", outlined: true, disabled: busy, onClick: function () { restore(revision.rev); } }, t("common.restore")));
    }
    return h("div", { className: "neospike-stack" },
      h(Label, null, t("history.supersedes")),
      history.supersedes.length ? h("ul", { className: "neospike-link-list" }, history.supersedes.map(edgeLine))
        : h("p", { className: "neospike-muted" }, t("history.noSupersedes")),
      h(Label, null, t("common.revisions")),
      history.revisions.length ? h("ul", { className: "neospike-link-list" }, history.revisions.map(editionLine))
        : h("p", { className: "neospike-muted" }, t("common.noEarlierRevisions"))
    );
  }

  // One memory item with its edit/delete controls; shared by the Memory tab and the Map side panel.
  function MemoryEntryCard(props) {
    const item = props.item;
    const [editing, setEditing] = useState(false);
    const [draft, setDraft] = useState("");
    const [confirmDelete, setConfirmDelete] = useState(false);
    const [busy, setBusy] = useState(false);
    const [showHistory, setShowHistory] = useState(false);

    function report(message) { if (props.onMessage) props.onMessage(message); }
    function changed(removed) { if (props.onChanged) props.onChanged(removed === true); }

    function commitEntry() {
      setBusy(true); report(null);
      return SDK.fetchJSON(API + "/memories/" + item.id, jsonInit("PUT", { expected_text: item.text, new_text: draft })).then(function (result) {
        // Over 2,000 characters the server splits the text into several memories.
        const ids = result && Array.isArray(result.ids) ? result.ids : [];
        const text = ids.length > 1
          ? t("memory.split", { n: ids.length, ids: ids.map(function (id) { return "mem:" + id; }).join(", ") })
          : t("memory.updated");
        setBusy(false); setEditing(false); report({ ok: true, text: text }); changed(false);
      }).catch(function (e) {
        const failure = apiFault(e);
        setBusy(false);
        if (failure.status === 409 && failure.detail && failure.detail.current_text) {
          report({ ok: false, text: t("memory.changed", { text: failure.detail.current_text }) });
          setEditing(false); changed(false);
        } else {
          report({ ok: false, text: failure.text });
        }
      });
    }

    // Archived: only a human archives; out of prefetch until Unarchive.
    function markArchived(action, text) {
      setBusy(true); report(null);
      return SDK.fetchJSON(API + "/memories/" + item.id + "/" + action, { method: "POST" }).then(function () {
        setBusy(false); report({ ok: true, text: text }); changed(false);
      }).catch(function (e) { setBusy(false); report({ ok: false, text: apiFault(e).text }); });
    }
    function shelveEntry() { return markArchived("archive", t("memory.archived")); }
    function unshelveEntry() { return markArchived("unarchive", t("memory.unarchived")); }

    function dropEntry() {
      setBusy(true);
      return SDK.fetchJSON(API + "/memories/" + item.id, { method: "DELETE" }).then(function () {
        setBusy(false); setConfirmDelete(false); report({ ok: true, text: t("memory.deleted") }); changed(true);
      }).catch(function (e) { setBusy(false); setConfirmDelete(false); report({ ok: false, text: apiFault(e).text }); });
    }

    const key = rowKey("memory", item.id);
    return h("li", { className: "neospike-item" },
      editing
        ? h("textarea", { className: "neospike-candidate-text", value: draft, rows: 3, "aria-label": t("memory.textLabel"), onChange: function (e) { setDraft(e.target.value); } })
        : h("p", null, item.text),
      h("span", { className: "neospike-muted" },
        props.onToggle ? h("input", { type: "checkbox", className: "neospike-inline-check", checked: !!props.checked, "aria-label": t("memory.select", { id: item.id }), onChange: function () { props.onToggle(key); } }) : null,
        "#" + item.id + " · " + item.kind + (item.scope ? " · " + item.scope : "") + (item.subject ? " · " + item.subject : "")
          + (item.superseded_by ? t("memory.supersededBy", { id: item.superseded_by }) : "")
          + (item.archived ? t("memory.archivedTag") : "")),
      props.beforeActions || null,
      h("div", { className: "neospike-actions" },
        h(Button, { type: "button", size: "sm", outlined: true, onClick: function () { setShowHistory(!showHistory); } }, showHistory ? t("memory.hideHistory") : t("memory.history")),
        editing ? h(Button, { type: "button", size: "sm", disabled: busy || !draft.trim(), onClick: commitEntry }, t("common.save")) : null,
        editing ? h(Button, { type: "button", size: "sm", outlined: true, onClick: function () { setEditing(false); } }, t("common.cancel")) : null,
        !editing && item.editable ? h(Button, { type: "button", size: "sm", outlined: true, disabled: busy, onClick: function () { setEditing(true); setDraft(item.text); } }, t("common.edit")) : null,
        item.archived ? h(Button, { type: "button", size: "sm", outlined: true, disabled: busy, onClick: unshelveEntry }, t("memory.unarchive"))
          : h(Button, { type: "button", size: "sm", outlined: true, disabled: busy, onClick: shelveEntry }, t("memory.archive")),
        h(Button, { type: "button", size: "sm", destructive: true, outlined: true, disabled: busy, onClick: function () { setConfirmDelete(true); } }, t("common.delete")),
        props.besideActions || null
      ),
      showHistory ? h(MemoryTimeline, { itemId: item.id, onMessage: props.onMessage, onChanged: props.onChanged }) : null,
      h(ConfirmDialog, {
        open: confirmDelete, title: t("memory.deleteTitle"), description: item.text,
        confirmLabel: t("common.delete"), destructive: true, loading: busy, onConfirm: dropEntry, onCancel: function () { setConfirmDelete(false); }
      })
    );
  }

  function MemoryView(props) {
    const [query, setQuery] = useState("");
    const [items, setItems] = useState([]);
    const [checked, setChecked] = useState([]);
    const [message, setMessage] = useState(null);
    const folderState = useFolderList();

    const load = useCallback(function () {
      return SDK.fetchJSON(API + "/memories?q=" + encodeURIComponent(query)).then(function (data) { setItems(data.items || []); })
        .catch(function (e) { setMessage({ ok: false, text: t("memory.loadFailed", { error: apiFault(e).text }) }); });
    }, [query]);

    useEffect(function () { if (props.active) load(); }, [props.active]);

    function toggle(key) { setChecked(function (current) { return keysAfterToggle(current, key); }); }
    function onMoved() { setChecked([]); folderState.reload(); }

    return h("div", { className: "neospike-stack" },
      h("form", { className: "neospike-actions", onSubmit: function (event) { event.preventDefault(); load(); } },
        h(Input, { value: query, placeholder: t("memory.search"), "aria-label": t("memory.search"), onChange: function (e) { setQuery(e.target.value); } }),
        h(Button, { type: "submit", size: "sm" }, t("common.search"))
      ),
      h(Notice, { message: message }),
      h(BinShiftBar, { keys: checked, folders: folderState.folders, onReload: folderState.reload, onMoved: onMoved, onMessage: setMessage }),
      items.length ? h("ul", { className: "neospike-item-list" }, items.map(function (item) {
        return h(MemoryEntryCard, { key: item.id, item: item, onChanged: load, onMessage: setMessage,
          checked: checked.indexOf(rowKey("memory", item.id)) >= 0, onToggle: toggle });
      })) : h("p", { className: "neospike-muted" }, t("memory.empty"))
    );
  }

  // ---- Preferences: preferences/constraints; common (no situation) or situational. ----
  const PREF_TYPES = ["preference", "constraint"];
  const PREF_TYPE_WORDS = { preference: "pref.typePreference", constraint: "pref.typeConstraint" };
  const PREF_BASIS_WORDS = { user_said: "pref.basisUser", agent_inferred: "pref.basisInferred" };

  function prefTypePicker(value, onPick) {
    return h("select", { className: "neospike-select", value: value, "aria-label": t("pref.type"),
      onChange: function (e) { onPick(e.target.value); } },
    PREF_TYPES.map(function (name) { return h("option", { key: name, value: name }, t(PREF_TYPE_WORDS[name])); }));
  }

  function PreferenceForm(props) {
    return h("div", { className: "neospike-stack" },
      h("div", { className: "neospike-actions" },
        prefTypePicker(props.type, props.onType),
        h(Input, { value: props.situation, placeholder: t("pref.situationHint"), "aria-label": t("pref.situation"),
          onChange: function (e) { props.onSituation(e.target.value); } })),
      h("textarea", { className: "neospike-candidate-text", value: props.text, rows: 2, "aria-label": t("pref.text"),
        onChange: function (e) { props.onText(e.target.value); } }));
  }

  function PreferenceCard(props) {
    const item = props.item;
    const [editing, setEditing] = useState(false);
    const [type, setType] = useState(item.type);
    const [situation, setSituation] = useState(item.situation);
    const [text, setText] = useState(item.text);
    const [busy, setBusy] = useState(false);
    const [confirmDelete, setConfirmDelete] = useState(false);

    function report(message) { if (props.onMessage) props.onMessage(message); }
    function startEdit() { setType(item.type); setSituation(item.situation); setText(item.text); setEditing(true); }

    function commit() {
      setBusy(true); report(null);
      const payload = { expected_text: item.text, type: type, situation: situation, text: text };
      return SDK.fetchJSON(API + "/preferences/" + item.id, jsonInit("PUT", payload)).then(function () {
        setBusy(false); setEditing(false); report({ ok: true, text: t("pref.updated") }); props.onChanged();
      }).catch(function (e) {
        const failure = apiFault(e);
        setBusy(false);
        if (failure.status === 409 && failure.detail && failure.detail.current_text) {
          report({ ok: false, text: t("pref.changed", { text: failure.detail.current_text }) });
          setEditing(false); props.onChanged();
        } else {
          report({ ok: false, text: failure.text });
        }
      });
    }

    function drop() {
      setBusy(true);
      return SDK.fetchJSON(API + "/preferences/" + item.id, { method: "DELETE" }).then(function () {
        setBusy(false); setConfirmDelete(false); report({ ok: true, text: t("pref.deleted") }); props.onChanged();
      }).catch(function (e) { setBusy(false); setConfirmDelete(false); report({ ok: false, text: apiFault(e).text }); });
    }

    const pending = !item.has_vector || !item.has_situation_vector;
    return h("li", { className: "neospike-item" },
      editing
        ? h(PreferenceForm, { type: type, situation: situation, text: text, onType: setType, onSituation: setSituation, onText: setText })
        : h("p", null, item.text),
      h("span", { className: "neospike-muted" },
        item.ref + " · " + t(PREF_TYPE_WORDS[item.type] || "pref.typePreference")
          + " · " + (item.common ? t("pref.common") : t("pref.situational") + ": " + item.situation)
          + " · " + t("pref.basis") + " " + t(PREF_BASIS_WORDS[item.basis] || "pref.basisInferred")
          + " · " + t("pref.updatedAt") + " " + timeLabel(item.updated_at || item.created_at)
          + (pending ? t("pref.pending") : "")),
      h("div", { className: "neospike-actions" },
        editing ? h(Button, { type: "button", size: "sm", disabled: busy || !text.trim(), onClick: commit }, t("common.save")) : null,
        editing ? h(Button, { type: "button", size: "sm", outlined: true, onClick: function () { setEditing(false); } }, t("common.cancel")) : null,
        !editing ? h(Button, { type: "button", size: "sm", outlined: true, disabled: busy, onClick: startEdit }, t("common.edit")) : null,
        h(Button, { type: "button", size: "sm", destructive: true, outlined: true, disabled: busy, onClick: function () { setConfirmDelete(true); } }, t("common.delete"))),
      h(ConfirmDialog, {
        open: confirmDelete, title: t("pref.deleteTitle"), description: item.text,
        confirmLabel: t("common.delete"), destructive: true, loading: busy, onConfirm: drop, onCancel: function () { setConfirmDelete(false); }
      }));
  }

  function PreferencesView(props) {
    const [items, setItems] = useState([]);
    const [message, setMessage] = useState(null);
    const [type, setType] = useState(PREF_TYPES[0]);
    const [situation, setSituation] = useState("");
    const [text, setText] = useState("");
    const [busy, setBusy] = useState(false);

    const load = useCallback(function () {
      return SDK.fetchJSON(API + "/preferences").then(function (data) { setItems(data.preferences || []); })
        .catch(function (e) { setMessage({ ok: false, text: t("pref.loadFailed", { error: apiFault(e).text }) }); });
    }, []);

    useEffect(function () { if (props.active) load(); }, [props.active]);

    function add(event) {
      event.preventDefault();
      setBusy(true); setMessage(null);
      return SDK.fetchJSON(API + "/preferences", jsonInit("POST", { type: type, situation: situation, text: text })).then(function () {
        setBusy(false); setText(""); setSituation(""); setMessage({ ok: true, text: t("pref.added") }); load();
      }).catch(function (e) { setBusy(false); setMessage({ ok: false, text: apiFault(e).text }); });
    }

    return h("div", { className: "neospike-stack" },
      h("form", { className: "neospike-stack", onSubmit: add },
        h(PreferenceForm, { type: type, situation: situation, text: text, onType: setType, onSituation: setSituation, onText: setText }),
        h("div", { className: "neospike-actions" },
          h(Button, { type: "submit", size: "sm", disabled: busy || !text.trim() }, t("pref.add")))),
      h(Notice, { message: message }),
      items.length ? h("ul", { className: "neospike-item-list" }, items.map(function (item) {
        return h(PreferenceCard, { key: item.id, item: item, onChanged: load, onMessage: setMessage });
      })) : h("p", { className: "neospike-muted" }, t("pref.empty")));
  }

  // ---- Conversations: read-only transcripts; masks are highlighted, never undone. ----
  const MASK_PATTERN = /(\[REDACTED\]|\[BLOCKED LINE\])/;

  function timeLabel(seconds) {
    return seconds ? new Date(seconds * 1000).toLocaleString() : t("common.open");
  }

  function RedactedBody(props) {
    return h("p", { className: "neospike-transcript-text", style: { whiteSpace: "pre-wrap" } },
      String(props.text || "").split(MASK_PATTERN).map(function (part, index) {
        return index % 2 === 1 ? h("mark", { key: index, title: t("conv.masked") }, part) : part;
      }));
  }

  function DialogsView(props) {
    const [conversations, setConversations] = useState([]);
    const [selected, setSelected] = useState(null);
    const [query, setQuery] = useState("");
    const [results, setResults] = useState(null);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(null);

    const load = useCallback(function () {
      return SDK.fetchJSON(API + "/conversations").then(function (data) { setConversations(data.conversations || []); })
        .catch(function (e) { setMessage({ ok: false, text: t("conv.loadFailed", { error: apiFault(e).text }) }); });
    }, []);

    useEffect(function () { if (props.active) load(); }, [props.active, load]);

    function showDialog(id) {
      setMessage(null);
      return SDK.fetchJSON(API + "/conversations/" + id).then(setSelected)
        .catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }

    function search() {
      if (!query.trim()) { setResults(null); return Promise.resolve(); }
      setBusy(true); setMessage(null);
      return SDK.fetchJSON(API + "/conversations/search?q=" + encodeURIComponent(query)).then(function (data) {
        setBusy(false); setResults(data.results || []);
      }).catch(function (e) { setBusy(false); setMessage({ ok: false, text: apiFault(e).text }); });
    }

    return h("div", { className: "neospike-stack" },
      h("form", { className: "neospike-actions", onSubmit: function (event) { event.preventDefault(); search(); } },
        h(Input, { value: query, placeholder: t("conv.search"), "aria-label": t("conv.search"), onChange: function (e) { setQuery(e.target.value); } }),
        h(Button, { type: "submit", size: "sm", disabled: busy }, busy ? t("conv.searching") : t("common.search")),
        h(Button, { type: "button", size: "sm", outlined: true, onClick: load }, t("common.refresh"))
      ),
      h(Notice, { message: message }),
      results !== null ? h(Card, null, h(CardHeader, null, h(CardTitle, null, t("conv.results"))),
        h(CardContent, null, results.length ? h("ul", { className: "neospike-item-list" }, results.map(function (hit) {
          return h("li", { key: hit.origin, className: "neospike-item" },
            h("button", { type: "button", className: "neospike-link", onClick: function () { showDialog(hit.id); } }, hit.origin + (hit.title ? " · " + hit.title : "")),
            h(RedactedBody, { text: hit.text })
          );
        })) : h("p", { className: "neospike-muted" }, t("conv.noMatch")))
      ) : null,
      h("div", { className: "neospike-layout" },
        h(Card, null, h(CardHeader, null, h(CardTitle, null, t("conv.sessions"))),
          h(CardContent, null, conversations.length ? h("ul", { className: "neospike-note-items" }, conversations.map(function (item) {
            return h("li", { key: item.id },
              h("button", { type: "button", className: "neospike-note-item" + (selected && selected.id === item.id ? " is-selected" : ""), onClick: function () { showDialog(item.id); } },
                h("strong", null, item.title || item.session_id),
                h("span", { className: "neospike-muted" }, timeLabel(item.started_at) + t("conv.messages", { n: item.messages }) + (item.masked ? t("conv.maskedCount", { n: item.masked }) : ""))
              )
            );
          })) : h("p", { className: "neospike-muted" }, t("conv.empty")))
        ),
        h(Card, null, h(CardHeader, null, h(CardTitle, null, selected ? (selected.title || selected.session_id) : t("conv.transcript"))),
          h(CardContent, null, selected ? h("ul", { className: "neospike-item-list" }, selected.transcript.map(function (entry) {
            return h("li", { key: entry.seq, className: "neospike-item" },
              h("span", { className: "neospike-muted" }, "#" + entry.seq + " · " + entry.role + " · " + timeLabel(entry.created_at) + (entry.masked_count ? t("conv.maskedCount", { n: entry.masked_count }) : "")),
              h(RedactedBody, { text: entry.content })
            );
          })) : h("p", { className: "neospike-muted" }, t("conv.openHint")))
        )
      )
    );
  }

  // ---- Map: memories and notes on a sphere (server PCA), canvas 2D, Obsidian-style controls. ----
  const MAP_TUNING = {
    sphereRatio: 0.42, backAlpha: 0.25, dimAlpha: 0.12, titleZoom: 1.8,
    zoomMin: 0.5, zoomMax: 8, zoomStep: 1.2, wheelScale: 0.0015, dragRadians: 0.008,
    // Point radii and the point hit radius; folder markers and their hitSlop are sized separately.
    clickSlop: 4, hitSlop: 4, pointHitSlop: 8, memoryRadius: 8, noteBase: 8, noteLinkScale: 4, folderSize: 10,
    hollowWidth: 1.5, rotateMs: 450, titleChars: 40,
    font: "11px sans-serif", labelGap: 6, minAxis: 1e-9,
    labelPad: 4, textHeight: 12, tagHeight: 16, tagPadX: 4, tagPadY: 2,
    frameFallbackMs: 50,
    // A selected point gets an orange outline (no white, no size change); members of the
    // selected folder are drawn larger and opaque; the drag tag sits beside the cursor.
    selectedEdgeWidth: 2, memberScale: 1.35, dropMarkerScale: 1.5, ghostOffset: 14
  };
  const MAP_COLORS = {
    // There is no recent-use ring; the Recent only filter selects recent points.
    folderEdge: "#7A5C00", dropEdge: "rgba(80,140,255,0.95)", selectedEdge: "#FF9800",
    neighbor: "rgba(127,127,127,0.6)", link: "rgba(80,140,255,0.85)", sphere: "rgba(127,127,127,0.35)",
    labelBack: "rgba(0,0,0,0.65)", labelText: "#FFFFFF"
  };
  const shownColor = MAP_VIEW.shownColor;
  const COLOR_CLASSES = [
    { id: "topic", label: "class.topic" }, { id: "doc", label: "class.doc" }, { id: "note", label: "class.note" },
    { id: "other", label: "class.other" }
  ];
  const UNFILED_SWATCH = "conic-gradient(" + COLOR_CLASSES.map(function (item) { return MAP_VIEW.CLASS_COLORS[item.id]; }).join(",") + ")";
  const MAP_KINDS = [
    { value: "memory", label: "kind.memory" }, { value: "human", label: "kind.human" },
    { value: "agent", label: "kind.agent" }, { value: "formula", label: "kind.formula" },
    { value: "task", label: "kind.task" }
  ];
  // Words for a map point's kind (not the raw value).
  const POINT_KIND_LABELS = { memory: "kind.memory", human: "kind.human", agent: "kind.agent", formula: "kind.formula",
    task: "kind.task", topic: "kind.topic" };
  function dotKindText(kind) { return POINT_KIND_LABELS[kind] ? t(POINT_KIND_LABELS[kind]) : String(kind || ""); }
  const UNFILED = "unfiled";

  function prefersDarkTheme() {
    const root = document.documentElement;
    if (root.classList.contains("dark")) return true;
    if (root.classList.contains("light")) return false;
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  // 3×3 rotation matrices, row-major arrays of 9 numbers.
  function unitMatrix3() { return [1, 0, 0, 0, 1, 0, 0, 0, 1]; }

  function multiply(a, b) {
    const out = new Array(9);
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        out[row * 3 + col] = a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
      }
    }
    return out;
  }

  function apply3(m, v) {
    return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
  }

  function rotationAxisAngle(axis, angle) {
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const x = axis[0], y = axis[1], z = axis[2];
    return [t * x * x + c, t * x * y - s * z, t * x * z + s * y,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c];
  }

  // Axis and angle that turn the viewed vector v to the viewer (0, 0, 1).
  function spinMatrixToFront(v) {
    const axis = [v[1], -v[0], 0];
    const length = Math.hypot(axis[0], axis[1]);
    const angle = Math.acos(Math.max(-1, Math.min(1, v[2])));
    if (length < MAP_TUNING.minAxis) return { axis: [0, 1, 0], angle: v[2] < 0 ? Math.PI : 0 };
    return { axis: [axis[0] / length, axis[1] / length, 0], angle: angle };
  }

  function nodeRadius(point) {
    if (point.lane === "memory" || MAP_VIEW.isTopic(point)) return MAP_TUNING.memoryRadius;
    return MAP_TUNING.noteBase + MAP_TUNING.noteLinkScale * Math.log2(1 + (point.backlinks || 0));
  }

  function trimHeading(text) {
    const value = String(text || "");
    return value.length > MAP_TUNING.titleChars ? value.slice(0, MAP_TUNING.titleChars - 1) + "…" : value;
  }

  function pathPointShape(g, point, x, y, r) {
    g.beginPath();
    if (point.lane === "memory" || MAP_VIEW.isTopic(point)) {
      g.arc(x, y, r, 0, 2 * Math.PI);
    } else if (point.kind === "task") {
      g.moveTo(x, y - r * 1.15); g.lineTo(x + r, y + r * 0.85); g.lineTo(x - r, y + r * 0.85); g.closePath();
    } else {
      g.rect(x - r, y - r, 2 * r, 2 * r);
    }
  }

  function pathFolderMarker(g, x, y, s) {
    const tab = s * 0.35;
    g.beginPath();
    g.moveTo(x - s, y - s * 0.7 - tab); g.lineTo(x - s * 0.1, y - s * 0.7 - tab); g.lineTo(x + s * 0.1, y - s * 0.7);
    g.lineTo(x + s, y - s * 0.7); g.lineTo(x + s, y + s * 0.7); g.lineTo(x - s, y + s * 0.7); g.closePath();
  }

  // Plain text beside an anchor, kept inside the canvas. Needs g.textBaseline = "top".
  function plotBody(g, text, anchorX, anchorY, offset, width, height) {
    const box = MAP_VIEW.placeLabel(anchorX, anchorY, g.measureText(text).width, MAP_TUNING.textHeight, offset, width, height, MAP_TUNING.labelPad);
    g.fillText(text, box.x, box.y);
  }

  // Hover tag (dark box), kept inside the canvas.
  function plotCaption(g, text, anchorX, anchorY, offset, width, height) {
    const boxWidth = g.measureText(text).width + 2 * MAP_TUNING.tagPadX;
    const box = MAP_VIEW.placeLabel(anchorX, anchorY, boxWidth, MAP_TUNING.tagHeight, offset, width, height, MAP_TUNING.labelPad);
    g.fillStyle = MAP_COLORS.labelBack;
    g.fillRect(box.x, box.y, boxWidth, MAP_TUNING.tagHeight);
    g.fillStyle = MAP_COLORS.labelText;
    g.fillText(text, box.x + MAP_TUNING.tagPadX, box.y + MAP_TUNING.tagPadY);
  }

  function fitsTerm(point, query) {
    return !query || String(point.title || "").toLowerCase().indexOf(query) >= 0;
  }

  function dateLabel(seconds) { return seconds ? new Date(seconds * 1000).toLocaleDateString() : ""; }

  function BinSidebar(props) {
    const [newName, setNewName] = useState("");
    const [renaming, setRenaming] = useState(null);
    const [pendingDelete, setPendingDelete] = useState(null);
    const [busy, setBusy] = useState(false);
    const dark = prefersDarkTheme();

    function run(request, text) {
      setBusy(true);
      return request.then(function () {
        setBusy(false); props.onMessage({ ok: true, text: text }); props.onChanged(); return true;
      }).catch(function (e) { setBusy(false); props.onMessage({ ok: false, text: apiFault(e).text }); return false; });
    }

    function create(event) {
      event.preventDefault();
      if (!newName.trim()) { props.onMessage({ ok: false, text: t("folder.needName") }); return; }
      run(SDK.fetchJSON(API + "/folders", jsonInit("POST", { name: newName })), t("folder.created")).then(function (ok) { if (ok) setNewName(""); });
    }

    function rename(event) {
      event.preventDefault();
      run(SDK.fetchJSON(API + "/folders/" + renaming.id, jsonInit("PUT", { name: renaming.name })), t("folder.renamed")).then(function (ok) { if (ok) setRenaming(null); });
    }

    function recolor(folder, color) {
      run(SDK.fetchJSON(API + "/folders/" + folder.id, jsonInit("PUT", { color: color })), t("folder.recolored"));
    }

    function remove() {
      const folder = pendingDelete;
      run(SDK.fetchJSON(API + "/folders/" + folder.id, { method: "DELETE" }), t("folder.trashed")).then(function (ok) {
        setPendingDelete(null);
        if (ok) props.onDeleted(folder.id);
      });
    }

    // While an item drag is on, every row is a drop target; the one under the cursor is marked.
    function releaseClass(target) {
      if (!props.dragging) return "";
      return props.dropTarget === target ? " is-drop-target" : " is-droppable";
    }

    function binLine(folder) {
      const active = props.focus === folder.id;
      const isRenaming = renaming && renaming.id === folder.id;
      return h("li", { key: folder.id, className: "neospike-folder" + (active ? " is-selected" : "") + releaseClass(String(folder.id)),
        "data-drop-folder": String(folder.id) },
        h("button", { type: "button", className: "neospike-folder-button", onClick: function () { props.onSelect(folder); } },
          h("span", { className: "neospike-folder-icon", style: { "--neospike-folder": MAP_VIEW.folderMarkerColor(folder) }, title: t("folder.lastUsed", { recency: folder.recency || "old" }), "aria-hidden": true }),
          h("span", { className: "neospike-folder-name" }, folder.name),
          h("span", { className: "neospike-color-dot", style: { background: shownColor(folder.color, dark) }, "aria-hidden": true }),
          h("span", { className: "neospike-muted" }, String(folder.count))
        ),
        active && !isRenaming ? h("div", { className: "neospike-actions" },
          h(Button, { type: "button", size: "xs", outlined: true, disabled: busy, onClick: function () { setRenaming({ id: folder.id, name: folder.name }); } }, t("folder.rename")),
          h("input", { type: "color", className: "neospike-color-input", value: folder.color.toLowerCase(), disabled: busy, "aria-label": t("folder.colour"),
            onChange: function (event) { recolor(folder, event.target.value); } }),
          h(Button, { type: "button", size: "xs", destructive: true, outlined: true, disabled: busy, onClick: function () { setPendingDelete(folder); } }, t("common.delete"))
        ) : null,
        isRenaming ? h("form", { className: "neospike-actions", onSubmit: rename },
          h(Input, { value: renaming.name, "aria-label": t("folder.name"), onChange: function (e) { setRenaming({ id: folder.id, name: e.target.value }); } }),
          h(Button, { type: "submit", size: "xs", disabled: busy || !renaming.name.trim() }, t("common.save")),
          h(Button, { type: "button", size: "xs", outlined: true, onClick: function () { setRenaming(null); } }, t("common.cancel"))
        ) : null
      );
    }

    return h("div", { className: "neospike-map-sidebar neospike-stack" },
      h("form", { className: "neospike-actions", onSubmit: create },
        h(Input, { value: newName, placeholder: t("folder.name"), "aria-label": t("folder.newName"), onChange: function (e) { setNewName(e.target.value); } }),
        h(Button, { type: "submit", size: "sm", disabled: busy }, t("folder.new"))
      ),
      props.error ? h("p", { className: "neospike-message is-error", role: "alert" }, t("folder.loadFailed", { error: props.error })) : null,
      h("ul", { className: "neospike-folder-list" },
        props.folders.map(binLine),
        h("li", { key: UNFILED, className: "neospike-folder" + (props.focus === UNFILED ? " is-selected" : "") + releaseClass(UNFILED),
          "data-drop-folder": UNFILED },
          h("button", { type: "button", className: "neospike-folder-button", onClick: function () { props.onSelect(null); } },
            h("span", { className: "neospike-color-dot", style: { background: UNFILED_SWATCH }, "aria-hidden": true }),
            h("span", { className: "neospike-folder-name" }, t("common.unfiled")),
            h("span", { className: "neospike-muted" }, String(props.unfiledCount))
          )
        )
      ),
      h(ConfirmDialog, {
        open: pendingDelete !== null, title: t("folder.trashTitle"), description: t("folder.trashCopy"),
        confirmLabel: t("common.delete"), destructive: true, loading: busy, onConfirm: remove, onCancel: function () { setPendingDelete(null); }
      })
    );
  }

  function MemoryFetcher(props) {
    const [item, setItem] = useState(null);
    const [failure, setFailure] = useState(null);
    const load = useCallback(function () {
      return SDK.fetchJSON(API + "/memories/" + props.id).then(function (data) { setItem(data); setFailure(null); })
        .catch(function (e) { setItem(null); setFailure(apiFault(e).text); });
    }, [props.id]);
    useEffect(function () { load(); }, [load, props.seq]);
    function onChanged(removed) { if (!removed) load(); props.onChanged(removed); }
    if (failure) return h("p", { className: "neospike-muted" }, failure);
    if (!item) return h("p", { className: "neospike-muted" }, t("common.loading"));
    return h("ul", { className: "neospike-item-list" }, h(MemoryEntryCard, { item: item, onChanged: onChanged, onMessage: props.onMessage,
      beforeActions: props.beforeActions, besideActions: props.besideActions }));
  }

  // Items linked to the panel's item (refers both ways, supersedes both ways, topic membership).
  // GET /connections reads the same edges as the map's lines. props.version changes on every map reload.
  const CONNECTED_KIND_LABELS = { topic: "kind.topic", note: "kind.note", memory: "kind.memory" };
  // Server relation labels (plugin_api._RELATION_LABELS) -> text keys, including the piece chain.
  const CONNECTED_RELATION_LABELS = {
    "refers": "connected.rel.refers", "referenced by": "connected.rel.referencedBy", "supersedes": "connected.rel.supersedes",
    "superseded by": "connected.rel.supersededBy", "topic member": "connected.rel.topicMember",
    "next piece": "connected.rel.nextPiece", "previous piece": "connected.rel.previousPiece"
  };

  function edgeRelationText(relation) { return CONNECTED_RELATION_LABELS[relation] ? t(CONNECTED_RELATION_LABELS[relation]) : relation; }

  // One GET /connections per opened item; the Connected list and the connected-text box read the same result.
  function useConnectionList(lane, id, seq, version) {
    const [result, setResult] = useState(null);
    const [failure, setFailure] = useState(null);
    useEffect(function () {
      let current = true;
      function handleLoaded(data) { if (current) { setResult(data); setFailure(null); } }
      function onFailed(e) { if (current) { setResult(null); setFailure(apiFault(e).text); } }
      setResult(null);
      SDK.fetchJSON(API + "/connections/" + lane + "/" + id).then(handleLoaded).catch(onFailed);
      return function () { current = false; };
    }, [lane, id, seq, version]);
    return { result: result, failure: failure };
  }

  // Read-only box with every connected item's DB text, below Save/Delete, above Connected.
  function LinkedBodyBox(props) {
    const items = props.result ? props.result.items : [];
    return h("textarea", { className: "neospike-editor", readOnly: true, rows: 10, spellCheck: false, value: MAP_VIEW.connectedText(items),
      placeholder: props.result ? t("connected.none") : t("common.loading"), "aria-label": t("connected.textLabel"),
      style: { width: "100%", boxSizing: "border-box" } });
  }

  function themeShiftNotice(result, verbKey) {
    const skipped = (result.skipped_ids || []).length + (result.skipped_notes || []).length;
    return t(verbKey, { n: result.moved }) + (skipped ? t("topic.skipped", { n: skipped }) : "");
  }

  // Move to... in a topic's panel: out of the topic, into another topic, or into a folder (also out of the topic).
  const TOPIC_REMOVE = "remove";
  function ThemeShiftBar(props) {
    const [target, setTarget] = useState("");
    const [topics, setTopics] = useState([]);
    const [busy, setBusy] = useState(false);
    function fetchTopics() { SDK.fetchJSON(API + "/topics").then(function (data) { setTopics(data.topics || []); }).catch(function () { setTopics([]); }); }
    useEffect(fetchTopics, []);
    function handleDestChange(event) { setTarget(event.target.value); }
    function move() {
      const payload = { items: props.keys };
      if (target === TOPIC_REMOVE) payload.remove_from_topic = props.topicId;
      else if (MAP_VIEW.topicIdOf(target) !== null) payload.to_topic = MAP_VIEW.topicIdOf(target);
      else { payload.remove_from_topic = props.topicId; payload.to_folder = true; payload.folder_id = target === UNFILED ? null : Number(target); }
      setBusy(true);
      return SDK.fetchJSON(API + "/topics/members/move", jsonInit("POST", payload)).then(function (result) {
        setBusy(false); setTarget("");
        props.onMessage({ ok: true, text: themeShiftNotice(result, "topic.moved") });
        props.onMoved();
      }).catch(function (e) { setBusy(false); props.onMessage({ ok: false, text: apiFault(e).text }); });
    }
    const others = topics.filter(function (topic) { return topic.id !== props.topicId; });
    return h("div", { className: "neospike-actions neospike-move-bar" },
      h("select", { className: "neospike-select", value: target, onChange: handleDestChange, onFocus: fetchTopics, "aria-label": t("topic.moveTo") },
        h("option", { value: "" }, t("topic.moveToPick")),
        h("option", { value: TOPIC_REMOVE }, t("topic.remove")),
        others.length ? h("optgroup", { label: t("topic.topics") }, others.map(function (topic) {
          return h("option", { key: "t" + topic.id, value: MAP_VIEW.dropTopicKey(topic.id) }, topic.title);
        })) : null,
        h("optgroup", { label: t("topic.folders") },
          h("option", { value: UNFILED }, t("common.unfiled")),
          (props.folders || []).map(function (folder) { return h("option", { key: "f" + folder.id, value: String(folder.id) }, folder.name); }))
      ),
      h(Button, { type: "button", size: "sm", outlined: true, disabled: busy || !props.keys.length || !target, onClick: move },
        t("common.move") + (props.keys.length ? " (" + props.keys.length + ")" : "")));
  }

  // A topic's panel is its title (read-only: no Save or Delete) and its members as the Connected list.
  function ThemeEntry(props) {
    const result = props.connections.result;
    return h("div", { className: "neospike-stack" },
      h("span", { className: "neospike-muted" }, t("kind.topic")),
      h("strong", { "data-topic-title": String(props.id) }, result ? result.title : t("common.loading")),
      h(LinkedRoll, { result: result, failure: props.connections.failure, lane: MAP_VIEW.TOPIC_LANE, id: props.id,
        onMap: props.onMap, onOpen: props.onOpen, folders: props.folders, onMoved: props.onMoved, onTopicMoved: props.onMoved,
        onMessage: props.onMessage, dropTarget: props.dropTarget }));
  }

  // The opened item. No hooks here; a topic and an item are separate components, so switching
  // between them remounts instead of changing the hook count.
  function ViewEntry(props) {
    return h(props.panel.type === MAP_VIEW.TOPIC_LANE ? ThemeView : RecordView, props);
  }

  function ThemeView(props) {
    const panel = props.panel;
    const connections = useConnectionList(panel.type, panel.id, panel.seq, props.version);
    return h(ThemeEntry, { id: panel.id, connections: connections, onMap: props.onMap, onOpen: props.onOpen, folders: props.folders,
      onMoved: props.onMoved, onMessage: props.onMessage, dropTarget: props.dropTarget });
  }

  // A memory or note: body, Save/Delete + folder Move, connected text, then Connected.
  function RecordView(props) {
    const panel = props.panel;
    const connections = useConnectionList(panel.type, panel.id, panel.seq, props.version);
    const [reloads, setReloads] = useState(0);
    const textBox = h(LinkedBodyBox, { result: connections.result });
    const moveBar = h(ViewShiftBar, { keys: [rowKey(panel.type, panel.id)], folders: props.folders, onMove: props.onMove });
    function onTopicMoved() { setReloads(reloads + 1); props.onMoved(); }
    const editor = panel.type === "note"
      ? h(MemoEditor, { request: { id: panel.id, seq: panel.seq + reloads }, onChanged: props.onNoteChanged, showSide: false,
        besideActions: moveBar })
      : h(MemoryFetcher, { id: panel.id, seq: panel.seq, onChanged: props.onItemChanged, onMessage: props.onMessage,
        besideActions: moveBar });
    return h("div", { className: "neospike-stack" }, editor, textBox,
      h(LinkedRoll, { result: connections.result, failure: connections.failure, lane: panel.type, id: panel.id,
        onMap: props.onMap, onOpen: props.onOpen, folders: props.folders, onMoved: props.onMoved, onTopicMoved: onTopicMoved,
        onMessage: props.onMessage, dropTarget: props.dropTarget }));
  }

  function LinkedRoll(props) {
    const result = props.result;
    const failure = props.failure;
    const topicId = result && result.is_topic ? props.id : null;
    // Rows are checkable; Select all and the Move bar sit above the list.
    // In a topic's panel the bar is Move to... (topic members), and the list takes dropped points.
    const [stored, setChecked] = useState([]);
    useEffect(function () { setChecked([]); }, [props.lane, props.id]);
    const items = result ? result.items : [];
    const checked = MAP_VIEW.keptKeys(stored, items);
    const allChecked = items.length > 0 && checked.length === items.length;
    function selectOrClearAll() { setChecked(allChecked ? [] : items.map(neighborKey)); }
    function onMoved() { setChecked([]); props.onMoved(); }
    function onTopicMoved() { setChecked([]); props.onTopicMoved(); }
    function row(item) {
      function open() { props.onOpen(item); }
      function toggle() { setChecked(keysAfterToggle(checked, item.key)); }
      return h("li", { key: item.key, className: "neospike-checkable" },
        h("input", { type: "checkbox", checked: checked.indexOf(item.key) >= 0, onChange: toggle, "aria-label": t("connected.select", { key: item.key }) }),
        h("button", { type: "button", className: "neospike-connected-item", onClick: open, "data-connected-key": item.key },
          h("span", { className: "neospike-connected-kind" }, CONNECTED_KIND_LABELS[item.kind] ? t(CONNECTED_KIND_LABELS[item.kind]) : item.kind),
          h("span", { className: "neospike-connected-title" }, item.title),
          h("span", { className: "neospike-muted" }, item.relations.map(edgeRelationText).join(", ") + (props.onMap(item.key) ? "" : t("connected.notOnMap")))));
    }
    const dropping = topicId !== null && props.dropTarget === MAP_VIEW.dropTopicKey(topicId);
    return h("div", { className: "neospike-stack neospike-connected", "data-drop-topic": topicId === null ? undefined : String(topicId),
      style: dropping ? { outline: "2px dashed var(--color-ring)", outlineOffset: "2px" } : undefined },
      h("strong", null, t("connected.head", { n: result ? items.length : "…" }) + (topicId !== null ? t("connected.dropHint") : "")),
      items.length ? h("div", { className: "neospike-actions" },
        h("label", { className: "neospike-toggle" },
          h("input", { type: "checkbox", checked: allChecked, onChange: selectOrClearAll, "aria-label": t("connected.selectAllLabel") }), t("connected.selectAll")),
        topicId !== null
          ? h(ThemeShiftBar, { keys: checked, topicId: topicId, folders: props.folders, onMoved: onTopicMoved, onMessage: props.onMessage })
          : h(BinShiftBar, { keys: checked, folders: props.folders, onMoved: onMoved, onMessage: props.onMessage })) : null,
      failure ? h("p", { className: "neospike-muted" }, failure)
        : !result ? h("p", { className: "neospike-muted" }, t("common.loading"))
          : items.length ? h("ul", { className: "neospike-connected-list" }, items.map(row))
            : h("p", { className: "neospike-muted" }, t("connected.none")));
  }

  function neighborKey(item) { return item.key; }

  function PartRoll(props) {
    const [stored, setChecked] = useState([]);
    useEffect(function () { setChecked([]); }, [props.title]);
    const checked = MAP_VIEW.keptKeys(stored, props.members);
    function onMoved() { setChecked([]); props.onChanged(); }
    return h("div", { className: "neospike-stack" },
      h("strong", null, props.title + " · " + props.members.length),
      props.onToggleExpand ? h("label", { className: "neospike-toggle" },
        h("input", { type: "checkbox", checked: !!props.expanded, onChange: props.onToggleExpand }), t("member.expand")) : null,
      h(BinShiftBar, { keys: checked, folders: props.folders, onMoved: onMoved, onMessage: props.onMessage }),
      props.members.length ? h("ul", { className: "neospike-member-list" }, props.members.map(function (point) {
        return h("li", { key: point.key, className: "neospike-checkable" },
          h("input", { type: "checkbox", checked: checked.indexOf(point.key) >= 0, "aria-label": t("common.selectTitle", { title: point.title }),
            onChange: function () { setChecked(keysAfterToggle(checked, point.key)); } }),
          h("button", { type: "button", className: "neospike-note-item", onClick: function () { props.onOpen(point); } },
            h("span", { className: "neospike-choice-title" }, point.title || point.key),
            h("span", { className: "neospike-muted" }, dotKindText(point.kind) + (point.superseded ? t("member.superseded") : "") + " · " + dateLabel(point.updated_at))
          )
        );
      })) : h("p", { className: "neospike-muted" }, t("member.empty"))
    );
  }

  // What the shapes and colours mean. HTML over the canvas, so it costs no redraw.
  const LEGEND_SHAPES = [
    { glyph: "●", label: "kind.memory" }, { glyph: "■", label: "kind.note" }, { glyph: "▲", label: "kind.task" },
    { glyph: "○", label: "legend.superseded" }
  ];
  const LEGEND_RECENCY = [{ id: "recent", label: "legend.recent" }, { id: "mid", label: "legend.mid" }, { id: "old", label: "legend.old" }];

  function legendSwatch(color, label) {
    return h("li", { key: label }, h("span", { className: "neospike-legend-swatch", style: { background: color }, "aria-hidden": true }), label);
  }

  function legendShape(shape) {
    return h("li", { key: shape.label }, h("span", { className: "neospike-legend-glyph", "aria-hidden": true }, shape.glyph), t(shape.label));
  }

  function legendClass(item) { return legendSwatch(MAP_VIEW.CLASS_COLORS[item.id], t("legend.unfiled", { label: t(item.label) })); }

  function legendRecency(item) {
    return h("li", { key: item.id },
      h("span", { className: "neospike-folder-icon", style: { "--neospike-folder": MAP_VIEW.RECENCY_COLORS[item.id] }, "aria-hidden": true }),
      t("legend.folderUsed", { label: t(item.label) }));
  }

  function MapColorKey(props) {
    const toggle = h("button", { type: "button", className: "neospike-legend-toggle", "aria-expanded": !!props.open, onClick: props.onToggle },
      props.open ? t("legend.open") : t("legend.closed"));
    if (!props.open) return h("div", { className: "neospike-legend" }, toggle);
    return h("div", { className: "neospike-legend" }, toggle,
      h("ul", null,
        LEGEND_SHAPES.map(legendShape),
        COLOR_CLASSES.map(legendClass),
        LEGEND_RECENCY.map(legendRecency)
      ));
  }

  function MapView(props) {
    const [data, setData] = useState({ points: [], folders: [] });
    const [loaded, setLoaded] = useState(false);
    const [loadError, setLoadError] = useState(null);
    const [folderList, setFolderList] = useState([]);
    const [folderError, setFolderError] = useState(null);
    const [message, setMessage] = useState(null);
    const [query, setQuery] = useState("");
    // Topic points have no kind checkbox; they are always shown.
    const [kinds, setKinds] = useState({ memory: true, human: true, agent: true, formula: true, task: true, topic: true });
    const [recentOnly, setRecentOnly] = useState(false);
    // focus: the folder chosen in the sidebar (id, UNFILED or null). panel: the opened item { type: lane, id, seq } or null.
    const [focus, setFocus] = useState(null);
    const [panel, setPanel] = useState(null);
    const [expanded, setExpanded] = useState({});
    const [zoomLevel, setZoomLevel] = useState(1);
    const [legendOpen, setLegendOpen] = useState(false);
    // Selected point keys (Ctrl/Shift+click) and the item drag shown in the sidebar.
    const [selection, setSelection] = useState([]);
    const [dragging, setDragging] = useState(false);
    const [dropTarget, setDropTarget] = useState(null);
    const canvasRef = useRef(null);
    const wrapRef = useRef(null);
    const ghostRef = useRef(null);
    const view = useRef({ rotation: unitMatrix3(), zoom: 1, panX: 0, panY: 0, width: 0, height: 0 });
    const drag = useRef(null);
    const dropRef = useRef(null);
    const hover = useRef(null);
    const animation = useRef(null);
    const latest = useRef(null);
    const paintRef = useRef(null);
    // One binding for the component's life. Its callbacks read refs, so they always see the newest render.
    const binding = useRef(null);
    if (binding.current === null) binding.current = MAP_VIEW.createCanvasBinding(measureCanvasBox, scheduleFrame, scheduleTimer, repaintFromState);

    function scheduleFrame(callback) { return window.requestAnimationFrame(callback); }
    function scheduleTimer(callback) { return window.setTimeout(callback, MAP_TUNING.frameFallbackMs); }
    function repaintFromState(size) { if (paintRef.current) paintRef.current(size); }
    function scheduleRepaint() { binding.current.onInteraction(); }

    // The canvas is sized by CSS (100% of its box); this reads that box. A hidden tab reads 0 and draws nothing.
    function measureCanvasBox() {
      const canvas = canvasRef.current;
      view.current.width = canvas ? canvas.clientWidth : 0;
      view.current.height = canvas ? canvas.clientHeight : 0;
      return { width: view.current.width, height: view.current.height, ratio: window.devicePixelRatio || 1 };
    }

    // The folder list comes from its own read-only route, so a failed or slow /map
    // (a CLI child) never hides the folders or the New folder box.
    // Only the newest load applies its answers: an older /map that answers late never brings a deleted item back.
    const loadSeq = useRef(0);
    const load = useCallback(function () {
      const seq = ++loadSeq.current;
      function isLatest() { return seq === loadSeq.current; }
      SDK.fetchJSON(API + "/folders").then(function (payload) { if (isLatest()) { setFolderList(payload.folders || []); setFolderError(null); } })
        .catch(function (e) { if (isLatest()) setFolderError(apiFault(e).text); });
      return SDK.fetchJSON(API + "/map").then(function (payload) { if (isLatest()) { setData(payload); setLoadError(null); setLoaded(true); } })
        .catch(function (e) { if (isLatest()) { setData({ points: [], folders: [] }); setLoadError(apiFault(e).text); setLoaded(true); } });
    }, []);
    // Map folders carry their sphere position; until /map succeeds the sidebar uses the plain list.
    const sidebarFolders = loaded && !loadError ? data.folders : folderList;

    // Re-entering the tab reloads, so an earlier "Moved N…" notice no longer describes the screen.
    useEffect(function () { if (props.active) { setMessage(null); load(); } }, [props.active, load]);

    const byKey = {};
    data.points.forEach(function (point, index) { byKey[point.key] = index; });
    const selectedFlags = {};
    selection.forEach(function (key) { selectedFlags[key] = true; });
    latest.current = { data: data, byKey: byKey, query: query.trim().toLowerCase(), kinds: kinds, recentOnly: recentOnly, focus: focus,
      selected: selectedFlags, opened: panel ? rowKey(panel.type, panel.id) : null, drawn: MAP_VIEW.drawnFlags(data.points, MAP_VIEW.expandedWithFocus(expanded, focus)) };

    function nodeConcealed(index, state) {
      const point = state.data.points[index];
      return !state.drawn[index] || !state.kinds[point.kind] || (state.recentOnly && !point.recent);
    }

    // Only the title filter dims. A selected folder highlights its members instead of dimming the rest.
    function nodeFaded(point, state) { return !fitsTerm(point, state.query); }

    function isTopicMember(point, state) {
      if (state.focus === null) return false;
      return state.focus === UNFILED ? point.folder_id === null && !MAP_VIEW.isTopic(point) : point.folder_id === state.focus;
    }

    function project(v) {
      const state = view.current;
      const p = apply3(state.rotation, v);
      const radius = Math.min(state.width, state.height) * MAP_TUNING.sphereRatio * state.zoom;
      return { x: state.width / 2 + state.panX + p[0] * radius, y: state.height / 2 + state.panY - p[1] * radius, z: p[2] };
    }

    function folderPalette(state) {
      const colors = {};
      state.data.folders.forEach(function (folder) { colors[folder.id] = folder.color; });
      return colors;
    }

    // Called only by the scheduler, with a non-zero size. The backing store follows the CSS size x devicePixelRatio.
    function draw(size) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const state = latest.current;
      const geometry = view.current;
      const ratio = size.ratio;
      const backWidth = Math.round(size.width * ratio), backHeight = Math.round(size.height * ratio);
      if (canvas.width !== backWidth) canvas.width = backWidth;
      if (canvas.height !== backHeight) canvas.height = backHeight;
      const g = canvas.getContext("2d");
      g.setTransform(ratio, 0, 0, ratio, 0, 0);
      g.clearRect(0, 0, geometry.width, geometry.height);
      g.font = MAP_TUNING.font;
      g.textBaseline = "top";
      const dark = prefersDarkTheme();
      const colors = folderPalette(state);
      const foreground = window.getComputedStyle(canvas).color;
      const screen = state.data.points.map(function (point) { return project([point.x, point.y, point.z]); });
      const order = screen.map(function (_, index) { return index; }).sort(function (a, b) { return screen[a].z - screen[b].z; });
      const itemDrag = drag.current && drag.current.mode === "item" && drag.current.moved;
      const hovered = !itemDrag && hover.current && hover.current.type === "point" ? hover.current.index : null;
      const sphereRadius = Math.min(geometry.width, geometry.height) * MAP_TUNING.sphereRatio * geometry.zoom;
      const titles = [];
      const lines = [];
      let outlined = false;

      function outline() {
        g.globalAlpha = 1;
        g.strokeStyle = MAP_COLORS.sphere; g.lineWidth = 1;
        g.beginPath(); g.arc(geometry.width / 2 + geometry.panX, geometry.height / 2 + geometry.panY, sphereRadius, 0, 2 * Math.PI); g.stroke();
        outlined = true;
      }

      if (hovered !== null) {
        const from = screen[hovered];
        const point = state.data.points[hovered];
        g.lineWidth = 1;
        point.neighbors.forEach(function (key) {
          const target = state.byKey[key];
          if (target === undefined || nodeConcealed(target, state)) return;
          g.strokeStyle = MAP_COLORS.neighbor; g.setLineDash([4, 3]);
          g.beginPath(); g.moveTo(from.x, from.y); g.lineTo(screen[target].x, screen[target].y); g.stroke();
        });
        g.setLineDash([]);
        plotEdges(hovered);
      }
      // The item open in the side panel keeps its link lines, the same edges as its Connected list.
      const opened = state.opened === null ? undefined : state.byKey[state.opened];
      if (opened !== undefined && opened !== hovered && !nodeConcealed(opened, state)) plotEdges(opened);

      function plotEdges(index) {
        const from = screen[index];
        state.data.points[index].links.forEach(function (key) {
          const target = state.byKey[key];
          if (target === undefined || nodeConcealed(target, state)) return;
          lines.push({ from: state.data.points[index].key, to: key });
          g.strokeStyle = MAP_COLORS.link; g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(from.x, from.y); g.lineTo(screen[target].x, screen[target].y); g.stroke();
        });
      }

      order.forEach(function (index) {
        const point = state.data.points[index];
        if (nodeConcealed(index, state)) return;
        const at = screen[index];
        if (!outlined && at.z >= 0) outline();
        const back = at.z < 0;
        const selected = state.selected[point.key] === true || point.key === state.opened;
        const member = isTopicMember(point, state);
        const dim = nodeFaded(point, state) && index !== hovered && !selected;
        g.globalAlpha = dim ? MAP_TUNING.dimAlpha : back && !member && !selected ? MAP_TUNING.backAlpha : 1;
        const color = MAP_VIEW.pointColor(point, colors, dark);
        const r = nodeRadius(point) * (member ? MAP_TUNING.memberScale : 1);
        pathPointShape(g, point, at.x, at.y, r);
        if (point.superseded) { g.strokeStyle = color; g.lineWidth = MAP_TUNING.hollowWidth; g.stroke(); }
        else { g.fillStyle = color; g.fill(); }
        if (selected) { g.strokeStyle = MAP_COLORS.selectedEdge; g.lineWidth = MAP_TUNING.selectedEdgeWidth; g.stroke(); }
        if (itemDrag && point.topic && dropRef.current === MAP_VIEW.dropTopicKey(point.id)) {
          g.strokeStyle = MAP_COLORS.dropEdge; g.lineWidth = MAP_TUNING.selectedEdgeWidth; g.stroke();
        }
        if (!dim && geometry.zoom >= MAP_TUNING.titleZoom && MAP_VIEW.titleVisible(at.x, at.y, at.z, geometry.width, geometry.height)) {
          titles.push({ key: point.key, x: at.x, y: at.y, z: at.z });
          g.fillStyle = foreground;
          plotBody(g, trimHeading(point.title), at.x, at.y, r + MAP_TUNING.labelGap / 2, geometry.width, geometry.height);
        }
      });
      if (!outlined) outline();
      // Read-only snapshot of this frame for the browser check: drawn points, title anchors and the view.
      canvas.neospikeFrame = { titles: titles, lines: lines, width: geometry.width, height: geometry.height,
        view: { rotation: geometry.rotation.slice(), panX: geometry.panX, panY: geometry.panY, zoom: geometry.zoom },
        points: order.filter(function (index) { return !nodeConcealed(index, state); }).map(function (index) {
          return { key: state.data.points[index].key, folder: state.data.points[index].folder_id, x: screen[index].x, y: screen[index].y, z: screen[index].z };
        }) };

      g.globalAlpha = 1;
      state.data.folders.forEach(function (folder) {
        if (folder.x === null) return;
        const at = project([folder.x, folder.y, folder.z]);
        const active = state.focus === folder.id;
        const target = itemDrag && dropRef.current === String(folder.id);
        g.globalAlpha = at.z < 0 && !itemDrag ? MAP_TUNING.backAlpha : 1;
        pathFolderMarker(g, at.x, at.y, MAP_TUNING.folderSize * (target ? MAP_TUNING.dropMarkerScale : 1));
        g.fillStyle = MAP_VIEW.folderMarkerColor(folder); g.fill();
        g.strokeStyle = itemDrag ? MAP_COLORS.dropEdge : MAP_COLORS.folderEdge; g.lineWidth = active || itemDrag ? 2 : 1; g.stroke();
        if (at.z >= 0 || itemDrag) {
          g.fillStyle = foreground;
          plotBody(g, folder.name, at.x, at.y, MAP_TUNING.folderSize + MAP_TUNING.labelGap, geometry.width, geometry.height);
        }
      });
      g.globalAlpha = 1;

      if (hover.current && !itemDrag) {
        const item = hover.current.type === "point" ? state.data.points[hover.current.index] : hover.current.folder;
        const at = hover.current.type === "point" ? screen[hover.current.index]
          : project([hover.current.folder.x, hover.current.folder.y, hover.current.folder.z]);
        const label = hover.current.type === "point" ? trimHeading(item.title) : item.name + " · " + item.count;
        plotCaption(g, label, at.x, at.y - MAP_TUNING.folderSize, MAP_TUNING.folderSize + MAP_TUNING.labelGap, geometry.width, geometry.height);
      }
    }

    paintRef.current = draw;
    // The one draw request for state. Every render (data, filter, focus, expansion, tab shown, panel)
    // re-reads the canvas size and paints now; a 0-sized canvas defers and onResize paints when the size arrives.
    useEffect(function () { binding.current.onRender(); });

    useEffect(function () {
      function onResize() { binding.current.onResize(); }
      if (!window.ResizeObserver || !canvasRef.current) {
        window.addEventListener("resize", onResize);
        return function () { window.removeEventListener("resize", onResize); };
      }
      const observer = new window.ResizeObserver(onResize);
      observer.observe(canvasRef.current);
      return function () { observer.disconnect(); };
    }, []);

    function applyZoomLevel(next) {
      view.current.zoom = Math.max(MAP_TUNING.zoomMin, Math.min(MAP_TUNING.zoomMax, next));
      setZoomLevel(view.current.zoom);
      scheduleRepaint();
    }

    useEffect(function () {
      const canvas = canvasRef.current;
      if (!canvas) return undefined;
      function handleScroll(event) {
        event.preventDefault();
        applyZoomLevel(view.current.zoom * Math.exp(-event.deltaY * MAP_TUNING.wheelScale));
      }
      canvas.addEventListener("wheel", handleScroll, { passive: false });
      return function () { canvas.removeEventListener("wheel", handleScroll); };
    }, []);

    function folderAtPoint(x, y) {
      let best = null;
      latest.current.data.folders.forEach(function (folder) {
        if (folder.x === null) return;
        const at = project([folder.x, folder.y, folder.z]);
        const distance = Math.hypot(at.x - x, at.y - y);
        if (distance <= MAP_TUNING.folderSize + MAP_TUNING.hitSlop && (!best || at.z > best.z)) best = { type: "folder", folder: folder, z: at.z + 2 };
      });
      return best;
    }

    function pickPointAt(x, y) {
      const state = latest.current;
      const marker = folderAtPoint(x, y);
      let best = null;
      state.data.points.forEach(function (point, index) {
        if (nodeConcealed(index, state)) return;
        const at = project([point.x, point.y, point.z]);
        const distance = Math.hypot(at.x - x, at.y - y);
        if (distance > nodeRadius(point) + MAP_TUNING.pointHitSlop) return;
        if (MAP_VIEW.closerHit(best, distance, at.z)) best = { type: "point", index: index, z: at.z, distance: distance };
      });
      if (best === null || marker === null) return best || marker;
      return MAP_VIEW.memberBeatsMarker(marker, state.data.points[best.index]) || best.z > marker.z ? best : marker;
    }

    function canvasPoint(event) {
      const box = canvasRef.current.getBoundingClientRect();
      return { x: event.clientX - box.left, y: event.clientY - box.top };
    }

    function cancelSpin() {
      if (animation.current) { window.cancelAnimationFrame(animation.current); animation.current = null; }
    }

    // A press on a point starts an item drag (the sphere stays still); a press elsewhere rotates or pans.
    function onPointerDown(event) {
      cancelSpin();
      const at = canvasPoint(event);
      const hit = pickPointAt(at.x, at.y);
      const point = hit && hit.type === "point" ? latest.current.data.points[hit.index] : null;
      drag.current = { mode: point ? "item" : "view", x: at.x, y: at.y, startX: at.x, startY: at.y, pan: event.shiftKey, moved: false,
        additive: event.ctrlKey || event.metaKey || event.shiftKey, point: point, keys: [] };
      if (canvasRef.current.setPointerCapture) canvasRef.current.setPointerCapture(event.pointerId);
    }

    // Sidebar row under the cursor (data-drop-folder), else a folder marker on the sphere.
    // Or a topic — its panel list (data-drop-topic) or its point on the sphere (not one being dragged).
    function findReleaseDest(event, at, keys) {
      const element = document.elementFromPoint(event.clientX, event.clientY);
      const row = element && element.closest ? element.closest("[data-drop-folder]") : null;
      if (row) return row.getAttribute("data-drop-folder");
      const topicArea = element && element.closest ? element.closest("[data-drop-topic]") : null;
      if (topicArea) return MAP_VIEW.dropTopicKey(topicArea.getAttribute("data-drop-topic"));
      if (element !== canvasRef.current) return null;
      const marker = folderAtPoint(at.x, at.y);
      if (marker) return String(marker.folder.id);
      const hit = pickPointAt(at.x, at.y);
      const point = hit && hit.type === "point" ? latest.current.data.points[hit.index] : null;
      return point && point.topic && keys.indexOf(point.key) < 0 ? MAP_VIEW.dropTopicKey(point.id) : null;
    }

    function shiftGhost(event, text) {
      const ghost = ghostRef.current;
      if (!ghost) return;
      if (text !== null) ghost.textContent = text;
      ghost.style.left = (event.clientX + MAP_TUNING.ghostOffset) + "px";
      ghost.style.top = (event.clientY + MAP_TUNING.ghostOffset) + "px";
      ghost.hidden = false;
    }

    function markDropTarget(target) {
      if (dropRef.current === target) return;
      dropRef.current = target;
      setDropTarget(target);
      scheduleRepaint();
    }

    function handleEntryDragShift(event, at, current) {
      // A topic point is not an item; it can be clicked (panel) and dropped on, never dragged.
      if (MAP_VIEW.isTopic(current.point)) {
        if (MAP_VIEW.isDragMove(current.startX, current.startY, at.x, at.y, MAP_TUNING.clickSlop)) current.moved = true;
        return;
      }
      if (!current.moved) {
        if (!MAP_VIEW.isDragMove(current.startX, current.startY, at.x, at.y, MAP_TUNING.clickSlop)) return;
        current.moved = true;
        current.keys = MAP_VIEW.dragKeys(selection, current.point.key);
        hover.current = null;
        canvasRef.current.style.cursor = "grabbing";
        setDragging(true);
        shiftGhost(event, MAP_VIEW.dragLabel(current.keys.length, current.point.title));
      } else {
        shiftGhost(event, null);
      }
      markDropTarget(findReleaseDest(event, at, current.keys));
    }

    function onPointerMove(event) {
      const at = canvasPoint(event);
      const current = drag.current;
      if (current && current.mode === "item") { handleEntryDragShift(event, at, current); return; }
      if (current) {
        const dx = at.x - current.x, dy = at.y - current.y;
        current.x = at.x; current.y = at.y;
        if (MAP_VIEW.isDragMove(current.startX, current.startY, at.x, at.y, MAP_TUNING.clickSlop)) current.moved = true;
        if (current.pan) {
          view.current.panX += dx; view.current.panY += dy;
        } else {
          const turn = multiply(rotationAxisAngle([0, 1, 0], dx * MAP_TUNING.dragRadians), rotationAxisAngle([1, 0, 0], dy * MAP_TUNING.dragRadians));
          view.current.rotation = multiply(turn, view.current.rotation);
        }
        scheduleRepaint();
        return;
      }
      const hit = pickPointAt(at.x, at.y);
      const before = hover.current;
      const same = (!hit && !before) || (hit && before && hit.type === before.type && (hit.type === "point" ? hit.index === before.index : hit.folder.id === before.folder.id));
      if (!same) { hover.current = hit; canvasRef.current.style.cursor = hit ? "pointer" : "grab"; scheduleRepaint(); }
    }

    function endEntryDrag() {
      if (ghostRef.current) ghostRef.current.hidden = true;
      if (canvasRef.current) canvasRef.current.style.cursor = "grab";
      const target = dropRef.current;
      dropRef.current = null;
      setDropTarget(null);
      setDragging(false);
      scheduleRepaint();
      return target;
    }

    function binName(target) {
      if (target === UNFILED) return t("common.unfiled");
      const folder = sidebarFolders.filter(function (item) { return String(item.id) === target; })[0];
      return folder ? folder.name : t("folder.fallback", { id: target });
    }

    // Shared by drag-and-drop and the panel's Move buttons. target: UNFILED or a folder id string.
    function shiftEntries(keys, target) {
      const payload = { folder_id: MAP_VIEW.dropFolderId(target), items: keys };
      return SDK.fetchJSON(API + "/folders/move", jsonInit("POST", payload)).then(function (result) {
        setMessage({ ok: true, text: MAP_VIEW.moveMessage(result.moved, binName(target)) });
        setSelection([]);
        return load();
      }).catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }

    function reopened(current) { return current ? Object.assign({}, current, { seq: Date.now() }) : current; }

    // Dragged memories join the topic by hand (kept by cron); notes are skipped by the server.
    function addToTheme(keys, topicId) {
      const payload = { items: keys, to_topic: topicId };
      return SDK.fetchJSON(API + "/topics/members/move", jsonInit("POST", payload)).then(function (result) {
        setMessage({ ok: true, text: themeShiftNotice(result, "topic.added") });
        setSelection([]);
        setPanel(reopened);  // the open panel may be that topic: its Connected list changed
        return load();
      }).catch(function (e) { setMessage({ ok: false, text: apiFault(e).text }); });
    }

    function onPointerUp(event) {
      const current = drag.current;
      drag.current = null;
      if (!current) return;
      if (current.mode === "item") {
        if (current.moved) {
          const target = endEntryDrag();
          if (MAP_VIEW.topicIdOf(target) !== null) addToTheme(current.keys, MAP_VIEW.topicIdOf(target));
          else if (target !== null) shiftEntries(current.keys, target);
        } else if (current.additive && !MAP_VIEW.isTopic(current.point)) {
          setSelection(keysAfterToggle(selection, current.point.key));
        } else {
          showNode(current.point);
        }
        return;
      }
      if (current.moved) return;
      const at = canvasPoint(event);
      const hit = folderAtPoint(at.x, at.y);
      if (hit) pickBin(hit.folder);
      else if (!current.additive && selection.length) setSelection([]);
    }

    function onPointerCancel() {
      const current = drag.current;
      drag.current = null;
      if (current && current.mode === "item" && current.moved) endEntryDrag();
    }

    function onPointerLeave() { if (!drag.current && hover.current) { hover.current = null; scheduleRepaint(); } }

    function onKeyDown(event) {
      if (event.key === "+" || event.key === "=") { event.preventDefault(); applyZoomLevel(view.current.zoom * MAP_TUNING.zoomStep); }
      else if (event.key === "-" || event.key === "_") { event.preventDefault(); applyZoomLevel(view.current.zoom / MAP_TUNING.zoomStep); }
    }

    function spinToward(target) {
      cancelSpin();
      const start = view.current.rotation;
      const turn = spinMatrixToFront(apply3(start, target));
      const began = window.performance.now();
      function step(now) {
        const progress = Math.min(1, (now - began) / MAP_TUNING.rotateMs);
        const eased = progress * (2 - progress);
        view.current.rotation = multiply(rotationAxisAngle(turn.axis, turn.angle * eased), start);
        scheduleRepaint();
        animation.current = progress < 1 ? window.requestAnimationFrame(step) : null;
      }
      animation.current = window.requestAnimationFrame(step);
    }

    // Clicking the selected folder again clears it. Selecting never closes an opened item.
    function pickBin(folder) {
      const id = folder === null ? UNFILED : folder.id;
      if (focus === id) { setFocus(null); return; }
      setFocus(id);
      if (folder !== null && folder.x !== null && folder.x !== undefined) spinToward([folder.x, folder.y, folder.z]);
    }

    function showNode(point) { setPanel({ type: point.lane, id: point.id, seq: Date.now() }); }

    // The sidebar and the points drop the folder now; the reload BinSidebar started confirms it.
    function handleBinDeleted(folderId) {
      setData(function (current) { return MAP_VIEW.withoutFolder(current, folderId); });
      setFolderList(function (list) { return MAP_VIEW.foldersWithout(list, folderId); });
      if (focus === folderId) setFocus(null);
      if (MAP_VIEW.isExpanded(expanded, folderId)) setExpanded(MAP_VIEW.toggledExpansion(expanded, folderId));
    }

    // A PartRoll move clears the map selection like every other move path.
    function postPartShift() { setSelection([]); return load(); }

    // A deleted memory or note leaves the list, the sphere, the selection and the panel at once.
    function onItemChanged(removed) {
      if (removed && panel) {
        const key = rowKey(panel.type, panel.id);
        setData(function (current) { return MAP_VIEW.withoutPoint(current, key); });
        setSelection(function (keys) { return keys.filter(function (other) { return other !== key; }); });
        setPanel(null);
      }
      load();
    }

    function toggleKindFilter(kind) { setKinds(Object.assign({}, kinds, { [kind]: !kinds[kind] })); }

    function closePanel() { setPanel(null); setFocus(null); setSelection([]); }

    function selectedGroup() {
      if (!selection.length) return null;
      return h("div", { className: "neospike-stack" },
        h("div", { className: "neospike-actions" },
          h("strong", null, t("map.selected", { n: selection.length })),
          h(Button, { type: "button", size: "xs", outlined: true, onClick: function () { setSelection([]); } }, t("common.clear"))),
        h(ViewShiftBar, { keys: selection, folders: sidebarFolders, onMove: shiftEntries }));
    }

    function folderGroup() {
      if (focus === null) return null;
      const folder = focus === UNFILED ? null : sidebarFolders.filter(function (item) { return item.id === focus; })[0];
      if (focus !== UNFILED && !folder) return h("p", { className: "neospike-muted" }, t("map.folderGone"));
      const members = data.points.filter(function (point) { return focus === UNFILED ? point.folder_id === null && !MAP_VIEW.isTopic(point) : point.folder_id === focus; })
        .slice().sort(function (a, b) { return (b.updated_at || 0) - (a.updated_at || 0); });
      return h(PartRoll, { title: folder ? folder.name : t("common.unfiled"), members: members, folders: sidebarFolders,
        expanded: folder ? MAP_VIEW.isExpanded(expanded, folder.id) : false,
        onToggleExpand: folder ? function () { setExpanded(MAP_VIEW.toggledExpansion(expanded, folder.id)); } : null,
        onOpen: showNode, onChanged: postPartShift, onMessage: setMessage });
    }

    // Body, Save/Delete with the folder Move bar beside it, the read-only connected text, then Connected.
    function memoryGroup() {
      if (!panel) return null;
      return h(ViewEntry, { panel: panel, version: data, folders: sidebarFolders, onMove: shiftEntries, onNoteChanged: onItemChanged,
        onItemChanged: onItemChanged, onMap: isPlotted, onOpen: showLinked, onMoved: postPartShift, onMessage: setMessage,
        dropTarget: dropTarget });
    }

    function isPlotted(key) { return byKey[key] !== undefined; }

    // A Connected row opens that item in the panel and turns the sphere to its point.
    function showLinked(item) {
      setPanel({ type: item.lane, id: item.id, seq: Date.now() });
      const index = byKey[item.key];
      if (index !== undefined) { const point = data.points[index]; spinToward([point.x, point.y, point.z]); }
    }

    const sideOpen = panel !== null || focus !== null || selection.length > 0;
    const unfiledCount = data.points.filter(function (point) { return point.folder_id === null && !MAP_VIEW.isTopic(point); }).length;
    return h("div", { className: "neospike-stack" },
      h("div", { className: "neospike-actions" },
        h(Input, { value: query, placeholder: t("map.filter"), "aria-label": t("map.filterLabel"), onChange: function (e) { setQuery(e.target.value); } }),
        MAP_KINDS.map(function (kind) {
          return h("label", { key: kind.value, className: "neospike-toggle" },
            h("input", { type: "checkbox", checked: kinds[kind.value], onChange: function () { toggleKindFilter(kind.value); } }), t(kind.label));
        }),
        h("label", { className: "neospike-toggle" },
          h("input", { type: "checkbox", checked: recentOnly, onChange: function () { setRecentOnly(!recentOnly); } }), t("map.recentOnly")),
        h(Button, { type: "button", size: "sm", outlined: true, onClick: function () { applyZoomLevel(view.current.zoom / MAP_TUNING.zoomStep); } }, "−"),
        h(Button, { type: "button", size: "sm", outlined: true, onClick: function () { applyZoomLevel(view.current.zoom * MAP_TUNING.zoomStep); } }, "+"),
        h(Button, { type: "button", size: "sm", outlined: true, onClick: load }, t("common.refresh")),
        h("span", { className: "neospike-muted" }, t("map.help", { n: data.points.length, zoom: zoomLevel.toFixed(1) }))
      ),
      h(Notice, { message: message }),
      h("div", { className: sideOpen ? "neospike-map has-panel" : "neospike-map" },
        h(BinSidebar, { folders: sidebarFolders, error: folderError, focus: focus, unfiledCount: unfiledCount, onSelect: pickBin,
          dragging: dragging, dropTarget: dropTarget, onChanged: load, onDeleted: handleBinDeleted, onMessage: setMessage }),
        h("div", { ref: wrapRef, className: "neospike-map-canvas" },
          h("canvas", { ref: canvasRef, tabIndex: 0, "aria-label": t("map.canvasLabel"),
            onPointerDown: onPointerDown, onPointerMove: onPointerMove, onPointerUp: onPointerUp, onPointerCancel: onPointerCancel,
            onPointerLeave: onPointerLeave, onKeyDown: onKeyDown }),
          MAP_VIEW.mapStatusText(loaded, loadError, data.points.length) !== null
            ? h("p", { className: loadError ? "neospike-message is-error" : "neospike-muted", role: loadError ? "alert" : null },
              MAP_VIEW.mapStatusText(loaded, loadError, data.points.length)) : null,
          h(MapColorKey, { open: legendOpen, onToggle: function () { setLegendOpen(!legendOpen); } })
        ),
        sideOpen ? h("div", { className: "neospike-map-panel neospike-stack" },
          h("div", { className: "neospike-map-panel-head" },
            h(Button, { type: "button", size: "sm", outlined: true, "aria-label": t("map.closePanel"), onClick: closePanel }, t("common.close"))),
          selectedGroup(), folderGroup(), memoryGroup()) : null
      ),
      h("div", { ref: ghostRef, className: "neospike-drag-ghost", hidden: true, "aria-hidden": true })
    );
  }

  // Keyboard/mouse alternative to drag-and-drop. onMove(keys, target) with target UNFILED or a folder id string.
  function ViewShiftBar(props) {
    const [target, setTarget] = useState("");
    const [busy, setBusy] = useState(false);
    function handleDestChange(event) { setTarget(event.target.value); }
    function done() { setBusy(false); }
    function move() {
      setBusy(true);
      props.onMove(props.keys, target).then(done, done);
    }
    return h("div", { className: "neospike-actions neospike-move-bar" },
      h("select", { className: "neospike-select", value: target, onChange: handleDestChange, "aria-label": t("common.moveToFolder") },
        h("option", { value: "" }, t("common.folderPick")),
        h("option", { value: UNFILED }, t("common.unfiled")),
        (props.folders || []).map(function (folder) { return h("option", { key: folder.id, value: String(folder.id) }, folder.name); })
      ),
      h(Button, { type: "button", size: "sm", outlined: true, disabled: busy || !props.keys.length || !target, onClick: move }, t("common.move")));
  }

  function paintTabs(active, setActive) {
    return h("div", { className: "neospike-shell" },
      h(TabsList, { className: "neospike-tabs" }, TAB_ITEMS.map(function (tab) {
        return h(TabsTrigger, { key: tab.value, value: tab.value, active: active === tab.value, onClick: function () { setActive(tab.value); } },
          t(tab.label));
      })),
      // Panels stay mounted so an unsaved note survives a tab switch.
      // Each panel has its own error boundary, so one failing tab leaves the others usable.
      h("div", { hidden: active !== "setup" }, h(PanelBoundary, null, h(InstallView))),
      h("div", { hidden: active !== "notes" }, h(PanelBoundary, null, h(MemosView, { active: active === "notes" }))),
      h("div", { hidden: active !== "docs" }, h(PanelBoundary, null, h(MemosView, { active: active === "docs", docs: true }))),
      h("div", { hidden: active !== "memory" }, h(PanelBoundary, null, h(MemoryView, { active: active === "memory" }))),
      h("div", { hidden: active !== "preferences" },
        h(PanelBoundary, null, h(PreferencesView, { active: active === "preferences" }))),
      h("div", { hidden: active !== "conversations" },
        h(PanelBoundary, null, h(DialogsView, { active: active === "conversations" }))),
      h("div", { hidden: active !== "config" }, h(PanelBoundary, null, h(SettingsView, { active: active === "config" }))),
      h("div", { hidden: active !== "map" }, h(PanelBoundary, null, h(MapView, { active: active === "map" })))
    );
  }

  // SDK.useI18n() reads the host I18nContext, so a language switch re-renders Page and every panel.
  // An SDK without useI18n (older host) stays English.
  const useHostI18n = typeof SDK.useI18n === "function" ? SDK.useI18n : null;

  function dashboardLocale() {
    const value = useHostI18n ? useHostI18n() : null;
    return value && typeof value === "object" ? value.locale : null;
  }

  function Page() {
    MAP_VIEW.setLanguage(MAP_VIEW.languageOf(dashboardLocale()));
    return h(Tabs, { defaultValue: "map" }, paintTabs);
  }

  window.__HERMES_PLUGINS__.register("neospike", Page);
})();
