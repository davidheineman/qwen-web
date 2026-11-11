// Monaco editor for the reward function (Python highlighting only; no language server).

import * as monaco from "monaco-editor/editor/editor.api";
import "monaco-editor/languages/definitions/python/register";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

(self as unknown as { MonacoEnvironment: monaco.Environment }).MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

export interface CodeEditor {
  getValue(): string;
  /** replace the contents without firing onChange */
  setValue(v: string): void;
  /** mark lines with an error (1-based) or clear it */
  setError(line: number | null, message?: string): void;
}

export function createEditor(el: HTMLElement, onChange: (v: string) => void, onBlur: () => void): CodeEditor {
  const editor = monaco.editor.create(el, {
    value: "",
    language: "python",
    theme: "vs",
    automaticLayout: true,
    minimap: { enabled: false },
    fontSize: 12.5,
    lineHeight: 19,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    lineNumbers: "on",
    lineNumbersMinChars: 3,
    glyphMargin: false,
    folding: false,
    scrollBeyondLastLine: false,
    renderLineHighlight: "none",
    overviewRulerLanes: 0,
    hideCursorInOverviewRuler: true,
    scrollbar: { alwaysConsumeMouseWheel: false, verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
    padding: { top: 8, bottom: 8 },
    tabSize: 4,
    insertSpaces: true,
    wordWrap: "on",
  });
  let silent = false;
  editor.onDidChangeModelContent(() => {
    if (!silent) onChange(editor.getValue());
  });
  editor.onDidBlurEditorText(onBlur);
  // grow with the content (up to a limit) instead of scrolling a tiny box
  const fit = () => {
    const h = Math.min(420, Math.max(120, editor.getContentHeight()));
    el.style.height = `${h}px`;
  };
  editor.onDidContentSizeChange(fit);
  fit();
  return {
    getValue: () => editor.getValue(),
    setValue(v) {
      silent = true;
      editor.setValue(v);
      silent = false;
    },
    setError(line, message) {
      const model = editor.getModel()!;
      monaco.editor.setModelMarkers(
        model,
        "reward",
        line === null
          ? []
          : [{ startLineNumber: line, endLineNumber: line, startColumn: 1, endColumn: model.getLineMaxColumn(line), message: message ?? "", severity: monaco.MarkerSeverity.Error }],
      );
    },
  };
}
