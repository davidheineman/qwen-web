import { Tokenizer } from "@huggingface/tokenizers";

export class Tok {
  private constructor(private tk: Tokenizer) {}

  static async load(file: (f: string) => string): Promise<Tok> {
    const [json, config] = await Promise.all([
      fetch(file("tokenizer.json")).then((r) => r.json()),
      fetch(file("tokenizer_config.json")).then((r) => r.json()),
    ]);
    return new Tok(new Tokenizer(json, config));
  }

  encode(text: string): number[] {
    return this.tk.encode(text, { add_special_tokens: false }).ids;
  }

  decode(ids: number[]): string {
    return this.tk.decode(ids, { skip_special_tokens: false });
  }

  /** Human-readable form of a single token (decoded bytes; whitespace made visible) */
  piece(id: number): string {
    return this.tk.decode([id], { skip_special_tokens: false });
  }

  static readonly CHAT_PREFIX = "<|im_start|>user\n";
  /** Qwen3 chat template (non-thinking mode) after the user message, with an empty assistant turn. */
  static readonly CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

  /** Split a single-turn chat prompt into the user message and everything after it (incl. any assistant prefix). */
  static splitChat(text: string): { user: string; suffix: string } | null {
    const m = /^<\|im_start\|>user\n([\s\S]*?)(<\|im_end\|>\n<\|im_start\|>assistant\n[\s\S]*)$/.exec(text);
    return m ? { user: m[1], suffix: m[2] } : null;
  }
}

export function visible(s: string): string {
  return (
    s
      .replace(/\n/g, "↵")
      .replace(/\t/g, "⇥")
      .replace(/\r/g, "␍")
      .replace(/ /g, "·")
      // any other whitespace / invisible / replacement chars (e.g. partial UTF-8 bytes): show the code point
      .replace(/[\s\p{Cc}\p{Cf}\uFFFD]/gu, (c) => `\\u${c.codePointAt(0)!.toString(16).padStart(4, "0")}`) || "∅"
  );
}
