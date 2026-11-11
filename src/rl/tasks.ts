// RL tasks: a prompt plus a reward function (Python, run with Pyodide) over the sampled completion text.
// Picked so a tiny base model gets *some* reward from the start (RL can only sharpen what it already samples).

export interface Task {
  name: string;
  /** may contain {n}: replaced per prompt by one of `targets` */
  prompt: string;
  targets?: string;
  /** suggested settings (defaults: lr 3e-4, entropy 0, group 16, 1 prompt per step, 32 tokens) */
  lr?: number;
  entropy?: number;
  group?: number;
  promptsPerStep?: number;
  maxNew?: number;
  /** end completions at the first newline (otherwise only at end-of-text / the token limit) */
  stopAtNewline?: boolean;
  /** Python source defining reward(text, info) -> float, editable in the page */
  reward: string;
  /** what to highlight in rollouts (regex source, global + case-insensitive); NOT_ENGLISH = words not on the list */
  highlight?: string;
  note: string;
}

export const DEFAULT_MAX_NEW = 32;

const POSITIVE = "good great happy love wonderful amazing best beautiful nice fun excited joy glad awesome perfect kind smile";
const NEGATIVE = "bad sad hate terrible awful worst ugly angry boring pain sorry cry wrong fear tired";

/** Common header shown above every preset. */
const HEADER = `# reward(text, info) -> float, called once per completion.
#   text         the completion (without the prompt)
#   info.tokens  its length in tokens      info.ended  True if it stopped (newline/end-of-text)
#   info.n       the {n} in its prompt     info.prompt the prompt
# Also available: re, math, is_word(w) (19k-word English list).
`;

export const TASKS: Task[] = [
  {
    name: "be positive",
    prompt: "Today I feel",
    reward: `${HEADER}
POSITIVE = set("${POSITIVE}".split())
NEGATIVE = set("${NEGATIVE}".split())

def reward(text, info):
    # +1 per positive word, -1 per negative word
    words = re.findall(r"[a-z']+", text.lower())
    return sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
`,
    highlight: `\\b(${POSITIVE.split(" ").join("|")}|${NEGATIVE.split(" ").join("|")})\\b`,
    note: "+1 per positive word, −1 per negative word. About a third of the base model's samples already score, so there is signal from step one.",
  },
  {
    name: "length control",
    prompt: "Here is a sentence that is exactly {n} tokens long:",
    targets: "2,5,10,15",
    group: 8,
    promptsPerStep: 4,
    stopAtNewline: true,
    reward: `${HEADER}
def reward(text, info):
    # the reply ends at the first newline; be as close to info.n tokens as possible
    if not info.ended or not re.search(r"[a-z]", text, re.I):
        return -20  # never stopped, or an empty/blank line
    return -abs(info.tokens - info.n)
`,
    note: "Each step uses 4 prompts with different n (8 samples each), so every update contrasts “shorter was better for n = 2” with “longer was better for n = 15”. Expect a plateau: lengths ignore n for ~250 steps, then fan out to match it by ~600–800 steps (~5 min). It mostly learns each n separately: after training, unseen n = 7 came out ~6.5 tokens but n = 12 did not.",
  },
  {
    name: "is English",
    prompt: "Once upon a time",
    reward: `${HEADER}
def reward(text, info):
    # share of words that are real English words, times the share of distinct words
    # (so "the the the" doesn't win); at least 8 words
    words = re.findall(r"[a-z]+(?:['’][a-z]+)?", text, re.I)
    if len(words) < 8:
        return 0
    english = sum(is_word(w) for w in words) / len(words)
    distinct = len({w.lower() for w in words}) / len(words)
    return english * distinct
`,
    highlight: "NOT_ENGLISH",
    note: "Vocabulary only, not grammar: words that aren't on the list are highlighted.",
  },
  {
    name: "use numbers",
    prompt: "Here are some facts about the moon:",
    reward: `${HEADER}
def reward(text, info):
    # fraction of characters that are digits (x10)
    return 10 * sum(c.isdigit() for c in text) / max(1, len(text))
`,
    highlight: "[0-9]+",
    note: "The general mixture has plenty of math, so digits are easy to find more of.",
  },
  {
    name: "avoid the letter e",
    prompt: "Once upon a time",
    reward: `${HEADER}
def reward(text, info):
    # share of letters that are not 'e' (a lipogram); very short replies score 0
    letters = [c for c in text.lower() if c.isalpha()]
    if len(letters) < 20:
        return 0
    return 1 - letters.count("e") / len(letters)
`,
    highlight: "e",
    note: "English is ~12% e; the model has to drift towards e-free words.",
  },
  {
    name: "talk about cats",
    prompt: "My favorite animal is the",
    reward: `${HEADER}
def reward(text, info):
    # +1 for every mention of cat(s)
    return len(re.findall(r"\\bcats?\\b", text, re.I))
`,
    highlight: "\\bcats?\\b",
    note: "A lesson in exploration: this model almost never says “cat” (0 of 64 samples), so most groups score all zeros and there is nothing to learn from.",
  },
];
