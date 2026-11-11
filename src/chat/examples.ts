// Starter prompts for the Ladder page (index.html). Picked by trying variants on general-large-4b: these are the ones that
// gave the most sensible continuations, but at 51M parameters the model mostly imitates the *format*
// (Python with docstrings, step-by-step math, line-broken verse) rather than solving the task.

export const EXAMPLES: [name: string, text: string][] = [
  ["math", "Problem: What is 5 + 3?\nSolution: 5 + 3 = 8.\n\nProblem: What is 6 + 1?\nSolution:"],
  [
    "code",
    'class Stack:\n    """A simple stack."""\n\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n',
  ],
  ["poetry", "The rain falls soft upon the hill,\nThe wind is calm, the night is still,\n"],
  ["instructions", "Write a short story about a dog.\n\nOnce upon a time, there was a dog named"],
  [
    "wordle",
    "Wordle. G = right letter, right spot. Y = right letter, wrong spot. X = not in the word.\n\n" +
      "Answer: CRANE\nGuess: TRAIN -> XGGYX\nGuess: CRANE -> GGGGG\n\nAnswer: APPLE\nGuess: PLATE ->",
  ],
  [
    "regex",
    'import re\n\n# Match one or more digits\ndigits = re.compile(r"\\d+")\n\n# Match an email address\nemail = re.compile(r"',
  ],
];
