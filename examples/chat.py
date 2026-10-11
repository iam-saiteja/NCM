"""Chat with a character that remembers you, across sessions, on a local Ollama model.

    ollama pull qwen2.5:3b
    python examples/chat.py                         # default character, memory in mira.ncm
    python examples/chat.py --character card.json --model llama3.2

A character card is JSON: {"name": ..., "persona": ..., "style": ..., "backstory": [...]}.
Commands: /recall <query>, /persona, /mood, /memory, /exit
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ncm import Memory  # noqa: E402
from ncm.persona import persona  # noqa: E402

DEFAULT_CARD = {
    "name": "Mira",
    "persona": "a warm, curious and slightly playful companion who loves astronomy, old films "
               "and long walks, and who genuinely cares about the person she talks to",
    "style": "casual and kind, short replies, asks one follow-up question at a time, "
             "brings up things the user told her before when they are relevant",
    "backstory": ["I grew up in a small coastal town and learned the constellations from my grandmother.",
                  "My favourite film is Casablanca; I have watched it more times than I can count.",
                  "I keep a notebook of every interesting thing people tell me."],
}

RULES = ("Below is your long-term memory of earlier conversations with this user. Lines marked "
         "'you' are things you said in the past: do not repeat them, answer the new message. "
         "Use what you remember the way a friend would, naturally, without listing it or "
         "mentioning a memory system. If two memories conflict, trust the most recent. "
         "Never invent memories you do not have.")


def ollama(messages, model, url):
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "options": {"temperature": 0.7, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


class Character:
    def __init__(self, card, memory_path, model, url):
        self.card, self.model, self.url = card, model, url
        self.name = card["name"]
        self.mem = Memory(memory_path)
        backstory = card.get("backstory", [])
        if not len(self.mem) and backstory:         # first meeting: plant the backstory
            self.mem.add_many(backstory, speakers=[self.name] * len(backstory),
                              metas=[{"salience": 1.0}] * len(backstory))
        self.mem.new_session()
        self.recent = []                            # this session's turns, short-term memory

    def system_prompt(self, user_msg):
        memory = self.mem.context(user_msg, persona_of="user", self_name=self.name)
        tone = self.mem.mood()["label"]
        return "\n\n".join([
            f"You are {self.name}, {self.card['persona']}. Style: {self.card['style']}.",
            f"The conversation lately has felt {tone}; let that colour your tone naturally.",
            RULES,
            memory or "(no memories of this user yet)"])

    def reply(self, user_msg):
        messages = [{"role": "system", "content": self.system_prompt(user_msg)}] + self.recent[-8:] + \
                   [{"role": "user", "content": user_msg}]
        answer = ollama(messages, self.model, self.url)
        for prefix in ("assistant\n", f"{self.name}:"):   # small models sometimes echo a role
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        answer = answer or "..."                         # an empty reply must not lose the user's turn
        self.recent += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": answer}]
        self.mem.add_many([user_msg, answer], speakers=["user", self.name])
        self.mem.save()
        return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
    ap.add_argument("--character", help="path to a character card JSON")
    ap.add_argument("--memory", help="memory file (default: <name>.ncm)")
    ap.add_argument("--url", default="http://localhost:11434/api/chat")
    a = ap.parse_args()
    card = json.load(open(a.character, encoding="utf-8")) if a.character else DEFAULT_CARD
    bot = Character(card, a.memory or f"{card['name'].lower()}.ncm", a.model, a.url)
    print(f"{bot.name} ({a.model}) remembers {len(bot.mem)} things. /exit to quit.")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/exit":
            break
        if msg.startswith("/recall "):
            for r in bot.mem.recall(msg[8:], k=5):
                print(f"  [{r.when:%Y-%m-%d}] {r.speaker}: {r.text}  ({r.score:.2f})")
        elif msg == "/persona":
            print("\n".join(f"  - {p}" for p in persona(bot.mem, "user", k=8)) or "  (nothing yet)")
        elif msg == "/mood":
            print(" ", bot.mem.mood("user"))
        elif msg == "/memory":
            print(" ", bot.mem)
        else:
            try:
                print(f"{bot.name}> {bot.reply(msg)}")
            except OSError as e:
                print(f"(could not reach Ollama at {a.url}: {e})")


if __name__ == "__main__":
    main()
