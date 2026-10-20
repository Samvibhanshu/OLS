"""Text renderer for the synthetic stream: turns each record's latent state (topic, sentiment, lineage kind) into a post.

The RAG layer needs *text* (queries, retrieval, quotable evidence).  The simulator only produces latent embeddings, so
we render posts from the same latent state.  The renderer never changes the simulator's random stream: it uses its own
RNG, so every earlier experiment still reproduces.

How each kind of record is written (this is the simulator's design, not a finding):
  independent root     fresh combination of opener / aspect / sentiment clause / tail (some reuse a common talking point)
  campaign seed        persuasive post with topic keywords repeated (keyword stuffing) and hashtags
  repost               "RT @user: " + parent text, verbatim
  quote                parent text + short own comment
  paraphrase           parent with ~50% of content words swapped for synonyms and a sentence dropped
  AI derivative        formal rewrite of the parent (synonyms + boilerplate framing)
  coordinated          parent text plus the campaign hashtags
Paraphrases and AI rewrites are *not* near-duplicates at the text level (cosine ~0.4-0.7), which is exactly where
text-level de-duplication struggles; the lineage graph is built from the embedding channel as in the patent.
"""
from __future__ import annotations
import re
import numpy as np
from ..config import K_INDEPENDENT, K_REPOST, K_QUOTE, K_PARAPHRASE, K_AI, K_COORD

TOPICS = [
    ("the metro fare hike", ["fare", "metro", "ticket prices", "commuters", "ridership"]),
    ("the new data privacy law", ["privacy", "data", "consent rules", "regulators", "fines"]),
    ("the city e-scooter ban", ["scooters", "sidewalks", "ban", "riders", "safety"]),
    ("the university tuition increase", ["tuition", "students", "fees", "scholarships", "campus"]),
    ("the national cricket team selection", ["selection", "squad", "captain", "form", "selectors"]),
    ("the electric vehicle subsidy", ["subsidy", "EV buyers", "charging", "budget", "incentive"]),
    ("the remote work mandate", ["remote work", "office", "managers", "commute", "flexibility"]),
    ("the public library closures", ["libraries", "closures", "funding", "community", "books"]),
]
QUERY_TEMPLATES = ["How do people feel about {t}?", "What is the public opinion on {t}?", "What are people saying about {t}?"]

OPEN = ["Honestly,", "Just saw the news on", "Thread on", "My take on", "Everyone is talking about", "Thinking about", "Update:", "So,"]
CLAUSE = {  # sentiment class -> clauses (0 negative, 1 neutral, 2 positive)
    0: ["this is a terrible decision", "I am really disappointed", "it will hurt ordinary people", "a badly planned move",
        "this makes things worse for everyone", "completely unfair and rushed", "I strongly oppose it", "nothing about this works"],
    1: ["not sure what to make of it", "there are arguments on both sides", "we will have to wait and see",
        "details are still unclear", "it might help some and not others", "waiting for more information", "mixed feelings really", "too early to tell"],
    2: ["this is a great step forward", "I am genuinely pleased", "it will help ordinary people", "a well thought out move",
        "this makes things better for everyone", "long overdue and fair", "I fully support it", "this actually works"],
}
ASPECT = ["the effect on {a}", "what it means for {a}", "{a} will notice it first", "the debate over {a}", "{a} are at the centre of it"]
TAIL = ["Curious what others think.", "Will follow this closely.", "More soon.", "Reading the details tonight.", "Discussed it with friends.",
        "Saw it on the evening news.", "Local reaction is interesting.", "Posting this from my commute."]
TALKING_POINT = "Experts say the numbers simply do not add up"
COMMENT = {0: ["this.", "exactly, so bad", "can't disagree", "yep, awful"], 1: ["interesting", "hmm", "fair point maybe", "noted"],
           2: ["this!", "exactly, so good", "hard to disagree", "yes, finally"]}
SYN = {"terrible": "dreadful", "disappointed": "let down", "hurt": "harm", "ordinary": "everyday", "badly": "poorly", "worse": "less good",
       "completely": "entirely", "unfair": "unjust", "rushed": "hasty", "oppose": "reject", "great": "excellent", "pleased": "happy",
       "help": "benefit", "well": "carefully", "better": "improved", "overdue": "belated", "fair": "just", "support": "back",
       "people": "citizens", "decision": "choice", "move": "step", "everyone": "all", "works": "delivers", "unclear": "vague",
       "arguments": "points", "mixed": "divided", "early": "premature", "wait": "hold on", "information": "detail", "step": "move",
       "forward": "ahead", "think": "reckon", "news": "reports", "talking": "debating", "thinking": "reflecting", "thread": "post",
       "effect": "impact", "debate": "argument", "centre": "heart", "notice": "feel", "means": "implies", "plans": "proposals",
       "numbers": "figures", "experts": "specialists", "simply": "just", "close": "careful", "closely": "carefully", "sure": "certain"}
_WORD = re.compile(r"[A-Za-z']+")


def _syn(text: str, rng, p: float) -> str:
    def rep(m):
        w = m.group(0); lw = w.lower()
        return SYN[lw] if lw in SYN and rng.random() < p else w
    return _WORD.sub(rep, text)


def _base(text: str) -> str:                       # strip RT prefix / earlier quote comment so chains do not grow
    text = re.sub(r"^RT @\w+: ", "", text)
    text = re.sub(r"(\s#\w+)+\s*$", "", text.split(" || ")[0])      # drop trailing hashtags
    return re.sub(r"^Public discourse suggests that ", "", text.replace(" This view appears to be widely shared.", ""))


def topic_phrase(topic: int) -> str:
    return TOPICS[topic][0]


def make_query(topic: int, rng: np.random.Generator | None = None, i: int = 0) -> str:
    return QUERY_TEMPLATES[(i if rng is None else int(rng.integers(len(QUERY_TEMPLATES))))
                           % len(QUERY_TEMPLATES)].format(t=TOPICS[topic][0])


def render_texts(corpus, seed: int = 0, disguise: float = 0.0) -> list[str]:
    """One post per record (records are time ordered and parents precede children).
    disguise: probability that a repost / quote / coordinated copy is *rewritten* (heavy synonym swap, new opener and tail)
    instead of copied verbatim: a stress knob for text-level de-duplication.  0 = the default renderer."""
    rng = np.random.default_rng(31_000 + seed)
    n = corpus.n; texts: list[str] = [""] * n
    lab, topic, par, kind, camp = corpus.label, corpus.topic, corpus.true_parent, corpus.kind, corpus.campaign
    tags = {}                                                   # campaign id -> hashtags
    for i in range(n):
        tp = int(topic[i]); name, aspects = TOPICS[tp]
        p = par[i]
        if p < 0:
            c = int(lab[i])
            if camp[i] >= 0:                                    # coordinated-campaign seed (keyword stuffed)
                kw = [a.replace(" ", "") for a in aspects[:3]]
                tags[int(camp[i])] = " ".join("#" + k for k in kw)
                cl = rng.choice(CLAUSE[c], 2, replace=False)
                texts[i] = f"BREAKING: {name}! {cl[0]}. {cl[1]}. {name} {aspects[0]} {aspects[1]}. Share if you agree {tags[int(camp[i])]}"
            else:                                               # organic (or sleeper) root
                body = f"{rng.choice(OPEN)} {name}: {rng.choice(ASPECT).format(a=rng.choice(aspects))}. {rng.choice(CLAUSE[c]).capitalize()}."
                if rng.random() < 0.08:
                    body += f" {TALKING_POINT}."
                texts[i] = body + " " + str(rng.choice(TAIL))
            continue
        base = _base(texts[p]); k = int(kind[i])
        if disguise > 0 and k in (K_REPOST, K_QUOTE, K_COORD) and rng.random() < disguise:
            texts[i] = f"{rng.choice(OPEN)} " + _syn(base, rng, 0.95) + " " + str(rng.choice(TAIL))
            continue
        if k == K_REPOST:
            texts[i] = f"RT @user{int(rng.integers(900))}: {base}"
        elif k == K_QUOTE:
            texts[i] = f"{base} || {rng.choice(COMMENT[int(lab[p])])}"
        elif k == K_PARAPHRASE:
            sents = [s for s in re.split(r"(?<=[.!?])\s+", base) if s]
            if len(sents) > 2: sents.pop()                       # drop the closing sentence (keeps the sentiment clause)
            texts[i] = _syn(" ".join(sents), rng, 0.55)
        elif k == K_AI:
            core = _syn(base, rng, 0.7)
            core = core if core.split(" ")[0].isupper() or core.startswith("BREAKING") else core[:1].lower() + core[1:]
            texts[i] = "Public discourse suggests that " + core + " This view appears to be widely shared."
        elif k == K_COORD:
            texts[i] = base + " " + tags.get(int(camp[i]) if camp[i] >= 0 else -9, "#trending")
        else:
            texts[i] = base
    return texts
