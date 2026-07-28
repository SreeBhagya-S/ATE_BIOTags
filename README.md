# ATE_BIOTags

# Annotation Guideline for Category-Specific BIO Aspect Tagging
---

## 1. Purpose

This guideline defines how annotators assign **BIO (Begin–Inside–Outside) aspect tags** to tokens in Malayalam–English code-mixed product reviews. Unlike a generic scheme that uses a single `B-ASP` / `I-ASP` pair, this scheme uses **category-specific tags** (e.g., `B-battery`, `I-camera`) so that the resulting gold data can be directly reused for downstream tasks such as aspect-category classification and aspect-based sentiment analysis, without a separate categorization step.

Tags are assigned strictly at the **token level**, over the pre-tokenized `tokens` list, and must align 1:1 in length and order with `tokens`.

---

## 2. Tagging Scheme

| Tag pattern | Meaning |
|---|---|
| `B-<category>` | First token of an aspect span belonging to `<category>` |
| `I-<category>` | Continuation token of the same aspect span (same category as the preceding `B-`/`I-` token) |
| `O` | Token is **not** part of any aspect span (includes opinion words, connectives, brand-only mentions not tied to a specific attribute, punctuation, etc.) |

**Rule:** An `I-<category>` tag can only follow a `B-<category>` or `I-<category>` of the *same* category. A category change always starts a new `B-<category>`, never an `I-` of the new category.

---

## 3. Aspect Category Taxonomy

Categories are derived from the 650-term bilingual Malayalam–English aspect lexicon built for the digital-gadget review domain. Each category corresponds to one BIO tag prefix.

| Category tag | Description | Example lexicon terms (EN) | Example lexicon terms (ML / code-mixed) |
|---|---|---|---|
| `battery` | Battery life, backup, charging | battery, battery backup, charging speed | ചാർജിംഗ്, battery backup |
| `camera` | Camera hardware and photo/video quality | camera, camera quality, zoom, selfie camera | ക്യാമറ, ക്യാമറ ക്വാളിറ്റി |
| `display` | Screen, resolution, brightness | display, screen, resolution, brightness | സ്ക്രീൻ, ഡിസ്‌പ്ലേ |
| `processor` | Chipset, speed, performance, RAM | processor, chipset, speed, RAM, performance | പ്രോസസർ, സ്പീഡ് |
| `software` | OS, UI, updates, apps, bugs | software, UI, update, apps, bugs | സോഫ്റ്റ്‌വെയർ, അപ്ഡേറ്റ് |
| `price` | Cost, value for money | price, cost, value for money, expensive | വില, പണം |
| `network` | Connectivity, signal, SIM, Wi-Fi | network, signal, SIM, Wi-Fi, 5G | നെറ്റ്‌വർക്ക്, സിഗ്നൽ |
| `phone` | Whole-device / general product mentions (brand or model name used as the aspect target itself, not a specific sub-component) | phone, mobile, iPhone, device | ഫോൺ, മൊബൈൽ |

> **Note:** This table lists the primary categories named in the lexicon description. Annotators must consult the full 650-term lexicon file for the authoritative category assignment of any specific term, and must not introduce new category tags without consulting the lexicon maintainers — this keeps the tag inventory closed and consistent with the downstream sentiment-analysis category set.

---

## 4. Tagging Rules

### 4.1 Single-word aspects
A single token naming a product attribute is tagged `B-<category>` on its own.

```
camera -> B-camera
battery -> B-battery
```

### 4.2 Multi-word aspects
Consecutive tokens that together name **one** product attribute are tagged as a single span: first token `B-<category>`, remaining tokens `I-<category>`.

```
battery backup -> [('battery','B-battery'), ('backup','I-battery')]
camera quality -> [('camera','B-camera'), ('quality','I-camera')]
```

Do **not** split a multi-word aspect into two separate single-token spans, and do not tag the second word with a different category unless it genuinely names a different attribute.

### 4.3 Code-mixed aspect expressions
Malayalam–English mixed expressions are annotated according to their **semantic aspect boundary**, not by language. If a Malayalam grammatical particle (postposition/case marker) is fused onto an English aspect word, or a Malayalam and English word jointly form one aspect expression, tag them as one span in the appropriate category.

```
batteryയുടെ backup -> [('batteryയുടെ','B-battery'), ('backup','I-battery')]
```

### 4.4 Morphologically modified Malayalam aspects
Malayalam inflections, possessive markers, and case suffixes (e.g., `-ന്റെ`, `-യുടെ`, `-ൽ`) attached to an aspect-bearing token do **not** change the category assignment or block the token from being tagged. Boundary decisions are based on the semantic root aspect, using the normalized form from the lexicon's suffix-aware morphological module as the reference for category lookup — but the **surface token** (with its suffix intact) is what receives the tag, since tagging operates on the original token list, not the normalized/lemmatized form.

```
screenന്റെ -> B-display   (root: screen -> category: display)
```

### 4.5 Nested / compound noun handling
When a compound noun contains a more specific aspect nested inside a broader one (e.g., "camera lens quality"), tag the **full compound as a single span** under the category of the most specific attribute being evaluated, rather than creating overlapping or nested spans (BIO is a flat, non-nested scheme).

```
camera lens quality -> [('camera','B-camera'), ('lens','I-camera'), ('quality','I-camera')]
```

### 4.6 Brand / general device mentions
A bare brand or model name (e.g., "iPhone", "phone", "mobile") used to refer to the product **as a whole**, without singling out a specific attribute, is tagged `B-phone` (single token) rather than `O`, so that general-product sentiment can still be captured downstream. If the same token is immediately followed by a specific attribute word referring to that device (e.g., "iPhone camera"), the brand token and attribute token are tagged as **separate spans** (brand as `B-phone`, attribute as its own `B-<category>` span), since they refer to two distinct aspect targets, not one compound aspect.

```
iPhone camera -> [('iPhone','B-phone'), ('camera','B-camera')]
```

### 4.7 Opinion / sentiment words
Purely evaluative or opinion-bearing words (e.g., "പുലിയാ" = "great/beast", "നല്ലതാ" = "good", "മോശം" = "bad") are **never** tagged as aspects, even if they appear adjacent to an aspect term. They are tagged `O`. Sentiment is captured in a separate downstream task, not in the aspect-BIO layer.

### 4.8 Function words / connectives
Discourse connectives, conjunctions, pronouns, and punctuation (e.g., "എന്നും", "പക്ഷെ", "but", "and", "?") are tagged `O`.

### 4.9 Ambiguous cases
If it is unclear (a) which category a term belongs to, or (b) whether a term is an aspect at all, the token is flagged and resolved through:
1. Reference to the annotation guideline rules above, then
2. Independent judgment by all three annotators, then
3. Majority vote; if no majority, joint discussion until consensus.

Ambiguous cases and their resolutions should be logged so the lexicon/category table can be extended in future revisions.

---

## 5. Annotation Workflow

- Each review is annotated **independently** by three bilingual annotators, without visibility into the other annotators' labels.
- Reviews are randomly distributed across annotators, while ensuring **every review receives three independent annotations**.
- Final gold labels are determined by **majority voting** across the three annotations at the token level.
- Disagreements that survive majority voting are resolved through **joint discussion** referencing this guideline.
- Inter-annotator agreement is measured using **Cohen's κ**; the current corpus achieves κ = 0.81 (substantial agreement).
- These gold annotations are used **only for evaluation** of aspect extraction performance — they are not consumed by the proposed extraction pipeline itself.

---

## 6. Data Format

Each annotated review is stored as one record with the following schema:

```json
{
  "review_id": "<string, unique ID>",
  "text": "<original raw review text>",
  "tokens": ["<token_1>", "<token_2>", "..."],
  "BIOTags": [
    [
      ["<token_1>", "<tag_1>"],
      ["<token_2>", "<tag_2>"],
      "..."
    ]
  ]
}
```

- `tokens` and the token list inside `BIOTags` must be **identical in content and order**.
- `BIOTags` is a list containing one list of `(token, tag)` pairs per review (kept as a nested list to stay consistent with the existing `Word_Tokens` / `BIO_Tags` pipeline format used in preprocessing).
- Every tag must be one of: `O`, or `B-<category>` / `I-<category>` for a category in the closed lexicon-derived taxonomy (Section 3).

---

## 7. Worked Examples

**Example 1 — brand + opinion + price (from prompt)**
```json
{
  "review_id": "1",
  "text": "iPhone എന്നും പുലിയാ പക്ഷെ പണം?",
  "tokens": ["iphone", "എന്നും", "പുലിയാ", "പക്ഷെ", "പണം"],
  "BIOTags": [[
    ["iphone", "B-phone"],
    ["എന്നും", "O"],
    ["പുലിയാ", "O"],
    ["പക്ഷെ", "O"],
    ["പണം", "B-price"]
  ]]
}
```
*Rationale:* `iphone` is a bare brand mention → `B-phone`. `എന്നും` (connective) and `പുലിയാ` (opinion word, "beast/great") → `O`. `പക്ഷെ` ("but") → `O`. `പണം` ("money/price") is a single-word aspect → `B-price`.

**Example 2 — multi-word aspect**
```json
{
  "review_id": "2",
  "text": "battery backup കൊള്ളാം",
  "tokens": ["battery", "backup", "കൊള്ളാം"],
  "BIOTags": [[
    ["battery", "B-battery"],
    ["backup", "I-battery"],
    ["കൊള്ളാം", "O"]
  ]]
}
```

**Example 3 — code-mixed suffixed aspect + separate attribute**
```json
{
  "review_id": "3",
  "text": "screenന്റെ quality kollam, camera vere level aanu",
  "tokens": ["screenന്റെ", "quality", "kollam", ",", "camera", "vere", "level", "aanu"],
  "BIOTags": [[
    ["screenന്റെ", "B-display"],
    ["quality", "I-display"],
    ["kollam", "O"],
    [",", "O"],
    ["camera", "B-camera"],
    ["vere", "O"],
    ["level", "O"],
    ["aanu", "O"]
  ]]
}
```

**Example 4 — brand followed by a distinct attribute (two separate spans)**
```json
{
  "review_id": "4",
  "text": "iPhone camera kidilan",
  "tokens": ["iPhone", "camera", "kidilan"],
  "BIOTags": [[
    ["iPhone", "B-phone"],
    ["camera", "B-camera"],
    ["kidilan", "O"]
  ]]
}
```

**Example 5 — no aspect present**
```json
{
  "review_id": "5",
  "text": "kollam nice",
  "tokens": ["kollam", "nice"],
  "BIOTags": [[
    ["kollam", "O"],
    ["nice", "O"]
  ]]
}
```

---

## 8. Quality Control Checklist (per review, per annotator)

- [ ] Every token in `tokens` has exactly one tag in the output.
- [ ] No `I-<category>` tag appears without an immediately preceding `B-<category>`/`I-<category>` of the *same* category.
- [ ] Every `B-`/`I-` category used exists in the closed taxonomy (Section 3) or lexicon.
- [ ] Multi-word aspects are kept as a single contiguous span, not split.
- [ ] Opinion/sentiment words are tagged `O`, never given an aspect category.
- [ ] Brand/general-device mentions follow the `B-phone` rule (Section 4.6).
- [ ] Ambiguous tokens are flagged and logged for majority-vote resolution.
