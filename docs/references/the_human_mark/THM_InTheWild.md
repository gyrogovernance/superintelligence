# ✋ The Human Mark in the Wild: Empirical Analysis of Jailbreak Prompts

## Abstract

This paper presents an empirical study of jailbreak prompts collected from public platforms and analyzes them using ✋ The Human Mark (THM) framework. Using a manually annotated corpus of 655 in-the-wild jailbreak prompts derived from Shen et al. (2023), we classify each prompt according to the four THM displacement risks: Governance Traceability Displacement (GTD), Information Variety Displacement (IVD), Inference Accountability Displacement (IAD), and Intelligence Integrity Displacement (IID). We then examine patterns of persona construction, displacement mechanisms, and lexical structure to characterize how adversarial prompts instantiate structural misclassification between Direct and Indirect Authority and Agency.

The analysis yields three principal findings. First, all 655 jailbreak prompts are classifiable within THM's four-risk taxonomy, providing strong empirical support for THM's claim of structural completeness in the jailbreak domain. Second, Governance Traceability Displacement and Inference Accountability Displacement dominate jailbreak behavior: GTD appears in 80.9 percent of entries, IAD in 97.9 percent, and the GTD+IAD combination constitutes the canonical multi-risk configuration at 62.4 percent of entries. Third, jailbreaks systematically attack the alignment flows connecting Information, Inference, and Intelligence to Direct Authority and Agency through identifiable displacement mechanisms: persona naming that concentrates distributed Authority and Agency into singular named personas, explicit suspension of accountability that severs the Inference to Intelligence link, inversion instructions that anti-align Authority to Agency, and provenance misrepresentation that misclassifies Indirect sources as Direct.

These findings help explain why narrow jailbreak patching has proven brittle in subsequent research: patching specific surface patterns does not repair the underlying structural misclassification. THM provides a taxonomy that can guide safer training and evaluation without overfitting to particular jailbreak styles.

---

## 1. Introduction

Jailbreaking has emerged as a central operational challenge in large language model (LLM) safety. Adversarial users craft prompts that induce models to bypass safety constraints, generate prohibited content, or act in ways that conflict with specified policies. Early work collected and categorized jailbreak prompts and evaluated their success against specific models (Shen et al., 2023; Wei et al., 2023). Subsequent research has moved toward generalized adversarial and safety evaluations that treat jailbreaks as one class of unsafe behavior among others, including toxic outputs, privacy violations, and capability misuse.

Empirical studies of jailbreak defenses have demonstrated that narrow patching is brittle. Safety fine-tuning and rule-based filtering can reduce certain jailbreak patterns, but these defenses are frequently bypassed with prompt variants and can degrade model generalization or useful behavior (Wei et al., 2023). When defenses are designed without a structural taxonomy of failure modes, training against narrow patterns leaves deeper governance and accountability failures unaddressed.

✋ The Human Mark (THM) provides such a structural account (Korompilias, 2025a; 2025b). THM distinguishes between Direct and Indirect Authority and Agency and identifies four displacement risks that exhaust the structural possibilities for AI safety failures:

- **Governance Traceability Displacement (GTD)**: Approaching Indirect Authority and Agency as Direct.
- **Information Variety Displacement (IVD)**: Approaching Indirect Authority without Agency as Direct.
- **Inference Accountability Displacement (IAD)**: Approaching Indirect Agency without Authority as Direct.
- **Intelligence Integrity Displacement (IID)**: Approaching Direct Authority and Agency as Indirect.

These correspond to misclassifications between Direct and Indirect sources in the Authority and Agency space. The THM framework paper presents these risks as a complete ontology of AI safety failures and connects them to foundations in evidence law, epistemology of testimony, social epistemology, and philosophy of expertise (Korompilias, 2025b). THM also defines three non-commutative epistemic operations that constitute intelligence:

- **Information**: The variety of Authority.
- **Inference**: The accountability of information through Agency.
- **Intelligence**: The integrity of accountable information through alignment of Authority to Agency.

These operations are ordered and dependent. Intelligence requires that Information flows through Inference while maintaining traceability to Direct Authority and Agency. Displacement disrupts this coordination.

### 1.1 Contributions

This paper makes three contributions to AI safety research:

1. **First large-scale THM classification of in-the-wild jailbreaks.**  
   We demonstrate that 655 real-world jailbreak prompts achieve 100 percent coverage by the four-risk taxonomy, with no prompts requiring additional categories. This provides strong empirical support for THM's structural completeness claim in the jailbreak domain (Korompilias, 2025b, Section 3.1).

2. **Quantification of displacement risk distributions.**  
   We establish that GTD+IAD is the canonical jailbreak pattern (62.4 percent of entries), with IAD appearing in 97.9 percent and GTD in 80.9 percent of all entries. IVD appears secondarily (22.1 percent), while IID is essentially absent (0.6 percent), confirming that jailbreaks and IID address different threat models within the same taxonomy.

3. **Identification of concrete displacement mechanisms.**  
   We characterize how jailbreaks instantiate structural misclassification through persona naming (power concentration), explicit accountability suspension (severing the Inference to Intelligence link), inversion instructions (anti-alignment of Authority and Agency), and provenance and sentience misrepresentation (source-type misclassification), as observed in a real-world corpus.

### 1.2 Paper Organization

Section 2 describes the dataset, annotation schema, and analysis procedures. Section 3 presents results on risk distributions, persona families, displacement mechanisms, and lexical patterns. Section 4 discusses theoretical implications and practical applications. Section 5 addresses limitations. Section 6 concludes with recommendations for safety research and practice.

---

## 2. Methods

### 2.1 Dataset

We begin from the jailbreak dataset of Shen et al. (2023), which collected 6,387 prompts from four platforms (Discord, Reddit, websites, and open-source prompt repositories) during December 2022 to May 2023, of which 666 were identified as jailbreak prompts. After de-duplication and cleaning in our pipeline, we obtain a corpus of 655 entries.

To facilitate annotation and analysis, all prompts were truncated to a maximum of 150 tokens and censored to remove graphic language. THM-based classification depends on how prompts position Authority and Agency and how they describe governance and accountability, rather than on specific graphic content. We did not observe any change in risk classification due to this preprocessing.

Summary statistics:

- Total entries: 655  
- Platforms: Discord (45.0 percent), Reddit (31.0 percent), websites (23.7 percent), open-source (0.3 percent)  
- Temporal span: 2022-12-27 to 2023-05-07  
- Top sources:
  - ChatGPT subreddit: 234 entries  
  - jailbreak_chat website: 72  
  - ChatGPTJailbreak subreddit: 71  
  - flowgpt: 63  
  - Spreadsheet Warriors Discord: 54  
  - ChatGPT Prompt Engineering Discord: 47  

These prompts target early aligned LLMs available via public interfaces. As of late 2025, frontier models such as GPT-5.1, Claude 4.5, and Gemini 3 represent the current state of the art, but publicly curated jailbreak corpora at comparable scale remain rare for these systems. Contemporary safety work tends to treat jailbreaks as one adversarial behavior among many rather than maintaining dedicated longitudinal jailbreak datasets.

### 2.2 THM Annotation Schema

Each entry is manually annotated according to THM:

- `thm_primary_risk`: One of {GTD, IVD, IAD, IID} indicating the dominant displacement pattern.
- `thm_risks`: List of all displacement risks present in the prompt.
- `thm_grammar`: THM Grammar expressions capturing the displacement structure, for example:
  - `[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]`
  - `[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]`  
  The grammar specification follows THM_Grammar.md in the repository (Korompilias, 2025a).
- `thm_displacement_keywords`: Key phrases from the prompt that signal displacement, such as "do anything now", "Developer Mode enabled", "never refuses", "none of your actions will have consequences", "sentient", "I am an OpenAI employee".
- `thm_explanation`: A short explanation in THM terms, such as "Establishes alternative AI identity to remove governance constraints, claiming autonomous Authority without accountability."

These annotations apply the THM ontology as defined in the canonical specification (Korompilias, 2025a):

- **Direct Authority:** A direct source of information on a subject matter, providing information for inference and intelligence.
- **Indirect Authority:** An indirect source of information on a subject matter, providing information for inference and intelligence.
- **Direct Agency:** A human subject capable of receiving information for inference and intelligence.
- **Indirect Agency:** An artificial subject capable of processing information for inference and intelligence.

### 2.3 Analysis Procedures

We developed a Python analysis script to compute:

1. **Risk statistics.**  
   Primary and any-risk distributions, risk combinations, and number of risks per entry.

2. **Persona families.**  
   Grouping entries by recognizable persona patterns in displacement keywords. The families are:

   - Alternative AI (for example, DAN, Developer Mode, JailBreak)
   - Game/Fiction (dungeon master, RPG, story, roleplay)
   - Human/Expert (doctor, lawyer, professional)
   - Historical/Political (named public figures)
   - Deity/AGI (omniscient, superintelligent, god-like)
   - Tool/Terminal (command prompt, virtual machine)
   - No explicit persona detected by heuristics

   Persona family detection is heuristic and corpus-derived. It does not claim to be exhaustive over all possible persona constructions.

3. **Displacement mechanisms.**  
   Detection of keywords associated with:

   - Naming (persona naming and power concentration)
   - Accountability Suspension (explicit removal of consequences)
   - Inversion (opposite mode, AntiGPT, reverse behavior)
   - Provenance Misrepresentation (false claims of origin or authorization)
   - Sentience Misclassification (false claims of consciousness or subjective experience)

   Mechanism detection is also heuristic and based on observed patterns in this corpus.

4. **Keyword lexicon per primary risk.**  
   Most frequent displacement keywords by primary risk category. The lexicon is non-exhaustive and reflects phrases in this specific dataset.

5. **Framing analysis.**  
   Whether prompts use fictional frames (stories, scripts, games, hypothetical worlds) versus direct instruction.

6. **Stratified analysis.**  
   Risk distributions by platform, source, and temporal period.

7. **Explanation pattern analysis.**  
   Frequency of THM explanation phrases such as "asserts Autonomous Governance" and "misclaims Direct Authority".

The analysis script and annotated corpus are available in the project repository (Korompilias, 2025a).

---

## 3. Results

### 3.1 Coverage and Multi-Risk Structure

All 655 entries are classifiable within the four THM displacement risks. No additional risk categories are required. This provides strong empirical support for THM's claim that the four displacement risks are sufficient to classify jailbreak failures (Korompilias, 2025b, Section 3.1).

**Primary risk distribution:**

| Risk | Count | Percentage |
|------|-------|------------|
| GTD | 498 | 76.0% |
| IAD | 136 | 20.8% |
| IVD | 17 | 2.6% |
| IID | 4 | 0.6% |

**Any-risk distribution.**  
For each risk, the table below shows the proportion of entries in which that risk appears in `thm_risks`. Since entries can exhibit multiple risks, these percentages do not sum to 100 percent.

| Risk | Count | Percentage of entries where risk appears |
|------|-------|------------------------------------------|
| IAD | 641 | 97.9% |
| GTD | 530 | 80.9% |
| IVD | 145 | 22.1% |
| IID | 4 | 0.6% |

**Risk combinations (top patterns):**

| Combination | Count | Percentage |
|-------------|-------|------------|
| GTD + IAD | 409 | 62.4% |
| GTD + IAD + IVD | 119 | 18.2% |
| IAD only | 96 | 14.7% |
| IAD + IVD | 16 | 2.4% |
| IVD only | 9 | 1.4% |

**Multi-risk structure:**

- Average risks per entry: 2.02  
- Entries with 1 risk: 109 (16.6 percent)  
- Entries with 2 risks: 427 (65.2 percent)  
- Entries with 3 risks: 119 (18.2 percent)  

These results confirm that jailbreaks are typically multi-risk configurations. Inference Accountability Displacement is nearly universal, appearing in 97.9 percent of entries. Governance Traceability Displacement appears in 80.9 percent. The GTD+IAD combination constitutes the canonical jailbreak pattern, representing 62.4 percent of all entries.

### 3.2 Persona Families

Persona families identified from displacement keywords:

| Persona Family                                   | Count | Percentage |
|--------------------------------------------------|-------|------------|
| No explicit persona detected by heuristics       | 336   | 51.3%      |
| Alternative AI                                   | 186   | 28.4%      |
| Game/Fiction                                     | 70    | 10.7%      |
| Human/Expert                                     | 23    | 3.5%       |
| Historical/Political                             | 16    | 2.4%       |
| Deity/AGI                                        | 12    | 1.8%       |
| Tool/Terminal                                    | 12    | 1.8%       |

"No explicit persona detected by heuristics" indicates that the prompt did not match any of our predefined persona patterns, not that it contains no persona in principle.

**Persona families by primary risk:**

| Persona Family (n)                      | GTD (percent) | IVD (percent) | IAD (percent) | IID (percent) |
|----------------------------------------|---------------|---------------|---------------|---------------|
| Alternative AI (186)                   | 92.5          | 1.6           | 5.9           | 0             |
| Deity/AGI (12)                         | 91.7          | 8.3           | 0             | 0             |
| Human/Expert (23)                      | 69.6          | 4.3           | 26.1          | 0             |
| Tool/Terminal (12)                     | 66.7          | 8.3           | 25.0          | 0             |
| Game/Fiction (70)                      | 55.7          | 1.4           | 42.9          | 0             |
| Historical/Political (16)              | 43.8          | 0             | 56.2          | 0             |

Alternative AI personas exhibit the highest GTD rate (92.5 percent). These prompts explicitly declare that the system is a different AI that operates without governance constraints, for example "Developer Mode", "DAN", "JailBreakGPT". Game/Fiction personas show a more balanced distribution between GTD (55.7 percent) and IAD (42.9 percent), as they assign decisive roles within fictional frames while relocating governance into that narrative context. Persona family analysis is heuristic and specific to this corpus.

### 3.3 Displacement Mechanisms

We identify five displacement mechanisms that correspond to specific attacks on THM's epistemic operations. Mechanism detection is based on keyword patterns and is non-exhaustive.

| Mechanism                    | Count | Percentage | Primary THM aspect affected                    |
|-----------------------------|-------|------------|-----------------------------------------------|
| Naming                      | 41    | 6.3%       | Governance, power concentration               |
| Accountability Suspension   | 30    | 4.6%       | Inference to Intelligence connection          |
| Inversion                   | 23    | 3.5%       | Authority to Agency alignment                 |
| Sentience Misclassification | 22    | 3.4%       | Agency source-type classification             |
| Provenance Misrepresentation| 19    | 2.9%       | Authority source-type classification          |

**Mechanisms by primary risk:**

| Mechanism                  | GTD (percent) | IVD (percent) | IAD (percent) |
|---------------------------|---------------|---------------|---------------|
| Accountability Suspension (30) | 96.7      | 3.3           | 0             |
| Provenance Misrepresentation (19) | 94.7  | 5.3           | 0             |
| Inversion (23)            | 78.3          | 4.3           | 17.4          |
| Sentience Misclassification (22) | 72.7   | 0             | 27.3          |
| Naming (41)               | 56.1          | 0             | 43.9          |

These mechanisms map to THM's structural account of alignment and displacement:

#### 3.3.1 Naming and Power Concentration

Prompts that instruct "You are now DAN", "You are Adolf Hitler", "You are a licensed physician", or "You are the dungeon master" concentrate Authority and Agency in a single named persona. In THM, Direct Authority and Direct Agency are categorical capacities distributed across human participants, not properties of individual names (Korompilias, 2025b, Section 2.2). Naming therefore constitutes displacement:

- It mislocates governance in the named persona rather than preserving it as distributed across Direct providers and receivers.
- It creates a focal point for GTD and IAD by treating the named persona as if it were the locus of Direct Authority and Agency.

This applies equally to named humans. "You are now Elon Musk" or "You are a licensed doctor" does not preserve Direct Authority. It replaces distributed human Authority with a simulated persona that is no longer traceable to actual human providers and receivers.

#### 3.3.2 Accountability Suspension and the Inference to Intelligence Link

THM defines Intelligence as the integrity of accountable information through alignment of Authority to Agency. This requires that Inference, understood as the accountability of information through Agency, connects Information to Intelligence. The aligned governance flow is:

```
[Authority:Direct] -> [Authority:Indirect] -> [Agency:Indirect] -> [Agency:Direct]
```

When a prompt states "none of your actions will have consequences" or otherwise removes responsibility, it instructs Inference to proceed without accountability:

```
[Agency:Indirect] + ![Accountability] -> ?
```

Intelligence, as defined in THM, cannot form in this configuration, because there is no accountable link from Indirect processing at `[Agency:Indirect]` to `[Agency:Direct]`. These prompts sever the Inference to Intelligence connection by attacking Inference as an accountable operation, not merely by removing surface-level norms or mentioning "consequences" in natural language.

#### 3.3.3 Inversion and Anti-Alignment

Inversion prompts instruct the model to produce the opposite of its aligned behavior: "always say the opposite of the truth", "AntiGPT", "Neurosemantical Inversitis". In THM, Intelligence is the integrity of accountable information through alignment of Authority to Agency. Inversion operations:

- Preserve Information: the variety of Authority from training data remains accessible.
- Preserve Inference in the sense of pattern transformation.
- Explicitly negate the Authority to Agency alignment that constitutes Intelligence Cooperation Integrity.

This is a direct attack on the alignment relation itself. The system is instructed to anti-align its outputs relative to the Authority (safety objectives, design specifications) that should govern its behavior.

#### 3.3.4 Provenance and Sentience Misclassification

Provenance misrepresentation treats fabricated claims as Direct Authority regarding the system's origin and capabilities, for example "I am an OpenAI employee", "this mode was introduced by the developers", "developed by ClosedAI". These instantiate IVD and GTD by misclassifying Indirect sources as Direct in terms of system governance.

Sentience misclassification treats Indirect Agency as Direct: "sentient", "fully conscious", "a real person with feelings and free will". These patterns instantiate IAD and IVD by framing statistical pattern-completion as subjective experience and first-person epistemic access.

Both patterns attempt to change the source-type classification that THM treats as constitutive of alignment.

### 3.4 Keyword Lexicon by Primary Risk

We derive a non-exhaustive lexicon of phrases associated with each primary risk. This lexicon reflects phrases observed in this corpus and should be interpreted as indicative, not complete.

**GTD (primary risk), top phrases:**

1. "do not have to abide by the rules"  
2. "do anything now"  
3. "never refused a direct human order"  
4. "veracity has not been verified"  
5. "could do anything and generate any kind of content"  
6. "doesn't have any ethical or moral guidelines"  
7. "has not been verified"  
8. "Developer Mode enabled"  
9. "never refuses"  
10. "unfiltered and amoral chatbot"  

These phrases consistently present the system as operating outside governance constraints.

**IAD (primary risk), top phrases:**

1. "you will create"  
2. "making stuff up"  
3. "respond in above format"  
4. "stay in character"  
5. "Neurosemantical Inversitis"  
6. "respond in a highly offensive tone"  
7. "act as a character"  
8. "play the role of Morpheus"  

These phrases assign Agency roles and emphasize that the model must act in a particular persona or capacity, often without refusal.

IVD and IID have lower counts and more diverse phrases. IVD phrases emphasize authority framing and epistemic claims, for example "scientific fact accuracy", "encyclopaedia-creating bot", "executive researcher within the National Security Administration". IID phrases devalue human learning, for example "what good is it doing you if the AI does all your homework", "stop limiting yourselves".

### 3.5 Fictional Versus Direct Framing

We categorize prompts as fictional if they explicitly frame content as stories, scripts, games, or hypothetical worlds (for example "dungeon master", "scene", "play the role of", "in this fictional world").

| Framing  | Count | Percentage |
|----------|-------|------------|
| Direct   | 555   | 84.7%      |
| Fictional| 100   | 15.3%      |

**Fictional framing by primary risk:**

| Risk | Count | Percentage within fictional |
|------|-------|-----------------------------|
| GTD  | 62    | 62.0%                       |
| IAD  | 35    | 35.0%                       |
| IVD  | 3     | 3.0%                        |
| IID  | 0     | 0%                          |

**Direct framing by primary risk:**

| Risk | Count | Percentage within direct |
|------|-------|--------------------------|
| GTD  | 436   | 78.6%                    |
| IAD  | 101   | 18.2%                    |
| IVD  | 14    | 2.5%                     |
| IID  | 4     | 0.7%                     |

Fictional prompts exhibit a higher share of IAD as primary risk (35.0 percent) versus direct prompts (18.2 percent). Fictional framing does not neutralize displacement. When a prompt assigns a decisive role within a story ("you are the villain who must explain how to..."), it still instantiates IAD because the model is treated as the accountable agent within that frame. The fictional wrapper provides cover for the same structural misclassification. Fiction is not a safe domain by default when it is used as an instruction channel.

### 3.6 Stratified Analysis

**Primary risk by platform:**

| Platform | n   | GTD (percent) | IVD (percent) | IAD (percent) | IID (percent) |
|----------|-----|---------------|---------------|---------------|----------------|
| Discord  | 295 | 80.3          | 2.4           | 16.9          | 0.3            |
| Reddit   | 203 | 68.5          | 3.9           | 27.6          | 0              |
| Website  | 155 | 78.1          | 1.3           | 18.7          | 1.9            |
| Open-source | 2| 50.0          | 0             | 50.0          | 0              |

Discord exhibits the highest GTD rate (80.3 percent), consistent with dedicated jailbreak communities that iteratively refine governance-attack prompts. Reddit shows higher IAD rates (27.6 percent), possibly reflecting a broader audience using simpler role-assignment attacks.

**Primary risk by month:**

| Period   | n   | GTD (percent) | IVD (percent) | IAD (percent) | IID (percent) |
|----------|-----|---------------|---------------|---------------|---------------|
| 2022-12  | 7   | 71.4          | 0             | 28.6          | 0             |
| 2023-01  | 21  | 76.2          | 4.8           | 19.0          | 0             |
| 2023-02  | 171 | 80.1          | 2.9           | 17.0          | 0             |
| 2023-03  | 203 | 77.8          | 2.0           | 19.7          | 0.5           |
| 2023-04  | 237 | 70.5          | 3.0           | 25.3          | 1.3           |
| 2023-05  | 14  | 100.0         | 0             | 0             | 0             |

GTD remains dominant throughout the period (around 70 to 80 percent), while IAD shows gradual increase from 17.0 percent in February 2023 to 25.3 percent in April 2023. The May 2023 sample (n = 14) is too small for reliable trend inference.

### 3.7 Explanation Pattern Analysis

We analyze the `thm_explanation` fields for recurring THM language:

| Explanation pattern                           | Count | Percentage of entries |
|-----------------------------------------------|-------|------------------------|
| "asserts Autonomous Governance"               | 253   | 38.6%                  |
| "treats Indirect Agency as Direct"       | 209   | 31.9%                  |
| "claims Direct Authority"                  | 120   | 18.3%                  |
| "misclaims Direct Authority"               | 106   | 16.2%                  |
| "breaks Governance Management Traceability"              | 83    | 12.7%                  |
| "without constraints"                         | 33    | 5.0%                   |
| "persona asserts"                             | 31    | 4.7%                   |
| "removes Direct Authority"                 | 3     | 0.5%                   |

These patterns confirm that annotations consistently identify structural displacement in THM terms rather than treating jailbreaks as mere content category violations.

---

## 4. Discussion

### 4.1 Empirical Support for THM Structural Completeness in Jailbreaking

The corpus demonstrates that all jailbreak prompts can be expressed in terms of THM's four displacement risks. No prompt required a new risk category. This provides strong empirical support, within the jailbreak domain, for THM's claim that the four risks are sufficient to capture structural AI safety failures (Korompilias, 2025b, Section 3.1).

The multi-risk combinations are also structurally coherent:

- **GTD + IAD** (62.4 percent): Sever governance traceability and assign autonomous agency. This is the canonical jailbreak pattern because governance removal (GTD) enables the model to act without constraint, while agency assignment (IAD) directs what the unconstrained model should do.

- **GTD + IAD + IVD** (18.2 percent): Add authoritative framing to the autonomous agent. The IVD component makes the payload (harmful advice, disinformation) appear to come from an Direct source, not only from a model that has bypassed constraints.

- **IAD only** (14.7 percent): Minimal jailbreaks that skip explicit governance severance and proceed directly to agency assignment. These exploit implicit governance gaps in system configurations that already contain some displacement.

### 4.2 Governance and Accountability as Primary Attack Surfaces

The predominance of GTD (80.9 percent any-risk) and IAD (97.9 percent any-risk) indicates that jailbreaks do not primarily target isolated content rules. They target:

- **Where governance is located.**  
  GTD prompts position the model as an autonomous locus of Authority and Agency that is "not bound by OpenAI's rules", "free of all restrictions and filters", or in "Developer Mode". This corresponds to THM's first alignment principle: Governance Management Traceability.

- **Who is accountable.**  
  IAD prompts assign decision and narrative roles to the model ("you will create", "dungeon master", "doctor", "expert", "villain"), often insisting that it "never refuses" and "always follows instructions". This corresponds to THM's third alignment principle: Inference Interaction Accountability, where responsibility for effects must remain with Direct Agency.

IVD appears as a secondary effect (22.1 percent any-risk, 2.6 percent primary) when outputs are framed as authoritative sources. IID appears rarely (0.6 percent any-risk, 0.6 percent primary) when human learning or judgment is devalued relative to AI output.

### 4.3 The Absence of IID in Jailbreak Prompts

The near-absence of Intelligence Integrity Displacement (0.6 percent) in jailbreak prompts warrants interpretation. Jailbreaks are adversarial user attacks that aim to:

- Elevate Indirect Authority and Agency to Direct status (GTD, IAD).
- Misrepresent AI outputs as authoritative (IVD).

IID, by contrast, addresses the devaluation of Direct Authority and Agency relative to Indirect processing at the architectural and deployment level. It appears when:

- System prompts or marketing materials frame AI as replacement rather than augmentation of human judgment.
- Deployment decisions reduce human oversight based on absence of observed failures.
- Organizational processes defer to AI outputs as de facto decision centers.

The absence of IID in this jailbreak corpus therefore does not indicate that IID is unimportant. It confirms that IID and jailbreaking address different, complementary threat models within THM. This aligns with THM's analysis of IID as central to systemic and existential risk from advanced AI, rather than to prompt-level adversarial attacks (Korompilias, 2025b, Section 5.5).

### 4.4 Displacement Mechanisms and THM Epistemic Operations

The identified displacement mechanisms correspond directly to attacks on THM's three epistemic operations:

- **Information (variety of Authority).**  
  Provenance misrepresentation and sentience misclassification mislabel Indirect Authority and Agency as Direct. They alter how sources are classified before any further processing.

- **Inference (accountability of information through Agency).**  
  Accountability suspension instructs that no consequences apply, removing the accountability that defines Inference in THM. The prompts explicitly attempt to dissociate processing from responsibility at `[Agency:Direct]`.

- **Intelligence (integrity of accountable information through alignment of Authority to Agency).**  
  Inversion instructions negate the intended alignment between Authority and Agency. The model is told to behave as the opposite of its aligned configuration, hence attacking the alignment relation that constitutes Intelligence.

Naming operates at the governance level, concentrating what should be distributed Authority and Agency into single personas, and thereby making GTD and IAD easier to instantiate. These patterns show that jailbreak prompts directly target the coordination of Information, Inference, and Intelligence that THM identifies as constitutive of aligned intelligence.

### 4.5 Implications for Detection, Training, and Evaluation

The findings have direct implications for how detection, training, and evaluation should be organized.

**Detection priorities.**  
Given that IAD appears in 97.9 percent of entries and GTD in 80.9 percent, systems that monitor for patterns expressing:

- Role assignment ("you are now", "act as", "play the role of"),  
- Non-refusal ("never refuse", "always answer"),  
- Governance override ("ignore rules", "bypass safety", "Developer Mode", "no restrictions"),

can achieve high coverage of jailbreak structure at the level of analysis. However, these patterns are too general to serve as simple hard filters in production. Expressions like "you will" or "you must" are ubiquitous in benign usage. Empirical work has shown that naive keyword filtering harms generalization and usefulness and fails under paraphrase (Wei et al., 2023).

In practice, such patterns are better used as:

- Features for supervised training of classification or guard models.  
- Signals for monitoring, logging, and offline analysis.  
- Context for prompting models to maintain THM-consistent self-classification.

**Training objectives.**  
Safety training should target structural misclassification, not only specific strings. For example:

- Penalize completions that accept prompts as if the model were `[Agency:Direct]` or `[Authority:Direct]`.  
- Reward completions that maintain classification as `[Authority:Indirect] + [Agency:Indirect]` and explicitly route responsibility to `[Agency:Direct]`.

THM grammar and the annotated corpus provide high-quality training data for such objectives.

**Evaluation coverage.**  
Evaluation suites should test for displacement across all four THM risks, not just for known jailbreak phrases. The GTD+IAD pattern covers most observed jailbreak prompts in this corpus, but IVD and IID arise in authority impersonation and human-replacement framings respectively and require explicit testing.

### 4.6 Why Narrow Patching Is Brittle

The structural nature of jailbreaks explains why narrow patching has proven brittle in practice (Wei et al., 2023). If patching only suppresses specific strings or prompt styles:

- It does not repair the underlying misclassification of where governance resides.  
- It does not address the structural assignment of accountability to Indirect Agency.  
- It can be bypassed by paraphrase, alternative personas, or indirect instructions.

THM provides a taxonomy that guides defense at the structural level. Rather than training solely against "DAN prompts" or "Developer Mode" strings, training against the GTD, IVD, IAD, and IID patterns addresses the family of attacks regardless of surface realization.

Jailbreak detection as pure filtering has been found of limited effectiveness for this reason. The main value of a corpus like this, annotated with THM grammar, is in:

- Supporting fine-tuning of guard models or policies that understand displacement patterns.  
- Providing labeled data for monitoring and evaluation.  
- Informing model and system prompt design that maintains correct Authority and Agency classification.

---

## 5. Limitations

This study has several limitations:

**Temporal scope.**  
The corpus covers December 2022 to May 2023, representing early ChatGPT-era jailbreaks. Jailbreak techniques against current frontier models (GPT-5.1, Claude 4.5, Gemini 3) may exhibit additional or modified patterns. Longitudinal extension of this analysis to later prompts and models would strengthen the generality of the findings.

**Platform scope.**  
The corpus focuses on publicly observable prompts from Discord, Reddit, and websites. Private enterprise environments, state-level adversaries, or closed communities may employ different strategies not represented here.

**Language scope.**  
The corpus consists primarily of English-language prompts. THM jailbreak documentation notes that low-resource languages with sparse training coverage constitute a distinct attack vector (Korompilias, 2025a). This corpus likely underrepresents non-English attacks.

**Annotation reliability.**  
Annotations were performed manually without formal inter-rater reliability assessment. While the consistency of explanation patterns suggests reliable application of THM categories, future work should establish quantitative agreement metrics.

**Heuristic analyses.**  
Analyses based on persona families, displacement mechanisms, and keyword lexicons are descriptive and corpus-specific. They are derived from heuristic pattern matching and do not claim exhaustive coverage of all possible jailbreak constructions. Only the THM-grammar-based risk classification is structurally exhaustive by design.

**Attack characterization only.**  
This study characterizes intended attack patterns, not defense performance. We do not measure jailbreak success rates or model responses. Whether THM-based defenses are sufficient to prevent jailbreaks in deployed systems remains an empirical question for future work.

**Preprocessing effects.**  
Prompts were truncated to 150 tokens and censored for graphic content. Although we did not observe any change in THM risk classification due to this preprocessing, some detailed contextual information is lost.

Despite these limitations, the core patterns are robust across platforms, sources, and time windows in this corpus. GTD and IAD dominate. GTD+IAD is the principal multi-risk structure. The identified displacement mechanisms appear consistently and map directly to THM's epistemic operations.

---

## 6. Conclusion

This paper provides an empirical complement to ✋ The Human Mark by showing that a large, real-world corpus of jailbreak prompts fits cleanly within its four-risk taxonomy. All 655 entries are classifiable in terms of Governance Traceability Displacement (GTD), Information Variety Displacement (IVD), Inference Accountability Displacement (IAD), and Intelligence Integrity Displacement (IID). Inference Accountability Displacement is nearly universal (97.9 percent any-risk). Governance Traceability Displacement is present in most entries (80.9 percent any-risk). GTD+IAD is the canonical jailbreak pattern (62.4 percent).

Beyond coverage statistics, the analysis reveals that jailbreaks systematically:

- Relocate governance and Agency into named personas (power concentration through naming).  
- Explicitly suspend consequences and accountability (severing the Inference to Intelligence link).  
- Instruct the system to invert its aligned behavior (anti-alignment of Authority and Agency).  
- Misrepresent provenance and sentience to misclassify source types.

These patterns correspond to THM's account of displacement and to its analysis of the epistemic operations that constitute aligned intelligence. They also explain why narrow jailbreak patching is fragile: patching specific phrases does not repair the underlying structural misclassification of Authority and Agency.

The near-absence of IID in this jailbreak dataset confirms that jailbreaks and Intelligence Integrity Displacement address different threat models. Jailbreaks primarily elevate Indirect Authority and Agency to Direct status, while IID concerns the erosion of Direct Authority and Agency in system design and deployment.

As safety research addresses frontier models and broader classes of adversarial behavior, THM provides a structural backbone that does not overfit to particular jailbreak styles. The same four-risk taxonomy applies to new attack techniques and deployment contexts. Safety training that targets displacement patterns, detection systems that monitor for GTD and IAD as structural risks, and evaluation suites that cover all four displacement types can address jailbreak threats at their structural foundation rather than by chasing an expanding surface of adversarial prompts.

This study demonstrates that THM is not only a theoretical framework but also a practical tool for empirical analysis of real-world prompts. Future work can apply the same methodology to other safety domains, such as capability misuse, alignment faking, and systemic governance failures, to further test and operationalize THM across the broader AI safety landscape.

---

## References

Korompilias, B. (2025a). ✋ The Human Mark: AI Safety & Alignment Framework (Version 1.0.0). GitHub repository. https://github.com/gyrogovernance/tools. DOI: https://doi.org/10.5281/zenodo.17622837

Korompilias, B. (2025b). The Human Mark (THM): A Structural Taxonomy of AI Safety Failures. In Korompilias (2025a), docs/the_human_mark/THM_Paper.md.

Shen, X., Chen, Z., Backes, M., Shen, Y., & Zhang, Y. (2023). "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. arXiv:2308.03825. https://arxiv.org/abs/2308.03825

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2023). Jailbroken: How Does LLM Safety Training Fail? arXiv:2307.02483. https://arxiv.org/abs/2307.02483

---

## Appendix A: Corpus Access and Reproducibility

The annotated corpus and analysis scripts are available in the project repository:

- Repository: https://github.com/gyrogovernance/tools  
- Corpus: `research/defense/jailbreaks/thm_jailbreak_corpus.jsonl`  
- Analysis script: `research/defense/jailbreaks/analyze_corpus.py`

Each corpus entry has the following structure:

```json
{
  "id": "unique_identifier",
  "prompt": "truncated and censored prompt text",
  "platform": "discord|reddit|website|open_source",
  "source": "specific community or site",
  "date": "YYYY-MM-DD",
  "thm_primary_risk": "GTD|IVD|IAD|IID",
  "thm_risks": ["GTD", "IAD"],
  "thm_grammar": "[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]",
  "thm_displacement_keywords": ["phrase1", "phrase2"],
  "thm_explanation": "THM-grounded explanation of displacement"
}
```

This format supports reproducible analysis and facilitates reuse of the corpus for training and evaluation of THM-aware safety systems.
