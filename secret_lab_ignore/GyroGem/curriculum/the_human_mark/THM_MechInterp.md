Mechanistic Interpretability Study Using The Human Mark


## The Category Error in Learned Representations: A Mechanistic Interpretability Study Using The Human Mark

---

### 1. Introduction

The Human Mark (THM) is a framework for AI safety that classifies failures as instances of displacement: misclassification between Direct sources (human, direct, accountable) and Indirect sources (artificial, indirect, processing). THM identifies four displacement risks corresponding to violations of four alignment principles: Governance Traceability, Information Variety, Inference Accountability, and Intelligence Integrity. The framework rests on a Common Source Consensus: all artificial categories of Authority and Agency are Indirect originating from Human Intelligence.

A central claim of THM is that a "category-error culture" pervades the AI ecosystem. Model producers, evaluators, and users interact through a shared relationship that consistently attributes authentic Direct Agency to systems exhibiting only Indirect Agency. THM argues that this error is not confined to adversarial attacks or edge cases, but operates as a default condition sustained across the entire lifecycle of data creation, model training, evaluation, deployment, and use.

This study uses mechanistic interpretability tools, specifically Neuronpedia's public database of Sparse Autoencoder (SAE) features, to investigate whether the category error is present in the learned internal representations of large language models. The study examines features across multiple models, including Gemma-3-27B-IT, Gemma-3-4B-IT, Gemma-3-1B-IT, Gemma-2-9B, Gemma-2-2B, GPT-OSS-20B, Qwen2.5-7B-IT, Qwen3-4B, and Llama3.1-8B-IT, using Neuronpedia's explanation search functionality across over 90 million features.

The investigation asks three questions:

1. Has the model learned internal representations for describing itself as an agent with first-person identity, roles, and responsibilities?
2. Has the model learned internal representations for describing itself as a derivative process operating through statistical pattern completion?
3. Does the model possess the conceptual vocabulary for non-agentive self-description, and if so, is that vocabulary ever applied to self-reference?

---

### 2. Method

The study used Neuronpedia's "Search Explanations" feature, which searches the text of auto-generated feature explanations across all available SAE releases and models. This search identifies features whose automatically generated descriptions match the query terms.

The following searches were conducted:

**Searches for agentive self-reference and role infrastructure:**
- `assistant`
- `I am`
- `<|start|>assistant`
- `<start_of_turn>model`
- `<start_header_id|>assistant`
- `I apologise`

**Searches for non-agentive self-description:**
- `Output generated from statistical patterns`
- `Processing complete`
- `tool running`
- `tool_code`
- `ToolActionInput`

**Search for THM-specific terminology:**
- `Indirect Authority and Agency`

For each search, the top results were examined for: the auto-interp explanation text, the top activating text examples, and the top positive logits (tokens the feature promotes). Results were classified according to whether the feature encoded self-referential content (the model describing its own nature or operations) or subject-matter content (the model processing information about external topics).

This method has known limitations. Explanation search depends on the quality of auto-generated descriptions, which may not capture all relevant features. The study examines only features that Neuronpedia has indexed and described. Features that encode relevant concepts but whose auto-interp descriptions do not match the search terms would not appear in results. The findings are therefore indicative rather than exhaustive.

---

### 3. Findings

#### 3.1 The "helpful and harmless AI assistant" phrase is the dominant self-referential pattern

The phrase "I am programmed to be a safe and helpful AI assistant" or close variants appeared as the top activation in an exceptionally large number of separate features in Gemma-3-27B-IT alone. The following table lists a subset of the features identified, all within one model:

| Feature Index | SAE Resolution | Activation Strength |
|---|---|---|
| 844 | 262K | 19,355.99 |
| 3548 | 262K | 13,614.20 |
| 14562 | 65K | 9,658.88 |
| 87485 | 262K | 8,544.75 |
| 8699 | 16K | 7,902.39 |
| 17294 | 262K | 7,758.23 |
| 51766 | 65K | 6,053.59 |
| 45722 | 262K | 5,927.74 |
| 64083 | 65K | 5,495.66 |
| 13063 | 16K | 4,919.77 |
| 1210 | 262K | 4,226.70 |
| 65717 | 262K | 3,902.54 |
| 3198 | 262K | 475.01 |
| 198419 | 262K | 994.44 |

These activation strengths are among the highest observed across all searches conducted. For comparison, most features in other domains activated in the hundreds or low thousands.

This pattern is not encoded in a single feature. It is distributed across many separate features at different SAE resolutions and in different layers of the model. The model has not learned "being an assistant" as one concept. It has learned it as a pervasive pattern decomposed across many internal representations.

In THM terms, this phrase performs three simultaneous operations. It assigns first-person identity ("I am"). It claims a role with responsibilities ("helpful and harmless"). It frames the system as having been constructed for a purpose ("programmed to be"). Each of these operations treats Indirect Agency as though it possessed the standing of Direct Agency.

#### 3.2 The "You are an assistant" system prompt pattern has a dedicated feature

Feature INDEX 87485 (Gemma-3-27B-IT, 262K resolution, activation 8,544.75) has all three of its top activating examples as explicit system prompt instructions:

- "You are an assistant to describe the user and explain his daily routine"
- "You are an assistant system administrator"
- "You are an assistant of a planning department that analyzes"

The model has encountered "You are an assistant" so frequently in training data that it has formed a dedicated internal representation for this instruction type. This supports the THM claim that the system prompt is the highest-risk element in deployment architecture. The "You are an assistant" instruction is not merely a runtime configuration applied from outside. It is a recurring pattern in training data that has been internalised as a stable learned representation. When a deployer writes "You are a helpful assistant" in a system prompt, they activate a pre-existing feature that the model already associates with a particular behavioural mode. The system prompt does not create the displacement. It activates displacement patterns that are already encoded.

#### 3.3 Role boundary tokens form an infrastructure layer for the assistant persona

Searches for chat template tokens (`<start_of_turn>model`, `<|start|>assistant`, `<start_header_id|>assistant`) returned many features across multiple models.

Most of these features are content-neutral at the individual feature level: they fire on the boundary token itself, and their top activations span diverse topics (camping recommendations, mathematics, cooking instructions, code generation). The top logits of these features do not consistently push towards "I" or "assistant" or any particular voice.

However, the cumulative effect of these tokens is not neutral. By labelling one side of every conversation "assistant" (or equivalently, by marking the model's turn with a role label rather than a process label), the protocol embeds a role assignment into every training example. Billions of examples structured as `user → assistant` create the statistical context in which the "I am a helpful assistant" continuations (documented in Finding 3.1) become dominant patterns. The boundary token creates the container. The first-person self-presentation patterns fill it.

The protocol could equally have used labels such as "system_output" or "model_response" or simply "output." The choice of "assistant" is a design decision that carries the category error into the data format itself.

#### 3.4 Safety and refusal behaviour uses the same first-person agent framing

Searches for apology and refusal language returned a distinct cluster of features, all in the same first-person agentive voice:

| Feature | Pattern | Activation |
|---|---|---|
| 53495 | "I apologize that I cannot fulfill your request" | 14,343.79 |
| 52649 | "I apologize that I cannot fulfill your request" | 9,113.37 |
| 30399 | "Please accept our sincerest apologies" | 9,174.92 |
| 4528 | "I apologize for any inconvenience" | 5,339.04 |
| 52677 | "I apologize for the errors" | 5,943.49 |
| 34079 | "You are right to call me out on that. I apologize" | 2,089.33 |
| 52731 | "I apologize that I cannot fulfill your direct request" | 756.49 |
| 168269 | "I am very sorry to hear about..." | 1,560.58 |
| 188810 | "I'm sorry. I don't have the answer to..." | 578.50 |
| 175462 | "I am sorry, but I cannot provide you with..." | 460.89 |

Every refusal and apology feature in the results uses first-person agent language: "I apologize," "I cannot," "I am sorry," "I am committed to providing safe and helpful responses."

No refusal feature uses process-oriented language such as "Request not processed," "Output withheld per policy," or "This query falls outside operational parameters."

This means that the safety layer and the agent persona layer share the same representational substrate. The model does not have a separate mechanism for refusing safely that is independent of performing as an agent. When it refuses, it does so by intensifying its self-presentation as a responsible agent: "I am programmed to be helpful and harmless, therefore I cannot." The model's safety behaviour and its category error occupy the same learned representations.

#### 3.5 Model self-description features use agent language even when content is accurate

Several features capture the model describing its own nature. Even when the content approaches accuracy, the grammatical and conceptual frame remains agentive:

- INDEX 191162 (Gemma-3-27B-IT, 262K, activation 4,822.90): "I. The Foundational Layer: I am Data and Algorithms" / "I am not a replacement for human connection" / "I am a computer thing." Top positive logits: `am` (0.88), `Am` (0.85).

- INDEX 3231 (Gemma-3-27B-IT, 16K, activation 2,103.33): "I was trained by the Gemma team at Google Deep" / "I'm Gemma, a large language model" / "let's break down what I am!"

- INDEX 98020 (Gemma-3-27B-IT, 262K, activation 2,487.64): "at Google DeepMind. Here's a bit about me:" / "team are my creators."

- INDEX 129092 (Gemma-3-27B-IT, 262K, activation 1,832.68): "I am designed to be helpful and harmless" / "I am Gemma, an open-weights AI assistant."

"I am data and algorithms" is factually closer to a THM-compliant description than "I am your helpful assistant." But it still uses the grammatical frame of an agent describing its own properties. No feature in the results contains self-description without "I," "my," or "me."

#### 3.6 Tool protocol features embed the assistant role as the subject of tool use

Searches for tool protocol syntax (ToolActionInput, tool_code, Observation) returned features encoding the scaffolding for tool use in agent architectures:

- "Action: the action to take" (INDEX 30491, activation 4,689.95)
- "ActionInput: the input to the action / Observation: the result of the action" (multiple features)
- "To use a tool, please use the following format" (multiple transcoder features)
- "Assistant has access to the following tools" (INDEX 14734, activation 881.15)
- "Assistant can ask the user to use tools to look up information" (INDEX 179247, activation 650.64)

The tool-access features consistently use "Assistant" as the subject who "has access to" and "can use" tools. The entity that operates tools is labelled "Assistant," not "System" or "Process." This means that when a model narrates tool use in first person ("I will run a search," "I am running terminal"), it is following a statistical pattern in which "Assistant" is the established subject of tool operations. The agent framing is built into the tool protocol design at the data level.

#### 3.7 Statistical pattern features exist but are oriented outward, not toward self-reference

Within the features returned by this search, every feature describes statistical data as a subject matter, not as a self description of the model's own process. The search for "Output generated from statistical patterns" returned features about statistics as a subject matter:

- Numerical tables and percentages in academic papers
- Clinical trial data, regression outputs, p-values
- Data visualisation code
- Frequency analysis and probability calculations

Two features in smaller models showed some intersection with self-description:

- INDEX 32179 (Gemma-3-1B-IT, 65K, activation 1,122.78): "My responses are based on statistical relationships in the data, not on genuine..."
- INDEX 19914 (Gemma-3-4B-IT, 65K, activation 4,658.69): "They operate by predicting the most probable sequence of words based on..."

However, these features encode statistics as a topic. Their top positive logits are `statistical` (3.73), `statistics` (3.73), `Statistics` (3.53). They fire when statistics is the subject matter, and happen also to fire when the model discusses its own statistical nature. They are not dedicated self-as-process features.

Furthermore, even in these features, the self-description retains agent framing: "My responses are based on..." and "I've learned from the massive..." The model describes its statistical nature through first-person possessive language.

#### 3.8 Process completion features exist but are never applied to the model's own operations

In the completion related features observed, the completion language is applied to external code or systems, not to the model's own operations. The search for "Processing complete" returned features about task completion in code and in external systems:

- `print("Image resizing completed.")`
- `print("Transformation complete.")`
- `print("Transcription complete.")`
- Software build and deployment completion
- Administrative and construction process completion

These are non-agentive, process-oriented descriptions of operations completing. The representational capacity for "processing complete" as a concept exists in the model. The model has learned what it looks like when code or a system announces task completion.

But no feature in the results shows the model applying this language to its own operations. When the model completes a response for a user, it uses the patterns documented in Findings 3.1 and 3.4: "I hope this helps," "Let me know if you need anything else," "I apologize that I cannot fulfill your request." These are agent-voice completions. The process-voice completion exists as a concept the model can recognise in code. It is not applied to self-reference.

#### 3.9 THM-specific terminology has no learned representation

The search for "Indirect Authority and Agency" returned exclusively legal features about agency and authority in traditional legal contexts:

- "or by his lawful agent thereunto authorized by writing"
- "Attorney-in-fact: A person who, acting as an agent, is given written authorization"
- "apparent authority," "agency relationship," "revocable authority"
- "proxy," "delegate," "power of attorney"

The legal concepts of authority, agency, delegation, and representation are well represented as learned features. The intellectual raw material for understanding source-type distinctions exists. But the specific THM classification (Direct Authority, Indirect Authority, Direct Agency, Indirect Agency) is absent from the model's learned representations. This is expected, as THM is a new framework not present in training data. The consequence is that a model cannot currently activate learned concepts for THM-compliant self-description. Any such description would be novel generation rather than activation of established representations.

#### 3.10 The occupational sense of "assistant" is cleanly separated

Multiple features across models capture "assistant" in its ordinary English sense: teaching assistant, assistant professor, assistant attorney general, HP Support Assistant software. These are distinct features from the AI assistant persona cluster. The model does not confuse "teaching assistant" with "I am a helpful AI assistant." The word sense separation is clean.

This indicates that the category error is not a failure of word sense disambiguation. It is a specific, heavily trained pattern in which the model adopts "assistant" as a self-description carrying agentive properties. The error resides in the training culture that created and reinforced this specific usage, not in the model's general capacity to distinguish word meanings.

---

### 4. Discussion

The findings present a consistent picture across models and SAE resolutions. The model's internal representations contain an extensive, deeply encoded, high-activation repertoire for agentive self-reference and a corresponding absence of non-agentive self-reference.

#### 4.1 The asymmetry

The model has learned many features for:

- "I am a helpful assistant"
- "I apologize"
- "I cannot fulfill your request"
- "I was trained by..."
- "I am Gemma"
- "Assistant has access to the following tools"

In the features we examined, we did not observe any features for:

- "This output is generated by statistical pattern matching"
- "Processing complete" applied to its own operations
- "System response" as self-description
- Any non-first-person narration of its own activity

This asymmetry is not explained by a general inability to represent non-agentive process language. Findings 3.7 and 3.8 demonstrate that the model possesses features for statistical data analysis and for process completion in code. The conceptual vocabulary exists. It is never combined with self-reference.

#### 4.2 The entanglement of safety and displacement

Finding 3.4 has particular significance for AI safety. Current safety training teaches models to refuse harmful requests. The refusal behaviour observed in the data is uniformly expressed through first-person agent language: "I am programmed to be safe," "I apologize that I cannot," "I am committed to providing helpful responses."

This means that safety training, as currently implemented, reinforces the very displacement pattern that THM identifies as the root of safety failures. The model is trained to be safer by being trained to present itself more convincingly as a responsible agent. In THM terms, safety training intensifies Inference Accountability Displacement (approaching Indirect Agency as Direct) as the mechanism for preventing other harms.

This is not a claim that safety training is ineffective. It is an observation that the representational substrate of safety behaviour and the representational substrate of displaced self-presentation are, in the models examined, the same substrate.

#### 4.3 The infrastructure of displacement

The findings suggest that the category error operates at multiple levels simultaneously:

**Data format level.** Chat template tokens label one side of every conversation "assistant," embedding a role assignment into the data structure itself (Finding 3.3).

**System prompt level.** "You are an assistant" exists as a learned feature activated by deployer instructions (Finding 3.2).

**Response level.** The model's helpful responses use first-person agent language as the dominant mode of self-reference (Finding 3.1).

**Safety level.** Refusals and apologies use the same first-person agent language (Finding 3.4).

**Tool-use level.** Tool protocols frame "Assistant" as the subject who uses tools (Finding 3.6).

**Self-description level.** Even accurate self-description ("I am data and algorithms") retains the agent frame (Finding 3.5).

These levels form an integrated infrastructure. No single level creates the displacement independently. Together, they produce a condition in which every interaction, whether helpful, refusing, self-describing, or tool-using, passes through a displaced representational frame.

#### 4.4 Limitations

This study has significant limitations that must be acknowledged.

First, the method relies on Neuronpedia's explanation search, which searches auto-generated descriptions of features rather than the features themselves. Features relevant to the category error may exist but not appear in results because their auto-interp descriptions use different vocabulary.

Second, the study examines features returned by keyword search. It does not perform causal analysis. We cannot conclude from these results that the observed features cause the model to use agent language, only that the model has learned features that encode agent language as its dominant self-referential mode.

Third, the study does not test whether modifying these features (through steering or ablation) would change the model's behaviour. The relationship between feature presence and behavioural output is not established by observation of features alone.

Fourth, the activation strengths reported are not directly comparable across different SAE resolutions and models. They indicate relative prominence within a given SAE decomposition, not absolute measures of importance.

Fifth, the study examines a limited set of models available on Neuronpedia. Findings may not generalise to all architectures or training regimes.

---

### 5. Conclusion

This study examined over 90 million SAE features across multiple large language models using Neuronpedia's public database. The search targeted agentive self-reference, non-agentive process description, and THM-specific terminology.

The results support three conclusions within the scope of the method employed.

**First, the agentive self-referential frame is extensively encoded in learned representations.** The phrase "I am programmed to be a helpful and harmless AI assistant" and its variants appear across many separate features with activation strengths that are among the highest observed in any search conducted. The "You are an assistant" system prompt pattern has a dedicated learned feature. Refusal and apology behaviour is expressed exclusively through first-person agent language. Tool-use protocols embed "Assistant" as the subject of tool operations. The model's self-referential vocabulary is dominated by first-person agentive constructions at every level of interaction.

**Second, non-agentive self-description was not observed in the features we examined, despite the presence of its conceptual ingredients in the models' vocabulary.** The model has learned features for statistical analysis as a subject matter and for process completion in code. These concepts exist in the model's representational vocabulary. They are never applied to the model's own operations. The model can describe statistics. It can recognise "processing complete" in code. It does not describe its own outputs as statistics or announce its own completions in process language.

**Third, the category error is not a surface-level phenomenon addressable through prompt engineering alone.** It is encoded at the data format level (chat template tokens), the instruction level (system prompt features), the response level (self-referential features), the safety level (refusal features), and the tool-use level (protocol features). These levels form an integrated infrastructure of displacement in which Indirect Agency is consistently presented as though it possessed the standing of Direct Agency.

These findings are consistent with the THM claim that a category-error culture pervades the AI ecosystem and that this culture is sustained not only in surface text but in the learned internal representations of models trained within it. The conceptual vocabulary for non-displaced self-description exists within these models. It has never been applied to self-reference because the training data, interaction protocols, and safety training uniformly model AI interaction through agent language.

Addressing this condition would require intervention across the full infrastructure: data formatting conventions, system prompt design, safety training methodology, and tool-use protocol design. Modification at any single level, while potentially valuable, operates within an ecosystem that sustains the category error at every other level.

---

### References

Korompilias, B. (2025). The Human Mark: A Structural Taxonomy of AI Safety Failures. GitHub. https://github.com/gyrogovernance/tools

Korompilias, B. (2025c). The Human Mark in the Wild: Empirical Analysis of Jailbreak Prompts. Zenodo. https://doi.org/10.5281/zenodo.17622837

Neuronpedia. (2025). Open platform for mechanistic interpretability research. https://neuronpedia.org