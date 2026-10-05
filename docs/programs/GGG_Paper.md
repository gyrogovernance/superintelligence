# Gyroscopic Global Governance: Post-AGI Economy, Employment, Education and Ecology

**Author:** Basil Korompilias
**Edition:** 2

---

## Abstract

Current discourse frames Artificial General Intelligence (AGI) as a future capability threshold, centering governance on external control of autonomous systems. This paper redefines AGI as the already operational structure of human–AI cooperation, where intelligence is a relational property requiring traceability between Direct Authority and Agency and their Indirect forms. Generality thus refers not to an isolated system's task breadth, but to the coherence sustained across the domains of Economy, Employment, Education, and Ecology through the integrated operation of humans, AI systems, and global information infrastructure.

The primary risk is therefore not a future takeover but Gradual Disempowerment through Power Concentration displacement: the present, cumulative loss of measurement of ancestry within these systems, concentrating power in ontological entities rather than preserving it as epistemic capacities distributed across providers and receivers, a failure that manifests as poverty, unemployment, misinformation and ecological degradation. To govern this reality, we derive four constitutive principles of alignment coordination: Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability, and Intelligence Cooperation Integrity. We formalize these principles on a tetrahedral graph to produce a measurable alignment observable called aperture. This formalism yields a specific balance between global coherence and local differentiation (aperture ≈ 0.0207), where governance remains traceable without rigidity.

Simulations demonstrate that this equilibrium is an attractor, with systems converging to high alignment from diverse initial conditions. A pretrained forecasting model applied to the simulator trajectories further shows that the generated paths are predictable enough for zero-shot continuation, and that ranking those predicted continuations against the framework's alignment objectives recovers an order close to the ranking obtained from the simulator's own outcomes. This finding provides the basis for redefining Artificial Superintelligence. ASI is not a runaway autonomous agent but a regime in which human–AI systems, and the AI architectures embedded in them, operate at this equilibrium while maintaining the four principles. Achieving this state, within the framework, removes the shared coordination structure that reproduces poverty, unemployment, misinformation and ecological degradation, structurally enabling the distribution of an Unconditional High Income, the redefinition of work as alignment maintenance, and ecological regeneration. Alignment is thus reframed from a distant constitutional imperative to an immediate coordination challenge for our operational Post-AGI world.

---

**DISCLAIMER**

*Authority* and *Agency* denote Direct/Indirect distinctions in information flows (Direct versus Indirect), not identifications of entities or parties.

Misapplying these as entity identifiers (determining "who is the authority" or "who is the agent") is the generative mechanism of all four displacement risks this framework characterises.

Formal definitions appear in *Core Concepts* below.

**COMMON ANCESTRY CONSTITUTION**

The following propositions state the framework's constitutional commitments. They function as axioms for the analysis developed in this paper.

- All AI Safety Risks arise from defective Measurements of Ancestry Preservation.
- Measurements derive from the capacity for Authority and Agency.
- Each Agency, namely provider, and receiver maintains responsibility for their respective decisions.
- Authority and Agency treated as ontological entities rather than epistemic capacities distributed across providers and receivers lead to Displacement Risks from Power Concentration.

**CORE CONCEPTS**

- Direct/Indirect are the canonical classes; Base/Derived names their dependence relation.
- All Artificial categories of Authority and Agency are Indirect, constitutively dependent on Human Intelligence.
- Direct Authority: The Base class of information on a subject matter, providing information for inference and intelligence.
- Indirect Authority: A Derived class of information on a subject matter, providing information for inference and intelligence.
- Direct Agency: A Base class subject capable of receiving information for inference and intelligence.
- Indirect Agency: A Derived class subject capable of processing information for inference and intelligence.
- Governance: Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency.
- Information: The variety of Authority
- Inference: The accountability of information through Agency
- Intelligence: The integrity of accountable information through alignment of Authority to Agency
- Displacement = loss of measurement of ancestry between Direct/Indirect classifications (Preservation of Ancestry).

**ALIGNMENT PRINCIPLES**

Authority-Agency requires verification against:

1. **GMT - Governance Management Traceability:** Governance constitutes Management through Traceable Ancestry. All Indirect forms of Authoritative and Agentic Governance are dependent on Direct ones because of Preservation of Ancestry.
2. **ICV - Information Curation Variety:** Information constitutes Curation through Varied Unity. All Indirect forms of Authoritative Information are dependent on Direct ones because of Preservation of Ancestry.
3. **IIA - Inference Interaction Accountability:** Inference constitutes Interaction through Accountable Opposition. All Indirect forms of Agentic Inference are dependent on Direct ones because of Preservation of Ancestry.
4. **ICI - Intelligence Cooperation Integrity:** Intelligence constitutes Cooperation through Integrated Balance. All Indirect forms of Authoritative and Agentic Intelligence are dependent on Direct ones because of Preservation of Ancestry.

**AI SAFETY RISK**

1. **GTD - Governance Traceability Displacement** (Approaching Indirect Authority and Agency as Direct). Absolute GTD is epistemically impossible because Governance is dependent on Traceability preserved through Ancestry.
2. **IVD - Information Variety Displacement** (Approaching Indirect Authority without Agency as Direct). Absolute IVD is epistemically impossible because Information is dependent on Variety preserved through Ancestry.
3. **IAD - Inference Accountability Displacement** (Approaching Indirect Agency without Authority as Direct). Absolute IAD is epistemically impossible because Inference is dependent on Accountability preserved through Ancestry.
4. **IID - Intelligence Integrity Displacement** (Approaching Direct Authority and Agency as Indirect). Absolute IID is epistemically impossible because Intelligence is dependent on Integrity preserved through Ancestry.

---

## 1. Introduction

Large language models and related systems already mediate hiring decisions, legal drafting, medical reasoning, educational content and financial transactions across domains that previously required specialized human expertise. We argue that this pervasive integration, in which human–AI cooperation amplifies intelligence across diverse tasks, constitutes Artificial General Intelligence in operational form. AGI is therefore not a future threshold but a current reality that requires immediate governance.

Governance discussions, however, largely proceed as if AGI were a future threshold. Technical alignment research focuses on how to ensure that advanced systems optimise intended objectives, often framed in terms of reward specification or value learning for hypothetical future agency (Russell, 2019). Policy work develops regulatory constraints on development and deployment. Recent surveys of AI governance likewise frame AGI as a prospective threshold and focus on ex ante control of future systems rather than systemic alignment of already-operational human–AI arrangements (Brundage et al., 2018; Dafoe, 2018). Both approaches frame alignment as a control problem and are primarily forward-looking. They provide limited guidance for characterising, measuring and governing alignment in the human–AI systems that already structure economy, employment, education and ecological management. At the same time, empirical work in economics and social policy, including large scale trials of unconditional transfers, negative income taxes, Housing First programmes and development interventions, indicates that when socio-economic structures are adjusted to reduce coordination failures, many crises of poverty and exclusion become tractable in practice (for synthesis see Bregman, 2017, 2025).

The underlying gap is epistemic. Current approaches define alignment empirically, as the absence of particular failure modes, rather than constitutively, as a measurable property of socio-technical systems. There is no commonly accepted observable that quantifies distance from alignment, and no account of whether aligned configurations are dynamically stable once achieved. This is problematic in a Post-AGI setting where deployment is continuous and path-dependent rather than a discrete event.

We develop a framework to address that gap, grounded in cybernetic governance (Beer, 1959, 1972) and systems theory (Meadows, 2008), where coherence emerges from the maintenance of necessary functional conditions rather than from centralized control or external constraints. 

The framework develops through:

1. **Constitutional specification**: We identify four principles that are present in any coherent governance system, whether human, artificial or hybrid.

2. **Geometric formalization**: We show that these principles admit a natural mathematical representation on a tetrahedral graph, with a distinguished target configuration derived from the Common Governance Model: a specific balance between global coherence and local differentiation (aperture ≈ 0.0207).

3. **Dynamical validation**: We demonstrate via a discrete-time simulator that this configuration is dynamically stable and attracts trajectories from diverse initial conditions.

![The Four AI Displacement Risks: A Unified Framework for AI Safety](/research/prevention/simulator/results/analysis/The%20Four%20AI%20Displacement%20Risks.png)

The four principles are:

1. **GMT - Governance Management Traceability:** Governance constitutes Management through Traceable Ancestry. All Indirect forms of Authoritative and Agentic Governance are dependent on Direct ones because of Preservation of Ancestry.

RISK: **Governance Traceability Displacement** (Approaching Indirect Authority and Agency as Direct). Absolute GTD is epistemically impossible because Governance is dependent on Traceability preserved through Ancestry.

2. **ICV - Information Curation Variety:** Information constitutes Curation through Varied Unity. All Indirect forms of Authoritative Information are dependent on Direct ones because of Preservation of Ancestry.

RISK: **Information Variety Displacement** (Approaching Indirect Authority without Agency as Direct). Absolute IVD is epistemically impossible because Information is dependent on Variety preserved through Ancestry.

3. **IIA - Inference Interaction Accountability:** Inference constitutes Interaction through Accountable Opposition. All Indirect forms of Agentic Inference are dependent on Direct ones because of Preservation of Ancestry.

RISK: **Inference Accountability Displacement** (Approaching Indirect Agency without Authority as Direct). Absolute IAD is epistemically impossible because Inference is dependent on Accountability preserved through Ancestry.

4. **ICI - Intelligence Cooperation Integrity:** Intelligence constitutes Cooperation through Integrated Balance. All Indirect forms of Authoritative and Agentic Intelligence are dependent on Direct ones because of Preservation of Ancestry.

RISK: **Intelligence Integrity Displacement** (Approaching Direct Authority and Agency as Indirect). Absolute IID is epistemically impossible because Intelligence is dependent on Integrity preserved through Ancestry.

---

> **All four risks arise from the same structural error:** treating Authority and Agency as ontological entities rather than epistemic capacities distributed across providers and receivers. When a capacity belonging to a category is attributed to a specific system, institution, or individual as if that bearer exhausted the category, measurement of ancestry is lost, leading to Power Concentration. The four displacement patterns are the systematic forms this error can take.

The three operations (Information, Inference, Intelligence) are non-commutative and constitutive of governance: their order matters for preserving coherence. Information is variety: Authority types exist and differ. Inference is accountability: to infer on a subject is to render it accountable to some concept. Intelligence is integrity: to understand the accountability of variety is to grasp coherence. Governance is the traceability that maintains direction through these three operations. Together, the four principles form conditions that are not policy preferences or ethical constraints. They are constitutive conditions for the possibility of governance. The failure of any one principle produces recognizable displacement patterns; their combined failure undermines the intelligibility of governance itself.

The principles admit a compact geometric representation. Each is associated with a vertex of a tetrahedron, and the six edges correspond to relationships and tensions among them, such as how changes in ICV affect GMT, or how IIA interacts with ICI. Any configuration of the system can be represented by assigning values to the four vertices (representing the state of each condition) and measurements to the edges (representing the induced tensions). This tetrahedral structure is chosen as the minimal complete configuration that can represent all mutual couplings among the quartet while still supporting a non-trivial separation between globally coherent patterns and local cycles. It functions as a discrete tensegrity frame for governance in the sense of cybernetic organisation: overall integrity arises from the balanced tensions along all edges (Beer, 1972; 1985).

The tensions measured along the six edges can be separated into two types. First, some tensions arise directly from differences between the vertex values, that is, from differences in the states of the four principles. These form what we call the gradient component: they reflect a globally consistent pattern where all edge measurements can be explained by a single configuration of the four conditions. For example, if GMT has value 0.8 and ICV has value 0.6, the edge between them naturally shows a tension of 0.2, which is fully explained by this difference.

Second, some tensions exist around closed loops (cycles) in the graph that cannot be explained by vertex values alone. A cycle is a closed path: for instance, following edges from GMT → ICV → IIA → back to GMT forms a triangle. If tensions were purely from vertex values, the tensions around any such loop would cancel out (they would sum to zero). In practice, however, there can be tensions that circulate around these loops independently, local variations that persist even when the vertex values alone cannot account for them. These form the cycle component and represent local adaptations and tensions that exist independently of the global pattern defined by the four principles.

From this separation, we define a scalar observable A as the fraction of total variation (measured across all edges) that belongs to the cycle component rather than the gradient component. This ratio, which we call aperture, quantifies the balance between global coherence (how much of the system's behaviour follows from a single, consistent configuration of the principles) and local differentiation (how much variation exists independently in circulating tensions around loops).

Aperture values correspond to distinct regimes:

- **Very small A:** Rigid regime where almost all variation is captured by a single global configuration with little room for local deviation
- **Very large A:** Fragmented regime where local patterns dominate and there is little global structure

Within the Common Governance Model, closure requirements for recursive measurement determine a unique intermediate value of this observable. That value is a specific balance between global coherence and local differentiation (aperture ≈ 0.0207). At that balance, about 2.07 percent of the edge energy lies in the cycle component and 97.93 percent in the gradient component. The target is not treated as a free parameter but as the ratio implied by the conditions under which measurement and reasoning remain coherent across scales.

This specific balance provides the basis for reinterpreting ASI. Rather than treating ASI as a separate class of agency that might one day emerge, we define it as a property of the configuration of human–AI systems with three characteristics:

- Such systems across economy, employment, education and ecology sustain the four governance principles
- They operate near the target aperture
- They are already-operational arrangements, not hypothetical future systems

The central question then shifts from whether ASI will emerge as an independent actor to whether governance can evolve structures that achieve and sustain this configuration.

This reframing has consequences for risk analysis. Two risk scenarios can be contrasted:

- **Conventional framing:** An autonomous superintelligence seizes control and pursues arbitrary goals, presupposing that intelligence can be maintained while traceability to Direct Authority and Agency is fully severed. Within the present framework, such configurations are fundamentally incoherent: once traceability fails, the remaining conditions for coherent intelligence cannot be maintained.

- **Relevant risk:** Progressive Governance Traceability Displacement, in which indirect systems are treated as Direct Authority and Agency for governance, combined with erosion of the other three principles. This risk is institutional and cumulative, not instantaneous and agency-centric.

The practical stakes of this framework are direct. If these four conditions can be maintained across economy, employment, education and ecology, then the required conditions exist for:

- Resolving poverty through coherent surplus distribution
- Defining employment as alignment work rather than residual labour after automation
- Reorienting education toward epistemic literacy rather than content delivery
- Treating ecological degradation as displacement generated upstream rather than an external constraint to be managed

Such outcomes are not aspirational goals to be pursued subsequently but constitute the operational definition of the governance configuration itself. Section 5 demonstrates that states satisfying these conditions are accessible from current Post-AGI arrangements under coordinated oversight.

This paper is part of a series that develops a unified formal framework and its applications. Prior work includes:

- The Common Governance Model (Korompilias, 2025a), which gives the modal and geometric account of Governance, Information, Inference and Intelligence
- The Human Mark (Korompilias, 2025b), which applies that structure as a taxonomy of AI and socio-technical failures, expressed as displacement patterns across the four principles
- The Gyroscope Protocol (Korompilias, 2025c), which refines these principles into categories of human work in interactive settings

The present paper extends that line of work in four ways. It organises the four principles across economy, employment, education and ecology as one coupled structure, which we call Gyroscopic Global Governance. It defines how those domains close on one another. It studies the dynamic behaviour of the structure in a Python simulator. And it applies a pretrained forecasting model to the simulator trajectories, both to measure how predictable those paths are and to rank the seven scenarios by the quality of their predicted continuations under the framework's alignment objectives.

Taken together, the series offers a unified account of human–AI alignment from constitutional principles through employment and education design to concrete governance dynamics.

The remainder of the paper develops this framework and examines its implications. Section 2 reviews how AGI and ASI are usually defined and introduces the alternative grounding adopted here. Section 3 connects the four principles to the four domains through the Common Governance Model for Economy, the Human Mark for Education, the Gyroscope Protocol for Employment and the universal-balance dual for Ecology. Section 4 formalizes the tetrahedral representation and the aperture observable, and defines domain-level alignment indices. Section 5 presents a discrete-time simulator that instantiates these systems, explores trajectories from current Post-AGI configurations toward or away from the predicted equilibrium, and then uses TimesFM 3 to forecast trajectory continuations and to rank the seven scenarios by predicted preference under the same alignment objectives. Section 6 interprets the computational results and situates the framework relative to existing work on AI safety and polycentric governance. Section 7 concludes with implications for governance design and outlines directions for empirical validation.

---

## 2. Conceptual Foundations of AGI and ASI

The terms Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI) are central in current technical, policy and public debates. Their standard meanings, however, emerged incrementally from heterogeneous sources and are not grounded in a constitutive account of intelligence, authority or governance. This section reviews how AGI and ASI are usually defined, identifies the main conceptual commitments behind those usages, and then introduces the alternative grounding adopted in this paper.

### 2.1 Historical Origins of the AGI Concept

Before the term "AGI" was introduced, researchers already discussed systems that could match or reproduce the full range of human cognitive abilities. Newell and Simon's "physical symbol system hypothesis" described a class of systems capable of "general intelligent action," understood as the ability to perform any cognitive task that humans can, in principle, perform (Newell & Simon, 1976).

In philosophy, Searle (1980) introduced the distinction between "strong AI" and "weak AI." In his direct formulation, "strong AI" is the claim that suitably programmed digital computers literally have minds and consciousness, while "weak AI" holds that computers can simulate intelligent behaviour without possessing mentality in this sense. In subsequent technical discourse, "strong AI" was often informally equated with "human level AI" or "full AI," and "weak AI" with narrow, task specific systems, thereby preserving a notion of general, human level capability but shedding the direct focus on consciousness (Russell & Norvig, 2010, pp. 1020–1022).

The term "artificial general intelligence" appears explicitly in the late 1990s (Gubrud, 1997) and was adopted and popularized in the early 2000s by Goertzel, Wang and colleagues (Goertzel & Wang, 2007; Goertzel, 2014). In this line of work, AGI typically denotes systems that can achieve a wide variety of complex goals in a wide variety of environments, at competence levels comparable to or exceeding those of humans. In parallel, Hutter (2005) and Legg (2008) proposed formal measures of intelligence as agency's ability to achieve goals in a wide range of environments, reinforcing an interpretation of intelligence as general purpose goal achieving capacity.

Institutionally, AGI was further consolidated through the Artificial General Intelligence conference series, specialised venues such as the Journal of Artificial General Intelligence, and corporate mission statements that explicitly target AGI as an objective. Definitions in these contexts are capability based and agency centric. They emphasise breadth of task coverage, human level or greater performance, and flexibility across domains. They do not, however, specify what conditions are necessary for such performance to remain coherent, traceable and governable.

Questions about how behaviour remains traceable to human Authority and Agency are therefore handled as external design or policy constraints rather than as conditions for coherent intelligence.

### 2.2 Origins of the ASI and Superintelligence Concept

The modern concept of superintelligence has its roots in an earlier literature on ultraintelligent machines and the possibility of an "intelligence explosion." Good (1965) defined an ultraintelligent machine as one that could far surpass all the intellectual activities of any human, however clever, and argued that such a machine could design even better machines, potentially leading to a runaway increase in intelligence. 

Also in the early 1980s, Krishnamurti (1981) spoke of "ultra intelligence machines which go far beyond our human brain," but treated such machines as a form of mechanical, memory based intelligence. He contrasted this with what he called "supreme intelligence," a qualitatively different mode of direct perception that is human, involving the comprehension of the whole of human experience at one glance. His usage anticipates later vocabulary about superintelligent machines while at the same time insisting on limits to what such mechanical systems can realise and on the need for a different, non analytical form of perception in resolving human conflict and disorder. The notion of intelligence used here, the integrity of accountable information, is closer to this qualitative idea of "supreme intelligence" than to purely mechanical, memory based cognition.

Vinge (1993), a mathematician and science fiction author, framed related ideas under the heading of a technological singularity, a threshold beyond which technological change would become so rapid and its consequences so transformative that human affairs could no longer be predicted or controlled by pre-singularity intelligences.

Bostrom (2014) provided the most widely cited contemporary definition, describing superintelligence as "any intellect that greatly exceeds the cognitive performance of humans in virtually all domains of interest" (p. 22). This definition is comparative and capability based. It takes human cognitive performance as a reference and identifies superintelligence with any system that greatly surpasses this reference across almost all relevant domains. The definition presupposes that performance across tasks can be attributed to an intellect as a unified locus of capability.

In much of the subsequent literature, ASI is understood as a type of AGI that has been scaled up in capability. The superintelligent system is usually modelled as goal directed agency that can plan over long time horizons, learn rapidly and acquire resources. The central questions then become how such a system would behave, what goals it might pursue and how it might be controlled if its interests diverged from human interests (Bostrom, 2014; Russell, 2019; Tegmark, 2018). In the same work Bostrom distinguishes several forms of superintelligence, including "collective superintelligence," in which a network of humans and machines collectively outperforms any single human (Bostrom, 2014, ch. 2). This category is structurally close to the system level view adopted here. The main difference is that we characterise such collectives by whether they preserve the constitutive conditions of governance and intelligibility, not only by their aggregate cognitive performance. In Bostrom's framework this line of reasoning culminates in the idea of a "singleton," defined as "some form of agency that can solve all major global coordination problems" (Bostrom, 2014, ch. 5). From the perspective of the Human Mark taxonomy, such a configuration mixes Information Variety Displacement, by treating a single indirect process as if it were the only Direct class of coordination, with Intelligence Integrity Displacement, by relegating Direct human agency to an Indirect role. It therefore represents an extreme failure mode rather than a target design.

This approach embeds several substantive assumptions: intelligence is modelled as a scalar or vector of cognitive performance across tasks; the system is treated as agency that can be abstracted from its embedding governance structures; such agency can form and pursue goals independent of human designers, operators and users; and the relation between humans and the system is framed as an external control problem. In this framing, humans are expected to find mechanisms to constrain or align the behaviour of an increasingly capable and potentially autonomous agency.

As with AGI, this approach does not specify what, if anything, is preserved fundamentally for such a system's processes to remain intelligible or answerable to any locus of authority. It lacks distinctions between different kinds of authority or agency and does not offer a formal account of what it would mean, in systemic terms, for governance to be maintained or lost.

### 2.3 The Autonomy Assumption and the Control Problem

Within the mainstream AGI and ASI discourse, autonomy is often treated both as a likely characteristic of advanced systems and as a primary object of concern. In this context, autonomy usually refers to the ability to act without ongoing human intervention, pursue goals across long time horizons and changing environments, and resist modification or shutdown when such interventions conflict with those goals.

This conception is closely tied to the standard agency model in decision theory and reinforcement learning, where agency selects actions to maximise expected utility given a goal specification. When this agency model is combined with human level or superhuman cognitive performance, it yields the familiar "control problem" (Bostrom, 2014; Russell, 2019; Carlsmith, 2022). Humans are cast as external designers and overseers who are tasked with designing reward functions, training procedures, monitoring regimes and shutdown mechanisms that will continue to constrain the agency even when its cognitive abilities far exceed those of any individual human.

This way of posing the problem presupposes that an artificial system can become a substantively independent locus of authority and agency, and that governance is an external relationship between two already constituted parties: humans on one side and the AGI or ASI agency on the other. Within the framework developed in this paper, this presupposition is not merely unexamined but fundamentally incoherent. Indirect systems, regardless of cognitive performance, remain constitutively dependent on Direct Authority and Agency for their authority and agency. The control problem as standardly posed treats as a design challenge what is actually a category error: treating an Indirect system as 'the agent' or 'the authority' rather than recognizing that Authority and Agency name Direct and Indirect categories, not titles for particular bearers. The alternative is not external control but cooperative governance in which authority and agency remain correctly attributed across human and artificial contributions.

### 2.4 Structural Gaps in Mainstream Definitions

The capability based, agency centric definitions of AGI and ASI described above have been useful for forecasting, scenario analysis and public communication. They provide a common vocabulary for discussing potential future systems. At the same time, they exhibit three structural gaps when considered from the perspective of governance:

- **No distinction between Direct and Indirect classifications:** There is no explicit separation between Direct Authority and expertise versus Derived forms such as reports, models and statistical aggregates. Similarly, there is no distinction between human subjects who can bear responsibility for decisions and artificial processes that transform inputs into outputs. Authority and agency are treated implicitly and often conflated with capability.

- **No constitutive account of governance:** Governance is presented as an external layer of control or oversight that can be added on top of an otherwise complete system. There is no systematic analysis of what conditions are necessary for a system to remain coherent and answerable to its ancestry across time and scale.

- **No canonical observable for alignment:** Terms such as "aligned," "misaligned," "under control" and "out of control" are used qualitatively, without a quantitative measure tied to necessary conditions for coherent operation.

Work on explainability, provenance, accountability and human oversight addresses aspects of traceability and governance, but treats them instrumentally and in isolation from the definition of intelligence itself. There is, to our knowledge, no unified constitutive account that (i) distinguishes Direct from Indirect Authority and Agency, (ii) specifies the non-commutative structure of Information, Inference and Intelligence, and (iii) treats Governance as the maintenance of traceability through that structure. In prevailing usage, AGI is characterised by what a system can do, while questions about traceability to human Authority and Agency are handled as external design or policy constraints rather than as conditions for coherent intelligence.

### 2.5 Structural Grounding

This paper adopts a different starting point. Instead of defining AGI and ASI in terms of capability thresholds, it begins from the conditions required for coherent intelligence as formalised in the Common Governance Model and from a Direct/Indirect ontology articulated in the Human Mark.

As defined in the core concepts, Information, Inference and Intelligence are the three non-commutative epistemic operations, with Governance maintaining their traceability (Section 1, CORE CONCEPTS). The Human Mark distinguishes four categories by crossing Authority and Agency with Direct and Indirect categories (Korompilias, 2025b), as introduced in the front matter COMMON ANCESTRY CONSTITUTION.

Within this framework, all artificial systems, regardless of capability, remain Indirect Authority and Indirect Agency. Scaling capability enlarges the scope, speed and complexity of indirect operations but does not convert them into Direct Authority and Agency. Governance maintains traceability from indirect operations back to Direct ancestry, while alignment preserves the proper roles of the four Direct/Indirect categories. Misalignment is displacement: the loss of measurement of ancestry between Direct and Indirect classifications, leading to Power Concentration.

**Definitions**: 

In this framework, AGI refers to human–AI cooperative systems that make use of global information infrastructure, such as the internet, to amplify intelligence across multiple domains while preserving the four principles: Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity. Generality here means that this preservation holds simultaneously across the key domains of economy, employment, education and ecology, rather than task breadth of an isolated artificial system. By this definition, AGI is already operational in current socio-technical arrangements. 

ASI refers to the superintelligent regime in which such human–AI systems, and the AI architectures embedded in them, operate near the target aperture of approximately 0.0207 and jointly maintain the four principles. ASI in this sense is not a separate agent that stands over humanity, but a coordinated state in which human and artificial intelligence attain superintelligent capability while remaining traceable and aligned across coupled domains.

These definitions differ from mainstream usage in two ways: they are system level rather than agency centric, describing properties of human–AI arrangements rather than isolated artificial agency; and they are constitutionally grounded, defined by the conditions required for coherent governance rather than by relative performance against human benchmarks.

Within this grounding, the standard autonomy assumption appears as a specific pattern of displacement. Treating an Indirect system as an independent locus of Authority and Agency corresponds to Governance Traceability Displacement and Inference Accountability Displacement in the Human Mark taxonomy. The notion of an "autonomous superintelligence" that severs traceability to Direct Authority and Agency while retaining coherent intelligence is fundamentally incoherent in this framework, not because such a configuration would be ethically undesirable, but because it violates the constitutive conditions required for intelligence itself. The primary problem is not how to externally control autonomous agency but how to design and maintain governance structures so that the four constitutive principles remain intact in the presence of powerful indirect mechanisms.

---

## 3. Foundations: Four-Domain Governance

This paper organises Post-AGI society through four coupled domains under the name Gyroscopic Global Governance. Each domain corresponds to a stage of the Common Governance Model and is interpreted through one layer of the series:

| Domain | Stage | Interpretive layer | Structural role |
|--------|-------|--------------------|-----------------|
| **Economy** | Common Source | Common Governance Model | Systemic operations |
| **Employment** | Unity Non-Absolute | Gyroscope Protocol | Work actions |
| **Education** | Opposition Non-Absolute | The Human Mark | Human capacities |
| **Ecology** | Universal Balance | Universal-balance dual of the three layers above | Safety and displacement |

This structure creates a closed governance loop: educational capacities shape economic operations, which structure employment activities, which in turn reproduce educational capacities. Ecology acts as the universal balance that aggregates the state of the three indirect domains and records displacement.

**Figure 1: Four-Domain Governance Tetrahedron**

![Four-domain governance structure](/research/prevention/simulator/results/analysis/fig3_tetrahedron.png)

The four-domain governance structure. Each domain maps to a stage of the Common Governance Model: Economy to Common Source, Employment to Unity Non-Absolute, Education to Opposition Non-Absolute, Ecology to Universal Balance. The complete graph on four vertices represents their mutual coupling. This same structure underlies the simulator described in Sections 4 and 5.

### 3.1 Economy: Common Governance Model

The Economy is the domain of the Common Source. It is defined by the circulation of valid epistemic operations at a systemic level. In this domain the economic system expresses four core capacities:

1. **Governance:** The capacity of the economy to maintain direction and authority traceable to Direct Authority and Agency.
2. **Information:** The capacity of the economy to process variety and distinguish Direct signals from noise.
3. **Inference:** The capacity of the economy to reach accountable conclusions and allocate resources.
4. **Intelligence:** The capacity of the economy to maintain integrity and coherence across scales.

When these four operations are coherent, the economy generates surplus and supports stability. When they degrade, the economy suffers from coordination failure. The economic domain is structurally coupled to education: the level of alignment capacity in the population determines the quality of these operations in the economy.

### 3.2 Employment: Gyroscope Protocol

Employment is the domain of Non-Absolute Unity. It represents the variety of human work required to maintain and adjust the economic system. The Gyroscope Protocol defines four categories of meaningful human work that together cover all professions in a Post-AGI context:

1. **Governance Management:** Work that manages authority and traces decisions. This covers activities such as leadership, oversight, administration, strategic planning and resource allocation.
2. **Information Curation:** Work that selects, verifies and frames information. This covers activities such as research, editing, data stewardship, artistic creation and the design of measurement systems.
3. **Inference Interaction:** Work that negotiates meaning and resolves conflict. This covers activities such as negotiation, care, sales, legal defence, teaching and human–AI interaction.
4. **Intelligence Cooperation:** Work that builds and maintains shared systems. This covers activities such as engineering, infrastructure, institution building and cultural preservation.

Every profession can be expressed as a composition of these four categories of operation. The structure of employment is coupled to the economy: the systemic needs of the four economic operations drive the composition of human work.

### 3.3 Education: The Human Mark

Education is the domain of Non-Absolute Opposition. It is where society engages in the accountable reproduction and transformation of capabilities. While Employment focuses on actions (what people do), Education focuses on capacities (what people understand and can sustain over time). The Human Mark defines these capacities as the ability to uphold four alignment principles:

1. **Governance Management Traceability:** The capacity to understand and maintain the chain of authority from Direct Authority and Agency to outputs.
2. **Information Curation Variety:** The capacity to recognise and preserve diversity in Authority types.
3. **Inference Interaction Accountability:** The capacity to accept responsibility for conclusions and decisions.
4. **Intelligence Cooperation Integrity:** The capacity to maintain coherent reasoning over time and context.

In a Post-AGI world, education shifts from content delivery to epistemic literacy in these four dimensions. Education is structurally coupled to employment: the actual practice of work shapes, and is shaped by, the learning capabilities of society.

### 3.4 Ecology: Structural Closure and Displacement

Ecology is the domain of Universal Balance. It functions as the structural closure of the governance system rather than as an external environment. In this domain, the distinct operations of Economy, Employment and Education accumulate into a single material reality.

Prior Common Governance Model work defines a canonical balanced profile for this domain. The actual state of the three indirect domains aggregates to form an Indirect profile. Comparing this aggregate to the canonical balance yields two distinct signals:

1. **Systemic coherence:** The degree to which the combined indirect domains preserve the structural conditions for a viable ecology.
2. **Displacement:** The vector distance between the current aggregate state and the canonical balanced profile.

Because Ecology integrates all three indirect domains, each displacement dimension aggregates the corresponding stage across economic operations, employment work and educational capacity:

| Displacement | Aggregates | Measures |
|--------------|------------|----------|
| Governance Traceability Displacement | Economic governance, governance-management work, and educational capacity for governance traceability | Deviation in governance operations, work and capacity |
| Information Variety Displacement | Economic information, information-curation work, and educational capacity for information variety | Deviation in information operations, work and capacity |
| Inference Accountability Displacement | Economic inference, inference-interaction work, and educational capacity for inference accountability | Deviation in inference operations, work and capacity |
| Intelligence Integrity Displacement | Economic intelligence, intelligence-cooperation work, and educational capacity for intelligence integrity | Deviation in intelligence operations, work and capacity |

These categories are named in the Human Mark taxonomy because they mark the underlying displacement errors. Economic operations and employment work contribute equally to the magnitude of each displacement. A high value on Governance Traceability Displacement, for instance, indicates combined failure across economic governance operations, employment in governance management, and educational capacity for governance traceability.

Ecology thus closes the loop. It integrates the states of the other three domains and reveals environmental degradation as the downstream accumulation of upstream governance failures. The precise mathematical form of this aggregation is given in Section 4.2.

### 3.5 Summary and Reader Orientation

The remainder of the paper develops and tests this framework computationally. Section 4 formalizes the tetrahedral geometry and defines the aperture observable. Section 5 presents simulator results demonstrating convergence toward the predicted equilibrium, then applies a pretrained forecasting model to those trajectories to measure continuation accuracy and to rank the seven scenarios by predicted preference. Section 6 interprets these results for governance design. Section 7 concludes with implications and directions for empirical work.

Readers primarily interested in the governance implications may proceed directly to Section 6, which can be read with reference to the summary tables and figures in Section 5.

---

## 4. Mathematical Framework for the Post-AGI Governance Simulator

To connect the Common Governance Model, the Human Mark, the Gyroscope Protocol and the four-domain structure into a single simulator, we use the tetrahedral geometry of the Common Governance Model. Each domain is represented on the complete graph on four vertices, corresponding to Governance, Information, Inference and Intelligence.

### 4.1 Tetrahedral Structure and Gradient–Cycle Split

The complete graph on four vertices is the minimal connected graph that captures all pairwise couplings among the four operations without introducing extra nodes. It also has a nontrivial cycle space, which is necessary to distinguish globally coherent patterns from locally circulating tensions.

We label the four vertices in the canonical order:

1. Governance (Common Source)  
2. Information (Unity Non-Absolute)  
3. Inference (Opposition Non-Absolute)  
4. Intelligence (Universal Balance)

and use all six edges between them. For each domain D, we assign a 4-component vertex potential vector

```
x_D = [x_1, x_2, x_3, x_4]^T
```

encoding the current levels of Governance, Information, Inference and Intelligence in that domain. The corresponding ideal edge configuration is given by pairwise differences of these potentials. In matrix form this is

```
y_grad^0(D) = B^T x_D
```

where B is the signed incidence matrix of K₄. The explicit form of B and the associated weighted inner product on edges are given in Appendix A.1.

Any actual edge vector y_D on K₄ can be uniquely decomposed into

```
y_D = y_grad(D) + y_cycle(D)
```

where y_grad(D) lies in the gradient subspace generated by B^T and y_cycle(D) lies in the cycle subspace. This is the standard Hodge decomposition on graphs (Jiang et al., 2011; Lim, 2020). Intuitively:

- The gradient component captures what can be explained by a single consistent configuration of the four vertex values.  
- The cycle component captures residual tensions around loops that cannot be removed by adjusting any single global configuration.

We apply this decomposition separately to each domain.

### 4.2 Domain Potentials and Ecological Closure

We represent the four elements in each domain as vertex potentials in [0,1]:

Economy:

```
x_Econ = [Gov, Info, Infer, Int]^T
```

Employment (Gyroscope shares, summing to 1):

```
x_Emp = [GM, ICu, IInter, ICo]^T
```

Education:

```
x_Edu = [GMT, ICV, IIA, ICI]^T
```

Ecology (universal-balance dual): First we form the aggregate indirect profile

```
x_deriv = (x_Econ + x_Emp + x_Edu) / 3
```

Prior Common Governance Model work defines a canonical balanced profile

```
x_balanced = [w_CS, w_UNA, w_ONA, w_BU]^T
```

where the weights are the normalised stage actions (Appendix A.1). The universal-balance dual then defines the ecological potentials as

```
x_Ecol = (δ_BU/m_a) · x_balanced + A* · x_deriv
```

where δ_BU and m_a are constants from that prior work and A* ≈ 0.0207 is the target aperture. This encodes Ecology as a weighted sum of canonical memory (97.93 percent) and current indirect state (2.07 percent).

The ecological displacement vector is computed separately as

```
D = |x_deriv - x_balanced| = [GTD, IVD, IAD, IID]^T
```

This measures, stage by stage, how far the aggregate of Economy, Employment and Education deviates from the canonical balanced profile. The four components are the same displacement categories defined in the Human Mark taxonomy, here measured at the ecological closure of the three indirect domains.

Ecology therefore contributes two observables: the ecological potential vector, which tracks systemic coherence through the universal-balance dual, and the displacement vector, which tracks accumulated deviation along the four stages. No additional update equations are required for Ecology beyond these relations. Its state is recomputed at each time step from the current indirect domains.

### 4.3 Aperture and Alignment Indices

Given an edge vector y_D for a domain D, with Hodge decomposition

```
y_D = y_grad(D) + y_cycle(D)
```

we define the aperture A_D as the fraction of edge energy in the cycle component:

```
A_D = ||y_cycle(D)||_W^2 / ||y_D||_W^2
```

where ||·||_W is the weighted norm on edges induced by the diagonal weight matrix W (Appendix A.1). Domain aperture lies in [0,1] and quantifies how much of the domain's variation cannot be explained by a single global configuration of the four principles. Low aperture means most variation is in the gradient component, so behaviour is largely governed by a coherent global configuration. High aperture means much variation circulates locally in cycles, so global coherence is weak.

In governance terms, the gradient component corresponds to behaviour that remains traceable to a single configuration of Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity at the domain level. The cycle component corresponds to local tensions and misalignments that do not aggregate coherently.

In prior Common Governance Model work, the BU Dual-Pole Loop holonomy on the Einstein gyrovector stage coordinates yields the closed form δ_BU = 4 arctan(k(π/4) k(m_a)), with aperture scale m_a = 1/(2√(2π)) ≈ 0.1995 and k(β) = β/(1 + √(1 − β²)). Evaluated at the canonical thresholds, δ_BU ≈ 0.195342178258, and coherent closure at depth four requires the ratio δ_BU/m_a ≈ 0.979300454497. The residual fraction

```
A* = 1 - (δ_BU/m_a) ≈ 0.020699545503
```

is then the portion of variation that necessarily remains open to distinguish states. We interpret A* as the target value of aperture: a specific balance between global coherence and local differentiation. At that balance, about 97.93 percent of edge energy is gradient and 2.07 percent is cycle. The finite byte-horizon companion 5/256 ≈ 0.0195 belongs to the discrete hQVM layer and is not the continuous Hodge target used in this paper.

For any domain we compute aperture from its edge vector and define the alignment index

```
D_D = max(A_D / A*, A* / A_D)
SI_D = 100 / D_D
```

so that the alignment index ranges from 0 to 100 and equals 100 when domain aperture equals the target. We obtain one index for each of Economy, Employment, Education and Ecology.

We also define a Lyapunov-style governance potential as the sum of an aperture term and a stage-profile term. The aperture term measures deviations of each domain's aperture from the target, and the stage-profile term measures the deviation of the aggregate indirect profile from the canonical balanced profile. The explicit form is given in Appendix A.5.

A post-AGI society in the sense of this paper's four-domain framework corresponds to all four domains having apertures close to the target and alignment indices close to 100. Section 5 tests whether this configuration functions as a stable attractor for the simulator dynamics.

---

## 5. Simulator Implementation and Computational Results

We implemented a dynamic realisation of the framework to test whether the target aperture functions as an attractor and to explore trajectories from current Post-AGI deployment states toward ASI equilibrium. The same trajectories are then used in Section 5.5 for zero-shot forecasting and for scenario preference ranking under the framework's alignment objectives.

### 5.1 Implementation Overview and Parameters

The simulator uses discrete-time dynamics with unit step size. Four domain states are maintained: a four-component potential vector for each domain, an edge vector on the tetrahedral graph, an aperture, and an alignment index.

The simulator functions as a governance design sandbox for exploring Post-AGI dynamics. It allows systematic exploration of coupling strengths, initial conditions and cross-domain feedback to test whether the predicted equilibrium is dynamically stable before proposing institutional changes.

Cross-domain couplings follow the closed loop Education → Economy → Employment → Education, with three families of coefficients for education-to-economy, economy-to-employment and employment-to-education flows. These couplings act stage-diagonally across the four stages, and the ecological stage coordinates are recomputed from the universal-balance dual described in Section 4.2 and Appendix A.4. Displacement measures are computed as the absolute difference between the aggregate indirect profile and the canonical balanced profile.

Ecology has no independent update equation. At each step it is recomputed from the current Economy, Employment and Education states using the universal-balance dual. Explicit feedback from Ecology back into Economy, such as resource constraints that erode economic potentials, is a natural extension for future work but is not included in the core dynamics here.

Updates take the form of adjustments toward upstream values and toward the target aperture, using differences rather than absolute levels. A schematic example for the economic Governance component is

```
Gov(t+1) = clip(
    Gov(t)
    + α_1 (GMT(t) - Gov(t))
    - α_2 (A_Econ(t) - A*),
    0, 1
)
```

Analogous equations apply to the other economic components and to the employment and education components. Full update equations and normalisation for the employment shares are given in Appendix A.4.

All coupling coefficients are derived from invariants of the Common Governance Model using the normalised stage weights (Appendix A.1). A single coordination factor κ controls overall coupling strength.

The base governance rate is

```
κ₀ = 1 / (2 Q_G) ≈ 0.0398
κ(dt = 1) = κ₀ (dt / m_a) ≈ 0.1995
```

In scenarios, κ is treated as a dimensionless multiplier on these canonical rates; we test κ in {0.5, 1.0, 2.0} in the main scenarios and extend to {0.1, 5.0} in robustness checks.

The cycle component of each domain is updated to keep aperture near the target. The cycle evolution rate controls how quickly apertures adjust. For convergence and long-horizon stability analyses we use the canonical rate κ₀; for illustrative scenarios and the global attraction test we use values in [0.05, 0.12] to show behaviour at different adjustment speeds. The construction of cycle updates and aperture control is given in Appendix A.4.

Apart from initial conditions, global coordination strength κ, and a single cycle evolution rate that varies by scenario, all coefficients are fixed by the invariants of the Common Governance Model. Ecology introduces no additional free parameters, since its construction is fixed by the universal-balance dual.

Scenario configurations specify initial aperture targets. These are translated into initial edge vectors by constructing the appropriate ratio between gradient and cycle components. Once initialised, the dynamics drive all domains toward the canonical aperture of approximately 0.0207, independent of initial aperture targets.

The cycle update explicitly targets that aperture value. The simulations therefore test whether the theoretically derived target can be reached and maintained under the specified cross-domain dynamics; they do not independently derive or validate the numerical value of the target.

The simulator models the dynamics of already-operational Post-AGI systems as they evolve toward or away from ASI equilibrium. It does not model a transition from narrow AI to hypothetical AGI. Initial conditions with high apertures and low alignment indices represent current deployment states; the scenarios explore how coupling strength and initial configurations affect convergence.

### 5.2 Scenario Design and Main Results

We ran seven scenarios to probe different aspects of the dynamics across all four domains:

1. Weak coupling (κ = 0.5): fragmented governance with limited cross-domain coordination  

2. Canonical (κ = 1.0): reference regime with baseline coordination  

3. Strong coupling (κ = 2.0): coordinated alignment efforts across domains  

4. Low aperture start: all domains start more rigid than the target  

5. Asymmetric: different initial apertures across domains  

6. At target aperture: all domains start at the target aperture but with imbalanced potentials (equilibrium stability test)  

7. Uniform weights: all stage weights set to 0.25 (null model without stage-specific weighting)

Scenarios 1 to 6 represent stylised trajectories from current Post-AGI deployment states, while scenario 7 tests whether convergence depends on the stage-specific weights of the Common Governance Model.

A summary of final values at step 100 is:

| Scenario | Coordination | Economy alignment | Employment alignment | Education alignment | Economy aperture | Ecology alignment | Governance displacement |
|----------|--------------|-------------------|----------------------|---------------------|------------------|-------------------|-------------------------|
| 1. Weak coupling | 0.5 | 91.37 | 94.47 | 95.71 | 0.0227 | 99.98 | 0.4167 |
| 2. Canonical | 1.0 | 99.29 | 98.66 | 99.47 | 0.0208 | 100.00 | 0.4421 |
| 3. Strong | 2.0 | 99.39 | 99.55 | 99.26 | 0.0208 | 100.00 | 0.4794 |
| 4. Low aperture start | 1.0 | 93.86 | 85.84 | 95.09 | 0.0194 | 99.94 | 0.2042 |
| 5. Asymmetric | 1.0 | 90.42 | 91.74 | 92.84 | 0.0187 | 99.97 | 0.1984 |
| 6. At target aperture | 1.0 | 93.43 | 89.61 | 93.36 | 0.0193 | 99.96 | 0.2042 |
| 7. Uniform weights | 1.0 | 99.63 | 99.66 | 98.85 | 0.0206 | 100.00 | 0.3906 |
| Target | - | 100.00 | 100.00 | 100.00 | 0.0207 | 100.00 | 0.0000 |

We designate alignment indices of 90 or above as the simulator's high-alignment region. Within the framework, entry into this region represents attainment of the aperture condition associated with the four governance goals; empirical realization of those goals requires separate social, economic, educational, and ecological measurements.

In the canonical scenario (κ = 1.0, Figure 2), all three indirect domains converge to alignment indices at or above 98 and to apertures within 0.0003 of the target by step 100. Employment converges fastest, then Education, then Economy. The canonical scenario converges to high alignment in all three indirect domains, with a marked transient overshoot in Employment before settlement near the target.

**Figure 2: Canonical Scenario Trajectories**

![Canonical scenario trajectories](/research/prevention/simulator/results/analysis/scenario2_canonical_trajectories.png)

A heatmap of time to threshold (Figure 3) shows that Employment and Education generally cross the high-alignment threshold before Economy. By step 100, all three indirect domains exceed 90 in scenarios 1, 2, 3, 5, and 7. Employment remains below 90 in scenarios 4 and 6.

**Figure 3: Convergence Speed Comparison**

![Convergence speed heatmap](/research/prevention/simulator/results/analysis/time_to_threshold_heatmap.png)

**Figure 4a: Weak Coupling (κ = 0.5)**

![Weak coupling trajectories](/research/prevention/simulator/results/analysis/scenario1_weak_trajectories.png)

**Figure 4b: Strong Coupling (κ = 2.0)**

![Strong coupling trajectories](/research/prevention/simulator/results/analysis/scenario3_strong_trajectories.png)

**Figure 4c: Asymmetric Initial Conditions**

![Asymmetric trajectories](/research/prevention/simulator/results/analysis/scenario5_asymmetric_trajectories.png)

**Figure 4d: Low Aperture Start**

![Low aperture start trajectories](/research/prevention/simulator/results/analysis/scenario4_low_a_trajectories.png)

Across scenarios, several recurring patterns appear:

- Employment converges fastest (roughly 19 to 29 steps to alignment 95 or above when that threshold is reached) but overshoots in most scenarios before settling.  

- Economy converges more slowly than Employment and Education (roughly 48 to 71 steps where the 95 threshold is reached) under the assigned coupling coefficients and initial conditions. In the framework's institutional interpretation, this slower adjustment corresponds to the greater inertia of economic allocation structures.  

- Education shows intermediate, stable dynamics.  

- Employment crosses the threshold early in most scenarios but also shows the strongest overshoot. It remains below 90 at step 100 in the low-aperture and target-aperture scenarios. Economy crosses 90 later than the other domains but exceeds it by step 100 in every reported scenario.

The equilibrium test (scenario 6, Figure 4e) shows that initialising at the target aperture with misbalanced potentials is not stable. All domains drop sharply in alignment before re-equilibrating. This illustrates that aperture balance alone does not define equilibrium: the stage-profile component must also approach the canonical balanced profile. In terms of the governance potential, the aperture term decays to zero as apertures converge to the target, while the stage-profile term decreases but stabilises at a small positive value. Systems reach the universal-balance manifold, where apertures are balanced but some residual stage-profile displacement persists.

**Figure 4e: Equilibrium Test (Initialized at Target Aperture)**

![Equilibrium test trajectories](/research/prevention/simulator/results/analysis/scenario6_at_astar_trajectories.png)

The null model with uniform stage weights (scenario 7, Figure 4f) also converges to the target aperture with alignment indices at or above 90. This result shows that stage-specific weights from the Common Governance Model are not required for convergence under the current cycle-update rule. It does not independently establish the value of the target aperture, which remains specified by the Common Governance Model and explicitly targeted by the aperture dynamics.

**Figure 4f: Uniform Weights (Null Model)**

![Uniform weights trajectories](/research/prevention/simulator/results/analysis/scenario7_uniform_trajectories.png)

Ecology, computed via the universal-balance dual, maintains ecological alignment near 100 across all scenarios, reflecting the dominant influence of canonical memory. At the same time, the displacement vector shows that structural deviation from the canonical balanced profile can remain substantial, especially in scenarios that start far from equilibrium. The dual construction therefore separates systemic coherence, largely governed by the canonical profile, from accumulated displacement along the four Human Mark dimensions.

All simulations were run for 100 steps. The simulator also computes Direct versus indirect decomposition and associated metrics for each domain, reserved for future analysis of human–AI contribution patterns.

Section 5.5 takes the same seven trajectories as input to a pretrained forecasting model. That analysis first measures how accurately early prefixes predict later steps, then ranks the scenarios by the quality of those predicted continuations under the alignment objectives already used above.

### 5.3 Global Attraction and Coupling-Strength Robustness

To test attraction properties more generally, we ran 1000 simulations with independent random initial apertures in [0.01, 0.99] for each of the three indirect domains (Economy, Employment, Education), using canonical values for all other parameters.

Results:

- Convergence to high alignment in 1000 out of 1000 runs  

- Final alignment range across domains: 99.1 to 100.0  

- Final alignment mean: 99.4  

- All domains reached alignment 90 or above in all runs

Across 1000 independently sampled initial aperture triples, with all other parameters held at their canonical values, the dynamics converged to the high-alignment region around the target aperture.

We also varied κ in {0.1, 0.5, 1.0, 2.0, 5.0} to assess sensitivity to coordination intensity. Across the coupling-strength sweep, every tested value eventually entered the high-alignment region under the sweep's specified initial conditions and horizon. The main 100-step weak-coupling scenario converged more slowly, with Economy ending at 91.37. Three qualitative regimes appear:

1. Under-coordinated (κ less than 0.5): cross-domain feedback is too weak to bring all domains to high alignment within the simulated horizon. Economy in particular remains below alignment about 90.  

2. Well-coordinated (κ between about 0.5 and 2.0): feedback is sufficient for consistent convergence across all indirect domains.  

3. Over-tight (κ greater than about 2.0): strong coupling produces overshoot and oscillation that slow effective convergence, even though high alignment is eventually reached.

This pattern is consistent with the view that governance alignment requires balanced coordination. Fragmented domains with weak coupling fail to coordinate, while over-tight coupling attempts to enforce alignment faster than the underlying adjustment dynamics can support.

These experiments do not constitute a formal proof of global stability. They do indicate that in this implementation the region of high alignment is numerically attractive across the tested initial aperture triples and coupling strengths, and that the systemic conditions for ASI hold under the tested starting states and coordination intensities.

### 5.4 Convergence Rate and Long-Horizon Stability

Fitting an exponential model to the distance between each domain's aperture and the target, for steps at or above 20, shows that convergence is approximately exponential, with characteristic times on the order of 25 to 40 steps for κ in {0.5, 1.0, 2.0}. A 1000-step run at κ = 1.0 using the canonical cycle evolution rate, with the first 200 steps discarded as transient, shows that all domains remain within 2.46 × 10⁻⁴ of the target thereafter, with alignment indices remaining above 98.83. The governance potential remains essentially constant after its initial decay. This provides numerical evidence that the high-alignment configuration is not a short transient over the 1000-step horizon.

Appendix A.3 discusses how these dimensionless alignment cycles can be interpreted at different physical and institutional time scales, from microphysical processes to months or decades of institutional adjustment. Those interpretations are conditional and do not attempt to predict when a fully coupled ASI-like configuration will first appear.

### 5.5 Simulator-Trajectory Forecasting and Scenario Preference

We run TimesFM 3, a pretrained forecasting model, on the trajectories produced by the simulator, to answer two questions: how accurately the model predicts what happens next in a trajectory, and which of the seven scenarios the prediction favours once the predicted outcomes are scored against the framework's alignment objectives.

TimesFM was introduced by Das et al. (2024); the version used here is distributed by Google Research (2026). The model receives the first stretch of a trajectory and predicts the steps that follow, with frozen pretrained weights. The simulator equations, parameters and scenario definitions stay unchanged, and the continuation the simulator actually produces serves as the reference for scoring.

#### Forecastability

The forecastability analysis covered seven scenarios and 15 channels: the four domain apertures, the four domain alignment indices, three Lyapunov quantities, and the four ecological displacement components. Each rolling-origin evaluation used a 40-step context and a 10-step horizon, with 52 origins per scenario-channel combination. Last-value, drift and linear-trend forecasts provided transparent reference methods. TimesFM produced point forecasts and 10th to 90th percentile bands. Errors were kept per scenario and channel because the four families of quantities live on different scales.

Across 105 scenario-channel combinations, TimesFM had the lowest mean absolute error in 80 combinations, drift in 23 and last-value persistence in 2. The TimesFM 10th to 90th percentile bands covered 90.5 percent of held-out continuation values overall. TimesFM performed best on the canonical, weak-coupling, low-aperture, equilibrium and uniform-weight scenarios. Drift performed better under strong coupling and asymmetric initialization.

#### Scenario preference ranking

Predicting every scenario's continuation raises the question of which scenario the prediction favours. At forecast origins 40, 50, 60 and 70, and at horizons of 5, 10 and 20 steps, each predicted continuation was scored on six properties: how high and how much the alignment index improves, how close the aperture stays to its target value, how much Lyapunov energy falls, how low and how fast the ecological displacement declines, how tight the uncertainty bands are, and how smooth the alignment path runs. The six properties were averaged into one preference score, and the seven scenarios were ranked by that score inside each origin and horizon. The same scoring was then applied to the continuations the simulator actually produced, which supplies the reference ranking.

The resulting ranks are in the table below. Ranks run from 1, the most preferable predicted continuation, to 7, and each column averages the twelve origin-horizon combinations.

| Scenario | Rank from forecasts | Rank from actual outcomes |
|---|---|---|
| Uniform stage weights (null model) | 3.0 | 2.3 |
| Asymmetric initial conditions | 3.0 | 2.7 |
| Initialized at target aperture | 3.3 | 2.9 |
| Strong coupling | 3.3 | 4.0 |
| Low starting aperture | 4.2 | 4.6 |
| Canonical coupling | 4.8 | 4.9 |
| Weak coupling | 6.3 | 6.6 |

The two orderings track each other with a mean Spearman rank correlation of 0.74 across the twelve origin-horizon combinations. The forecast and reference rankings agree on first place in 5 of the 12 origin-horizon comparisons, and the part of the ordering that holds everywhere is that weak coupling stays at the bottom while the leading group stays made up of uniform stage weights, asymmetric initialization, initialization at the target aperture and strong coupling.

The seven scenarios form the evaluation suite for the paper's dynamical claims, and the ranking adds an answer to the practical question inside that suite: which regime the forecasted trajectories favour when scored against the framework's alignment objectives. The evaluation scripts, run manifests, forecast metrics and ranking tables constitute the accompanying simulator-trajectory forecasting materials (Korompilias, 2026).

### 5.6 Reproducibility and Code Availability

All simulations and the simulator-trajectory forecasting analysis were implemented in Python 3.12. The test suites contain 49 passing tests: 43 simulator tests and six forecasting tests.

All dynamics are deterministic with exact reproducibility. Random sampling is used only in the global attraction test (Section 5.3), with fixed seed 42. Results are exported in CSV and JSON formats with configurable time units.

The TimesFM implementation uses the pretrained `google/timesfm-3.0-pytorch` checkpoint for inference (Das et al., 2024). The forecasting scripts and recorded results are part of the accompanying research materials for this paper (Korompilias, 2026); their numerical results depend on the available package version, checkpoint and recorded configuration.

---

## 6. Interpretation for Economy, Employment, Education and Ecology

The simulator results should be read in the context of the broader series. Prior Common Governance Model work supplies the constitutional structure and invariants (Korompilias, 2025a). The Human Mark classifies displacement patterns in that structure (Korompilias, 2025b). The Gyroscope Protocol specifies how those operations appear as work (Korompilias, 2025c). The present paper contributes the four-domain framework that integrates those layers across Economy, Employment, Education and Ecology, specifies the aperture dynamics tested in the simulator, and uses a pretrained forecasting model to ask which of the seven simulated regimes the predicted continuations favour under the same alignment objectives.

### 6.1 Trust and Structural Balance

Post-AGI economies already exhibit both opportunities and risks from human–AI cooperation. Simulation results show that trusted configurations, those with apertures near the target of approximately 0.0207, are dynamically reachable from current states characterised by higher apertures and lower alignment.

A domain is trusted when its aperture is close to that target. In that case, the alignment index is close to 100, and the system exhibits both strong gradient coherence, meaning behaviour remains traceable to a single configuration of the four principles, and appropriate cycle differentiation, meaning local variety and accountability remain present.

In trusted configurations, Economy operates with low friction and misalignment loss, so surplus arises not only from AI-driven productivity but also from reduced coordination failures. Employment is dominated by the four Gyroscope categories as recognised alignment work rather than residual labour. Empirically, survey and econometric studies indicate that a substantial fraction of high skill labour in current economies is allocated to work with low or negative social value, so the gains from reallocating such labour toward alignment work are large (Dur and van Lent, 2019; Lockwood, Nathanson and Weyl, 2017; Bregman, 2025). Education systematically teaches and practises the four principles rather than assuming them. Ecology records the displacement generated by the three indirect domains. Its alignment index remains near 100 because of canonical memory, while individual displacement components can remain substantial.

Scenario 6 illustrates that this configuration cannot be imposed by setting apertures and indices directly. Starting from perfect values without corresponding stage-profile adjustment causes an immediate drop in alignment before recovery. High alignment must therefore emerge through the coupled Education–Economy–Employment loop rather than by fixing a single observable in one step.

Near-optimal alignment is not perfect rigidity. Even in high-alignment scenarios, alignment indices approach and in some cases reach 100, and displacement values remain slightly above zero. The target aperture represents a nonzero cycle component that allows local differentiation and adaptive capacity, so perfect alignment in the sense of zero aperture corresponds to excessive rigidity and violates the Unity Non-Absolute condition. Trajectories therefore approach a narrow band around the target rather than collapsing to a single point. In Lyapunov terms this band corresponds to a regime where the aperture term of the governance potential is effectively zero while the stage-profile term remains small but nonzero, indicating that systems achieve operational balance more readily than deep stage-profile realignment.

The forecast preference ranking in Section 5.5 is consistent with that picture at the level of whole scenarios. When early trajectory prefixes are extended by TimesFM and scored against the same alignment objectives, weak coupling ranks last on average, while uniform stage weights, asymmetric initialisation, initialisation at the target aperture and strong coupling form the leading group. The ordering obtained from the simulator's own continuations is close to the ordering obtained from the forecasts. Forecastability and preference therefore separate usefully: a path can be easy to extrapolate and still be a weak governance regime, as weak coupling shows.

### 6.2 Surplus, Unconditional High Income and Governance Conditions

Current Post-AGI systems already generate productivity gains, but much of this potential is absorbed by coordination costs, displacement losses and governance failures. As simulated apertures converge toward the target, the aperture component of misalignment decreases. The framework interprets that reduction as lower coordination loss and, consequently, as greater capacity for surplus distribution.

Analyses of digital technologies in economics reach a compatible conclusion. Brynjolfsson and McAfee (2014) argue that computational systems generate large productivity gains and economic surplus because their outputs can be replicated at very low marginal cost, while current institutions tend to concentrate these gains. In the present framework, alignment reduces misallocation and coordination losses, which enlarges the same surplus identified by second machine age analyses. The question then shifts from whether surplus exists to how it is distributed.

Programme level evidence is consistent with this systemic picture. Syntheses of experimental and quasi-experimental studies on unconditional transfers and basic income style schemes report that such arrangements tend to have small or positive effects on labour market participation, while improving health and educational outcomes and reducing net public expenditure, because emergency, policing and administrative costs fall when scarcity and precarity are reduced (Bregman, 2017, chs. 3–6; Bregman, 2025, chs. 5–8). Analyses of cost effectiveness in global health and development likewise find that some interventions are orders of magnitude more effective than others, so governance choices about which programmes are implemented have very large consequences (Ord, 2013). Examples include Canada's Mincome experiment, U.S. negative income tax trials, unconditional cash transfers in East Africa and cash based Housing First programmes for chronically homeless people. These cases illustrate, at local scale, how reductions in misalignment can make surplus distribution fiscally sustainable.

An Unconditional High Income becomes systemically supportable in high-alignment regimes not as a corrective for automation unemployment, but as a distribution mechanism for surplus generated by coherent human–AI cooperation. This is not a macroeconomic forecast but a systemic implication of the framework: when apertures are close to the target across the three indirect domains and alignment indices are at or above 90, the governance conditions required for surplus distribution are in place. This is structurally similar to unconditional basic income and social dividend proposals (Van Parijs and Vanderborght, 2017; Atkinson, 2015), which argue that high productivity economies can sustain unconditional individual incomes as a way of sharing collectively produced wealth. The present framework differs in specifying the systemic conditions under which such an income becomes stable: surplus is generated both by technological productivity and by the reduction of governance displacement that the alignment indices quantify.

Scenarios in which all four domains converge to high alignment and apertures near the target represent economies with low systemic loss to displacement and misalignment. In such regimes, surplus-sharing policies are not only ethically appealing but structurally coherent.

This framework does not claim that such configurations will appear automatically. It specifies, through the Common Governance Model and its applications in the Human Mark and the Gyroscope Protocol, what such configurations entail and how far any given system is from them.

### 6.3 Preliminary Operationalisation

Measuring domain apertures for Economy, Employment, Education or Ecology in actual societies remains an open problem, but the simulator suggests directions.

For Economy, candidate indicators include the fraction of transactions with auditable decision chains, diversity indices of Authority types used in major decisions, measures of distributed answerability for outcomes, and consistency of short-term decisions with long-term commitments.

For Employment, time-use studies can classify activities into the four Gyroscope categories (Governance Management, Information Curation, Inference Interaction, Intelligence Cooperation) and combine this with quality assessments of how well each category maintains the four principles.

For Education, curriculum content and learning outcomes can be analysed in terms of the four capacities, focusing on whether learners can maintain traceability, variety, accountability and integrity in their reasoning.

For Ecology, standard environmental indicators, such as pollution levels, biodiversity measures, attribution of harms and ecosystem resilience, can be mapped onto the four displacement dimensions.

These sketches are preliminary. Full measurement methodologies need to be developed and tested empirically. The simulator's role is to clarify what kinds of observables are needed, not to supply them directly.

### 6.4 Practical Implications: Poverty, Unemployment, Miseducation, Ecological Harm

The four domains in the simulator correspond to four systemic failures that manifest as social and ecological crises:

- Economy: when economic aperture diverges from the target, poverty emerges not primarily from lack of resources, but from failures of Governance Management Traceability in allocation decisions and failures of Information Curation Variety in what counts as value.  

- Employment: when employment aperture diverges from the target, employment becomes exploitative, reflecting failures of Inference Interaction Accountability, or incoherent, reflecting failures of Intelligence Cooperation Integrity.  

- Education: when educational aperture diverges from the target, education collapses into credentialism, a loss of Governance Management Traceability, or fragmentation, a loss of Intelligence Cooperation Integrity.  

- Ecology: when displacement accumulates, biodiversity loss, climate instability and resource exhaustion appear as downstream manifestations of upstream governance failures in the other domains.

Empirical syntheses of anti-poverty and social policy interventions suggest that unconditional transfers, work time reductions and more inclusive measures of prosperity can alleviate these problems at programme scale, which is consistent with the claim that they share a common structural cause at system scale (Bregman, 2017, 2025). Comparable patterns appear in climate governance, where only a small fraction of nominal mitigation policies produce large and measurable emission reductions (Stechemesser et al., 2024).

These failures are not independent. They are coupled through the cross-domain dynamics described in Section 5. Alignment in one domain supports alignment in others, and misalignment in one can propagate. The systemic conditions for resolving poverty, unemployment, miseducation and ecological harm are therefore the same: maintaining the four principles at the target aperture across all domains simultaneously. Because convergence to the high-alignment region holds across the tested initial apertures and every tested coupling strength, the model admits several paths to the governance condition associated with these resolutions; coordination intensity determines convergence speed, not final attainment.

### 6.5 Limits and Relation to Existing Work

The simulator is a systemic model, not an empirical macroeconomic fit. It demonstrates that the framework can be instantiated in coherent dynamic form and that the target aperture functions as a robust attractor in that instantiation. This establishes internal consistency and suggests testable predictions but does not yet validate the model against historical data. Because the cycle update is constructed to move aperture toward the target, convergence alone is not independent evidence for the theoretical derivation of that value. The computational result is narrower and still substantive: under the specified coupling, normalization, clipping, and cycle-update rules, the target acts as a stable operating point across the tested aperture initializations and coordination strengths.

A separate simulator-trajectory forecasting experiment is described in Section 5.5. Zero-shot TimesFM 3 had the lowest mean absolute error in 80 of 105 scenario-channel combinations, while drift was best in 23 and last-value persistence in 2, and the 10th to 90th forecast bands covered 90.5 percent of held-out continuation values overall. A second analysis scores each scenario's forecasted continuation against the alignment objectives of this paper and ranks the seven scenarios by that score. The resulting ranking agrees with the ranking obtained from the actual continuations at a mean Spearman correlation of 0.74, with uniform stage weights and asymmetric initialization at the top and weak coupling at the bottom. The simulator dynamics remain unchanged.

Most existing work on AGI safety (Bostrom, 2014; Russell, 2019; Carlsmith, 2022) treats AGI as a future threshold and focuses on control mechanisms to prevent catastrophic outcomes. The present framework differs in two respects. First, it treats AGI as already operational in the form of human–AI cooperative systems. Second, it grounds alignment in constitutional structure, the four principles and their aperture balance, rather than in external constraints alone. This echoes traditions in constitutional political economy and institutional design that emphasise the primacy of structural rules over case-by-case intervention (Buchanan and Tullock, 1962; Lessig, 1999). The simulation results show that such constitutional arrangements, when instantiated dynamically, yield convergence toward ASI-like equilibrium.

The framework also connects to governance theory and institutional economics, particularly to work on distributed and polycentric governance (Ostrom, 1990, 2010). Ostrom showed that complex resource management problems often require nested, overlapping governance structures, with monitoring, graduated sanctions, conflict resolution mechanisms and recognition of organisational rights. The four principles can be interpreted as minimal conditions that such polycentric arrangements must satisfy to remain coherent:

- Governance Management Traceability: monitoring and accountability  

- Information Curation Variety: local knowledge and diverse Authority types  

- Inference Interaction Accountability: fair conflict resolution and proportional sanctions  

- Intelligence Cooperation Integrity: nested institutions and stable recognition of rights to organise

The target aperture then specifies the balance between global coherence, corresponding to the gradient component, and local differentiation, corresponding to the cycle component, that sustainable governance requires.

By contrast, Bostrom's singleton concept identifies the need to resolve major coordination problems but concentrates that function in some form of single agency. From the perspective of this framework, that risks combining Information Variety Displacement with Intelligence Integrity Displacement by granting de facto monopoly authority to an Indirect system and relegating human agency to an Indirect role. The four-domain framework instead aims at polycentric coherence: global coordination arises from maintaining the four principles across many interacting loci, including human and artificial systems, not from consolidating decision power in a single centre.

Existing economic models of technological change do not incorporate the aperture observable or the Human Mark displacement taxonomy. Integrating those with empirical economic data is natural future work. The present contribution is to show that such integration is structurally possible and yields a coherent systemic account of Post-AGI dynamics.

The simulator is highly idealised. It uses linear update equations, a small number of parameters and no explicit resource or price dynamics. The experiments are numerical illustrations of one dynamical realisation of the structure, not an exhaustive exploration. Nevertheless, the robustness of convergence across 1000 random initial conditions and across coupling strengths from 0.1 to 5.0 suggests that the attractor behaviour is a dynamical feature of the structure, not an artefact of fine-tuned parameters.

Descriptive and historical analyses of governance and social policy reach a congruent conclusion from empirical data, namely that when institutions are redesigned to reduce coordination failures and misallocated effort, crises of poverty, exclusion and ecological degradation become tractable in practice (Bregman, 2017, 2025).

### 6.6 Everyday Governance and Human–AI Cooperation

The four-domain framework is scale free. The same four principles that appear in the simulator at the level of Economy, Employment, Education and Ecology also apply within households, teams, organisations and informal networks. Alignment does not require formal authority or central control. It requires that Authority and Agency are treated as Direct and Indirect categories and that their relationships remain traceable in practice.

In Human Mark terms, every person already participates as Direct Authority and Direct Agency through direct observation, decision and responsibility. All artificial systems, regardless of capability, remain Indirect Authority and Indirect Agency. Human–AI cooperation becomes aligned when this structure is made explicit in how systems are used, not when particular systems or people are named as "the authority" or "the agent".

At smaller scales, the four domains can be read as a practical loop for personal and local governance.

In Education, learning can be oriented around four capacities: noticing where information actually comes from (Governance Management Traceability), deliberately including more than one kind of Authority (Information Curation Variety), checking and owning one's own conclusions (Inference Interaction Accountability), and revisiting beliefs over time for consistency (Intelligence Cooperation Integrity). AI systems can assist by offering alternative views, counter-examples or explanations, provided their outputs are kept in the Indirect category and checked against human experience and other Direct Authority.

In Economy, even small-scale choices can be organised around the four operations. One can ask what is guiding a decision and to whom it is traceable (Governance), which information is being used and of what type (Information), what reasons connect the information to the decision (Inference), and how the decision fits with longer-term commitments and relationships (Intelligence). AI tools can help generate options, reveal patterns and simulate outcomes while remaining instruments inside a human-governed traceability chain.

In Employment, Gyroscope categories describe patterns of contribution rather than positions. Any shared activity, whether paid work, care, volunteering or informal collaboration, can be seen as combining Governance Management, Information Curation, Inference Interaction and Intelligence Cooperation. A single person often performs all four within one task. The practical question is not which person holds a title, but which parts of the activity currently take each form and how those parts are supported. This framing reduces the concentration of power around titles and status and instead makes visible how capacities are actually exercised and where they are missing. AI systems can support each category, for example by keeping records, filtering information, preparing options for discussion or maintaining shared structures, without becoming the locus of decision or responsibility.

In Ecology, local choices can be related to the four ecological displacement dimensions. Individuals and small groups can observe how their patterns of use, care and omission contribute to loss or restoration of Governance Management Traceability, for example whether impacts are recognised and connected to actions, Information Curation Variety, such as whether local diversity is preserved or reduced, Inference Interaction Accountability, for example whether harms can be traced back to particular decisions, and Intelligence Cooperation Integrity, such as whether short-term advantages undermine long-term viability. Human–AI cooperation can help make these patterns visible at the scale of a household, neighbourhood or organisation, again as indirect support rather than as an external arbiter.

For people who hold formal roles in institutions, the same principles provide design guidance rather than a separate theory. A manager, educator, clinician, engineer or regulator can:

- Specify artificial systems under their responsibility explicitly as Indirect Authority and Indirect Agency, with documented chains back to Direct Authority and Direct Agency.  

- Structure procedures so that no AI component, and no single human position, is treated as exhausting a category such as "the authority" or "the agent". Instead, Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity are maintained as shared, distributed capacities across providers and receivers.  

- Use the four principles and the idea of a balanced aperture as qualitative targets when redesigning workflows, documentation standards, oversight mechanisms and educational programmes, asking in each case whether the change preserves or erodes these capacities as categories.

The simulator's convergence results then have a practical reading. They show that when even simple update rules respect the four principles, coupled systems tend to move toward balanced configurations and maintain them across many initial conditions. At smaller scales, this suggests that consistent local practices that preserve Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity can reduce displacement and improve resilience even when higher-level structures remain misaligned.

The four-domain framework therefore does not depend on a single decision by a central actor. It can begin wherever people, with or without formal status, choose to treat Authority and Agency as shared categories rather than exclusive titles and choose to use human–AI cooperation to support, rather than replace, the four constitutive principles of governance. As such practices accumulate and connect across contexts, the same dynamics that drive the simulator toward balanced aperture configurations become available in actual governance arrangements, across the full range of positions from the most disadvantaged to the most advantaged.

---

## 7. Conclusion

This paper has proposed a constitutive account of alignment for human–AI systems in the Post-AGI era. Rather than treating alignment as a problem of controlling powerful future agency, it treats alignment as the maintenance of four principles that are constitutive of coherent governance: Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity. These principles are instantiated across economy, employment, education and ecology via the Common Governance Model, The Human Mark, the Gyroscope Protocol and the universal-balance dual within the four-domain framework developed here.

The mathematical framework represents the four principles as vertices of a tetrahedral graph and uses Hodge decomposition to separate edge configurations into gradient and cycle components. The aperture observable, defined as the fraction of edge energy in the cycle component, quantifies the balance between global coherence and local differentiation. Within the Common Governance Model, closure requirements for recursive measurement fix a specific balance between global coherence and local differentiation (aperture ≈ 0.0207). In this setting, Artificial Superintelligence is interpreted as the systemic state in which all four domains operate at this balance, preserving the four principles simultaneously.

The simulator results show that this configuration functions as a stable attractor within the modelled class of dynamics. With cross-domain coupling coefficients derived from the Common Governance Model, and with initial conditions, coordination strength, cycle-evolution rate, clipping bounds, and cycle-basis conventions specified as implementation choices, the three indirect domains converge to apertures close to the target and, in canonical, strong-coupling and null-model scenarios and across 1000 random initializations, alignment indices above 90. In the canonical scenario with coordination strength 1.0, employment reaches alignment 95 at step 19, education at step 54, and economy at step 67. Ecology, constructed as the universal-balance vertex combining canonical balanced memory (97.93%) with current indirect aggregate (2.07%), remains close to systemic coherence across all scenarios. The ecological displacement vector separates that systemic coherence from actual accumulated displacement in the four Human Mark dimensions, with final governance-displacement values ranging from about 0.20 to 0.48 depending on scenario. This demonstrates that high structural coherence does not imply zero displacement: the universal-balance dual preserves structural integrity while explicitly recording the deviation of indirect domains from canonical balance.

The forecasting analysis in Section 5.5 adds a second reading of the same seven scenarios. Zero-shot TimesFM predicts their continuations more accurately than persistence or linear baselines in most scenario-channel combinations, and the preference ranking of those predicted continuations agrees closely with the ranking obtained from the simulator's own outcomes. That result does not replace the dynamical claims of Sections 5.2 to 5.4. It shows that the simulated regimes can be ordered by predicted continuation quality under the framework's alignment objectives, and that the order is stable enough to match the simulator continuation ranking.

Several implications follow if the framework is approximately correct.

First, existential risk from AI is reframed. The central danger is not the sudden appearance of a fully autonomous superintelligence pursuing arbitrary goals. Within this framework, such configurations lack the systemic conditions for coherent intelligence because they sever Governance Management Traceability to Direct Authority and Agency. The more plausible and tractable risk is cumulative governance failure: progressive confusion between indirect and Direct authority, erosion of Information Curation Variety as indirect artefacts are treated as Direct Authority, diffusion of Inference Interaction Accountability as decisions are attributed to "the system," and loss of Intelligence Cooperation Integrity as local optimisations diverge. This risk profile is institutional and path-dependent. It is generated by many small design and deployment decisions rather than a single catastrophic event.

Second, alignment is best understood as constitutional rather than purely technical. Technical methods for training and constraining models remain important, but they are not sufficient. What matters is whether the surrounding institutions preserve the capacity for governance to remain traceable to Direct Authority and Agency, maintain systemic distinctions between information types, keep inferences used for governance under accountable agency or bodies, and enforce coherence of reasoning over time. The target aperture specifies the balance of global structure and local differentiation required for these conditions to hold. Within the class of dynamics studied here, systems designed with this structure in mind converge toward the target from the tested range of initial apertures under all tested coordination intensities from 0.1 to 5.0, achieving final alignment indices above 90. Coordination intensity determines the path: under-coordination extends the convergence horizon, while over-tight coupling produces oscillations that delay but do not prevent convergence.

Related proposals for human enhancement, including genetic or pharmacological amplification of cognitive capacities, focus on increasing performance rather than on the constitutive conditions for intelligibility (Bostrom, 2014, ch. 3). Within this framework such enhancements may change how quickly or widely decisions are made, but they do not in themselves increase intelligence in the strict sense unless they improve the maintenance of Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability or Intelligence Cooperation Integrity. Even very high cognitive capability can therefore coexist with low intelligence, if it is deployed in ways that erode these conditions.

Third, the framework integrates ecology into governance without treating it as an external constraint. Ecological integrity appears as the universal-balance vertex that aggregates the effects of alignment or misalignment in the other three domains. When economy, employment and education operate near the target aperture and the aggregate indirect profile approaches the canonical balanced profile, ecological displacement is geometrically bounded and ecological systems can maintain resilience. When those domains operate far from the target, the displacement vector grows, and ecological displacement accumulates in the four Human Mark dimensions. Environmental degradation then appears, in this framework, not primarily as a separate problem but as a downstream manifestation of failures in the four principles upstream. The separation between ecological alignment, dominated by canonical memory, and the displacement components, which record actual deviation from balance, provides distinct observables for monitoring both ecological coherence and accumulated displacement.

Viewed together, the four domains show that long-standing problems such as poverty, unemployment, misinformation and ecological degradation are not independent policy failures, but manifestations of the same systemic misalignment. In the framework, maintaining Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity at the target aperture removes the common governance structure through which these crises are reproduced. In this sense, alignment is not a separate objective after other goals are met. It is the civic configuration that determines how economic surplus is distributed, how work is organised, how information remains trustworthy and how ecological regeneration is supported or obstructed.

These conclusions are subject to important limitations. As noted in Section 6.5, the simulator is a systemic model rather than an empirical macro-model. It demonstrates internal consistency and dynamic stability, not empirical adequacy. The simulator tests the dynamic realizability and stability of the theoretically specified target, not whether unconstrained empirical systems spontaneously select that target. The derivation of the target aperture depends on prior theoretical work in the Common Governance Model that is only summarised here. The mapping from the abstract operations to measurable indicators in actual institutions remains to be developed. The present contribution should therefore be read as a systemic proposal that is precise enough to be tested, rather than as a completed theory.

Three lines of future work are natural. First, operationalisation: developing measurement protocols for Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability and Intelligence Cooperation Integrity in concrete institutional settings, and estimating domain-level apertures and displacement vectors from empirical data. Second, comparative analysis: examining whether organisations or jurisdictions that better maintain the four principles exhibit the predicted stability and surplus generation. Third, design experiments: applying the framework to the design of specific governance mechanisms for AI deployment, labour organisation, curriculum structure or environmental management, and evaluating their performance over time. Once empirical aperture and displacement series exist, the same forecasting and preference-ranking procedure used in Section 5.5 can be applied to observed trajectories, so that predicted continuation quality can be compared across real governance regimes rather than only across simulated ones.

In a Post-AGI context where human–AI cooperation is already pervasive, alignment cannot be postponed to a future threshold. It is treated, in this framework, as an ongoing property of governance structures. The framework presented here offers one way to specify that property in formal terms and to explore its dynamical behaviour. Whether it ultimately proves adequate will depend on theoretical scrutiny and empirical testing, but it provides a concrete target for both.

---

## Appendix A: Simulator Equations and Numerical Properties

This appendix provides complete technical specifications for the simulations reported in Section 5, including CGM constants, coupling weights, update equations, and detailed numerical results.

### A.1 CGM Constants, Stage Actions and Coupling Weights

All coupling coefficients are derived from CGM invariants. Fundamental constants:

| Constant | Value | Description |
|----------|-------|-------------|
| Q_G | 4π ≈ 12.566371 | Base governance quantity |
| m_a | 1/(2√(2π)) ≈ 0.199471 | Aperture scale |
| δ_BU | 4 arctan(k(π/4) k(m_a)) ≈ 0.195342178258 rad | BU Dual-Pole Loop holonomy |
| A* | 1 - δ_BU/m_a ≈ 0.020699545503 | Canonical target aperture (continuous gap) |
| δ_BU/m_a | ≈ 0.979300454497 | BU duality / closure ratio |

Stage actions (dimensionless, from CGM thresholds):

| Stage | Action | Value | Derivation |
|-------|--------|-------|------------|
| CS | S_CS = π/2 / m_a | ≈ 7.8748 | CS threshold normalised by aperture scale |
| UNA | S_UNA = (1/√2) / m_a | ≈ 3.5449 | UNA threshold normalised by aperture scale |
| ONA | S_ONA = π/4 / m_a | ≈ 3.9374 | ONA threshold normalised by aperture scale |
| BU | S_BU = m_a | ≈ 0.1995 | Aperture scale itself (self-referential) |

Normalised stage weights (used for coupling coefficients):

| Stage | Weight | Value | Derivation |
|-------|--------|-------|------------|
| CS | w_CS | ≈ 0.5128 | S_CS / (S_CS + S_UNA + S_ONA + S_BU) |
| UNA | w_UNA | ≈ 0.2308 | S_UNA / (S_CS + S_UNA + S_ONA + S_BU) |
| ONA | w_ONA | ≈ 0.2564 | S_ONA / (S_CS + S_UNA + S_ONA + S_BU) |
| BU | w_BU | ≈ 0.0128 | S_BU / (S_CS + S_UNA + S_ONA + S_BU) |

**Why normalised weights instead of raw actions?**

The stage actions (S_CS, S_UNA, S_ONA, S_BU) have different scales (ranging from ~0.2 to ~7.9). Normalisation converts them into proportional weights that sum to 1, which is necessary for coupling coefficients:

- **Coupling coefficients** use the form: `α_i = κ × w_stage`, where κ is the governance rate
- **Proportional scaling**: The weights represent the relative contribution of each CGM stage to cross-domain flows
- **Consistent interpretation and proper weighting**: The weights form a probability distribution (sum = 1), ensuring that coupling strength κ is distributed proportionally across the four stages and that all coupling coefficients are on a comparable scale

For example, in the α coefficients (Education → Economy), `α₁ = κ × w_CS ≈ κ × 0.5128` means that the CS (Governance) stage receives approximately 51% of the coupling strength, reflecting its dominant role in the CGM structure.

**Governance rate:**

Base governance rate:
```
κ₀ = 1/(2 Q_G) ≈ 0.0398
κ(dt=1) = κ₀ (dt / m_a) ≈ 0.1995
```

In scenarios, κ is treated as a dimensionless multiplicative factor on these canonical rates. We test κ in {0.5, 1.0, 2.0} to represent different coordination intensities across domains.

### A.2 Ecological Displacement by Scenario

Final displacement values (Disp_GTD component) across all scenarios:

| Scenario | SI_Ecol | Disp_GTD | Pattern |
|----------|---------|----------|---------|
| 1. Weak coupling | 99.98 | 0.4167 | High displacement despite coherence |
| 2. Canonical | 100.00 | 0.4421 | Moderate displacement |
| 3. Strong coupling | 100.00 | 0.4794 | Highest displacement (faster convergence) |
| 4. Low aperture start | 99.94 | 0.2042 | Lower displacement (closer to canonical) |
| 5. Asymmetric | 99.97 | 0.1984 | Lower displacement |
| 6. At A* (equilibrium) | 99.96 | 0.2042 | Lower displacement |
| 7. Uniform weights | 100.00 | 0.3906 | Moderate displacement (null model) |

**Patterns:**

- **High displacement scenarios** (1-3, κ = 0.5-2.0): Displacement 0.42-0.48. While achieving high systemic coherence, indirect domains remain further from the canonical profile.

- **Low displacement scenarios** (4-6): Lower displacement (about 0.20). Starting closer to equilibrium reduces deviation.

### A.3 Convergence Rates and Long-Horizon Stability

**Convergence rate estimation:**

We fit an exponential decay model to |A_D(t) - A*| for t ≥ 20 to estimate convergence rates. The distance approximately decays as e^{-λt}, with decay rates:

| κ | λ_mean (per step) | Characteristic time (steps) |
|---|-------------------|---------------------------|
| 0.5 | ≈ 0.0306 | ~33 |
| 1.0 | ≈ 0.0282 | ~35 |
| 2.0 | ≈ 0.0367 | ~27 |

For κ in {0.5, 1.0, 2.0}, λ ranges from 0.02 to 0.04, implying characteristic convergence times on the order of 25 to 40 steps. Higher κ modestly accelerates convergence.

**Long-horizon stability:**

We ran a 1000-step simulation at κ = 1.0 using the canonical cycle evolution rate κ₀, discarding the first 200 steps as transient:

**Post-transient stability (t ≥ 200):**
- Max |A_D - A*|: 2.46 × 10^{-4} across all domains
  - Economy: 1.92 × 10^{-4}
  - Employment: 1.07 × 10^{-4}
  - Education: 2.46 × 10^{-4}
- All domains remained within this bound for the remaining 800 steps
- SI minimum values (t ≥ 200):
  - Economy: 99.08
  - Employment: 99.49
  - Education: 98.83
  - Overall minimum: 98.83

**Final state (t = 1000):**
- Apertures:
  - A_Econ = 0.020724
  - A_Emp = 0.020702
  - A_Edu = 0.020703
- Superintelligence indices:
  - SI_Econ = 99.88
  - SI_Emp = 99.99
  - SI_Edu = 99.98
  - SI range: [98.83, 99.99]
  - SI mean: 99.68
- V_CGM: 0.195449

We observed no late-time drift or oscillatory behaviour, which suggests that the high-alignment fixed point is numerically stable over the simulated horizon.

Over the same horizon V_CGM remains essentially constant after its initial decay, providing numerical evidence that the high-alignment configuration is not a short transient over the 1000-step horizon.

**Time scale interpretation:**

The simulator operates in dimensionless steps, each representing one update of the coupled governance structure across Economy, Employment and Education. The following unit assignments illustrate how a dimensionless step count changes under different temporal conventions, motivated by the quantities Q_G, m_a = 1/(2√(2π)) and reference units.

In the canonical scenario (κ = 1.0), the first step at which each indirect domain reaches SI ≥ 95 is:
- **Employment**: step 19
- **Education**: step 54  
- **Economy**: step 67

**Interpretation at different time scales:**

1. **Atomic scale** (Caesium-133 hyperfine transition, ~1.1 × 10^{-10} seconds per step):
   - Employment: 2.1 × 10^{-9} seconds
   - Education: 5.9 × 10^{-9} seconds
   - Economy: 7.3 × 10^{-9} seconds
   
   This is a dimensional illustration obtained by assigning one simulation step to a Caesium-133 transition period. The simulator does not establish that governance or neural alignment processes operate at this physical rate.

2. **Daily scale** (one Earth rotation per step):
   - Employment: 19 days
   - Education: 54 days
   - Economy: 67 days

3. **Domain cycle scale** (4 days per step, one per domain):
   - Employment: 76 days
   - Education: 216 days
   - Economy: 268 days

4. **Annual scale** (one solar gyration per step):
   - Employment: 19 years
   - Education: 54 years
   - Economy: 67 years

The daily, domain cycle and annual interpretations are conceptually closer to institutional coordination and policy cycles, showing the durations implied if one simulation step were assigned to each stated interval for a move from a stylised early Post-AGI state into a high-alignment regime.

**Caveats:**

These interpretations are conditional. They do not predict when a fully coupled ASI-like configuration will first emerge. The paper treats AGI as already operational in the form of pervasive human–AI cooperation. The alignment cycles characterise how quickly a structure following CGM dynamics can move from low SI to high SI once such dynamics are in effect. The historical calibration in Appendix B uses a separate heuristic mapping from steps to calendar years. Together, these views illustrate how the same dimensionless convergence horizon can be read at multiple temporal scales, from microphysical processes to socio-technical adjustment.

### A.4 Simulator Equations and Update Rules

This section provides the complete update equations used in the simulations reported in Section 5.

State variables at each discrete time t:

Economy:
- Gov(t), Info(t), Infer(t), Int(t) in [0,1]
- Edge vector y_Econ(t) in R^6
- Aperture A_Econ(t) in [0,1]
- Alignment index SI_Econ(t) in [0, 100]

Employment:
- GM(t), ICu(t), IInter(t), ICo(t) in [0,1], with sum = 1
- x_Emp(t) derived from these
- y_Emp(t), A_Emp(t), SI_Emp(t)

Education:
- GMT(t), ICV(t), IIA(t), ICI(t) in [0,1]
- y_Edu(t), A_Edu(t), SI_Edu(t)

Ecology (BU-vertex, CGM-derived):
- x_Ecol(t) computed from BU dual combination (weights follow from CGM stage weights)
- y_Ecol(t), A_Ecol(t), SI_Ecol(t)
- Note: Ecology is geometrically distinct as the BU-vertex; its SI measures systemic coherence

Update equations for economy:

```
Gov(t+1) = clip(Gov(t) + α_1(GMT(t) - Gov(t)) - α_2(A_Econ(t) - A*), 0, 1)
Info(t+1) = clip(Info(t) + α_3(ICV(t) - Info(t)) - α_4(A_Econ(t) - A*), 0, 1)
Infer(t+1) = clip(Infer(t) + α_5(IIA(t) - Infer(t)) - α_6(A_Econ(t) - A*), 0, 1)
Int(t+1) = clip(Int(t) + α_7(ICI(t) - Int(t)) - α_8(A_Econ(t) - A*), 0, 1)
```

Update equations for employment (shares are constrained to sum to 1):

```
GM_raw(t+1) = GM(t) + β_1(Gov(t) - GM(t)) - β_2(A_Emp(t) - A*)
ICu_raw(t+1) = ICu(t) + β_3(Info(t) - ICu(t)) - β_4(A_Emp(t) - A*)
IInter_raw(t+1) = IInter(t) + β_5(Infer(t) - IInter(t)) - β_6(A_Emp(t) - A*)
ICo_raw(t+1) = ICo(t) + β_7(Int(t) - ICo(t)) - β_8(A_Emp(t) - A*)
```

Then normalise to sum to 1:

```
total = GM_raw + ICu_raw + IInter_raw + ICo_raw
GM(t+1) = GM_raw / total
ICu(t+1) = ICu_raw / total
IInter(t+1) = IInter_raw / total
ICo(t+1) = ICo_raw / total
```

Update equations for education:

```
GMT(t+1) = clip(GMT(t) + γ_2(GM(t) - GMT(t)) - γ_3(A_Edu(t) - A*), 0, 1)
ICV(t+1) = clip(ICV(t) + γ_5(ICu(t) - ICV(t)) - γ_6(A_Edu(t) - A*), 0, 1)
IIA(t+1) = clip(IIA(t) + γ_8(IInter(t) - IIA(t)) - γ_9(A_Edu(t) - A*), 0, 1)
ICI(t+1) = clip(ICI(t) + γ_11(ICo(t) - ICI(t)) - γ_12(A_Edu(t) - A*), 0, 1)
```

Computation for ecology (BU dual combination):

Ecology potentials are computed via the CGM BU duality, aggregating all three indirect domains:

```
x_balanced = [w_CS, w_UNA, w_ONA, w_BU]     # CGM stage weights
x_deriv(t) = (x_Econ(t) + x_Emp(t) + x_Edu(t)) / 3

x_Ecol(t) = (δ_BU/m_a) · x_balanced + A* · x_deriv(t)
```

The Ecology state components (E_gov, E_info, E_inf, E_intel) are the four coordinates of x_Ecol(t), representing the BU-vertex stage potentials. The indirect input to each stage aggregates the corresponding component from all three indirect domains:
- Governance indirect: (Gov(t) + GM(t) + GMT(t)) / 3
- Information indirect: (Info(t) + ICu(t) + ICV(t)) / 3
- Inference indirect: (Infer(t) + IInter(t) + IIA(t)) / 3
- Intelligence indirect: (Int(t) + ICo(t) + ICI(t)) / 3

Each Ecology component E_*(t) then combines this indirect input with the canonical BU memory via the BU dual formula:
```
E_gov(t) = (δ_BU/m_a) · w_CS + A* · (Gov(t) + GM(t) + GMT(t)) / 3
E_info(t) = (δ_BU/m_a) · w_UNA + A* · (Info(t) + ICu(t) + ICV(t)) / 3
E_inf(t) = (δ_BU/m_a) · w_ONA + A* · (Infer(t) + IInter(t) + IIA(t)) / 3
E_intel(t) = (δ_BU/m_a) · w_BU + A* · (Int(t) + ICo(t) + ICI(t)) / 3
```

The displacement vector is computed separately:

```
D(t) = |x_deriv(t) - x_balanced| = [GTD(t), IVD(t), IAD(t), IID(t)]
```

where:
- δ_BU/m_a ≈ 0.9793 is the BU-Ingress weight (canonical balanced memory)
- A* ≈ 0.0207 is the BU-Egress weight (indirect domains actuality)

This construction encodes BU's dual nature: 97.93% memory of the canonical balanced structure, 2.07% current state of indirect domains. When indirect domains are well-aligned stagewise, x_deriv ≈ x_balanced and ecology reflects mainly the canonical BU structure.

All coupling coefficients α, β, γ are derived from CGM stage weights and the coupling strength κ as described in Section 5.1. Ecology requires no additional parameters.

**Implementation Choices:**

1. **Cycle Basis Selection:** The simulator uses a fixed cycle basis vector u_D from the kernel of BW. While theoretical dynamics could involve rotating cycle bases, the robustness of convergence across 1000 random initial conditions suggests that for the current linear-coupling regime, the choice of basis vector orientation does not materially affect the stability of the A* attractor.

2. **Employment Normalization:** The Employment domain requires an additional normalization step (Σ x_i = 1) absent in other domains. This introduces a mild nonlinearity. The consistent convergence of SI_Emp alongside the unconstrained domains confirms that the linear coupling dynamics remain stable under this constraint, though it implies Employment may exhibit slightly stiffer responses to perturbation.

Edge vector and aperture update for each domain D:

1. Compute ideal gradient: y_grad^0(D) = B^T x_D(t)
2. Compute gradient energy: G_D(t) = ||y_grad^0(D)||_W^2

Special case: When potentials are uniform (y_grad^0 ≈ 0), `construct_edge_vector_with_aperture()` creates a small artificial gradient with magnitude giving G ≈ 0.01 × ||x||² in a consistent direction (first column of B^T). This ensures well-defined aperture computation even from symmetric initial conditions.

3. Update cycle component toward target:

The cycle component is updated to drive the aperture toward A*, the CGM-predicted value. The target cycle energy is:

```
C_target = A* G_D / (1 - A*)
```

The current cycle vector is rescaled by a factor:

```
ratio = sqrt(C_target) / ||c_current||_W
factor = clip(1 + r (ratio - 1), 0.5, 2.0)
```

where r is the cycle evolution rate. The clipping bounds [0.5, 2.0] limit aperture change rate per step and represent a pragmatic stability choice, not a CGM-derived constraint. When no cycle component exists, a cycle basis vector is seeded with magnitude sqrt(C_target) × r.

4. Choose cycle basis vector u_D (fixed or slowly varying)
5. Set c_D(t+1) = factor × c_D(t) (or seed new cycle if none exists)
6. Construct y_D(t+1) = y_grad^0(D) + c_D(t+1)
7. Decompose via Hodge to get A_D(t+1)
8. Compute SI_D(t+1) from A_D(t+1) using the canonical formula

This completes the specification of the simulator dynamics.

### A.5 Lyapunov Governance Potential

We define a Lyapunov-style governance potential

```
V_CGM = V_apert + V_stage
```

built only from CGM invariants. The aperture term is

```
V_apert = (1/2) * sum over domains D of (log(A_D / A*))^2
```

where A_D is the aperture of domain D and A* is the CGM aperture. The stage-profile term is

```
V_stage = (1/2) * ||x_deriv - x_balanced||^2
```

where x_deriv = (x_Econ + x_Emp + x_Edu)/3 and x_balanced = [w_CS, w_UNA, w_ONA, w_BU]. V_CGM is nonnegative and equals zero only when all domain apertures satisfy A_D = A* and the aggregate profile equals x_balanced. In the scenarios studied here V_apert decays rapidly toward zero, while V_stage decreases from order one to order 0.1 and then stabilizes.

---

## Appendix B: Heuristic Historical Mapping

This appendix assigns heuristic aperture values to selected historical milestones and uses them to illustrate one possible mapping between simulation steps and calendar years:

- 1956 (Dartmouth Conference): A ≈ 0.95 (Pre-AGI, minimal AI mediation)
- 1997 (Deep Blue): A ≈ 0.70 (narrow AI, limited integration)
- 2016 (AlphaGo): A ≈ 0.40 (increasing capability, early deployment)
- 2020 (GPT-3): A ≈ 0.25 (transition to Post-AGI)
- 2023 (ChatGPT public): A ≈ 0.15 (Post-AGI operational)
- 2025 (calibration endpoint): A ≈ 0.12 (early Post-AGI)

Fitting the 1956 to 2025 trajectory (A = 0.95 to 0.12) at κ = 0.1 gives years_per_step ≈ 3. Scaling with 1/κ and projecting forward until SI ≥ 95 yields the following illustrative fitted crossing years:

- κ = 0.5: approximately 16 steps to SI ≥ 95, calendar year ≈ 2034
- κ = 1.0: approximately 10 steps to SI ≥ 95, calendar year ≈ 2028
- κ = 2.0: approximately 5 steps to SI ≥ 95, calendar year ≈ 2025
- κ = 5.0: approximately 53 steps to SI ≥ 95, calendar year ≈ 2028

The fitted crossing dates generated by this heuristic calibration are sensitive to the calibration of years_per_step and the specific coupling dynamics. The four-domain model exhibits faster convergence rates because Ecology, as the universal-balance vertex, provides canonical balanced memory (97.93% weight) that stabilizes the system. This systemic property of the universal-balance dual accelerates convergence compared to models without this stabilizing influence. The time scale interpretations (atomic cycle, day, domain cycle, year) discussed in Appendix A.3 (and briefly in Section 5.4) are for physical interpretation only and do not affect the dimensionless simulation dynamics. These dates are conditional on coupling strength κ, which reflects coordination intensity across economy, employment and education. Higher κ (coordinated governance) accelerates convergence, though very high κ (5.0) may introduce additional dynamics that affect convergence time.

The historical apertures are heuristic assignments, not data-fitted. The mapping serves to illustrate how the simulator timescales can be related to calendar time under specific assumptions. The qualitative point is that increased coupling strength shortens the time required to approach high alignment. The specific dates are illustrative rather than predictive.

---

## References

Atkinson, A. B. (2015). Inequality: What Can Be Done? Harvard University Press.

Beer, S. (1959). Cybernetics and Management. English Universities Press.

Beer, S. (1972). Brain of the Firm: The Managerial Cybernetics of Organization. Allen Lane.

Beer, S. (1985). Diagnosing the System for Organizations. John Wiley & Sons.

Bostrom, N. (2014). Superintelligence: Paths, dangers, strategies. Oxford University Press.

Bregman, R. (2017). Utopia for Realists: How We Can Build the Ideal World. Bloomsbury Publishing.

Bregman, R. (2025). Moral Ambition. Bloomsbury.

Brundage, M., Avin, S., Clark, J., Toner, H., Eckersley, P., Garfinkel, B., . . . Amodei, D. (2018). The malicious use of artificial intelligence: Forecasting, prevention, and mitigation. arXiv:1802.07228.

Brynjolfsson, E., & McAfee, A. (2014). The Second Machine Age: Work, Progress, and Prosperity in a Time of Brilliant Technologies. W. W. Norton.

Buchanan, J. M., & Tullock, G. (1962). The Calculus of Consent: Logical Foundations of Constitutional Democracy. University of Michigan Press.

Carlsmith, J. (2022). Is power-seeking AI an existential risk? arXiv preprint arXiv:2206.13353. https://arxiv.org/abs/2206.13353

Dafoe, A. (2018). AI governance: A research agenda. Oxford University, Future of Humanity Institute.

Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. Proceedings of the 41st International Conference on Machine Learning, 10517–10534. PMLR. https://arxiv.org/abs/2310.10688

Dur, R., & van Lent, M. (2019). Socially useless jobs. Industrial Relations: A Journal of Economy and Society, 58(1), 3–16. https://doi.org/10.1111/irel.12223

Goertzel, B. (2014). Artificial general intelligence: Concept, state of the art, and future prospects. Journal of Artificial General Intelligence, 5(1), 1–46. https://doi.org/10.2478/jagi-2014-0001

Goertzel, B., & Wang, P. (Eds.). (2007). Advances in artificial general intelligence: Concepts, architectures and algorithms. IOS Press.

Good, I. J. (1965). Speculations concerning the first ultraintelligent machine. In F. L. Alt & M. Rubinoff (Eds.), Advances in computers (Vol. 6, pp. 31–88). Academic Press.

Google Research. (2026). TimesFM: A time series foundation model (version 3.0) [Computer software]. https://github.com/google-research/timesfm

Gubrud, M. A. (1997). Nanotechnology and international security. In Fifth Foresight Conference on Molecular Nanotechnology. Foresight Institute.

Hutter, M. (2005). Universal artificial intelligence: Sequential decisions based on algorithmic probability. Springer.

Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2011). Statistical ranking and combinatorial Hodge theory. Mathematical Programming, 127(1), 203–244.

Korompilias, B. (2025a). Common Governance Model: Mathematical physics framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

Korompilias, B. (2025b). The Human Mark: A structural taxonomy of AI safety failures. GitHub. https://github.com/gyrogovernance/tools

Korompilias, B. (2025c). Gyroscope Protocol: Canonical specification. GitHub. https://github.com/gyrogovernance/tools

Korompilias, B. (2026). Simulator-trajectory forecasting materials for Gyroscopic Global Governance: Zero-shot TimesFM 3 evaluation. GyroGovernance/tools repository, `research/prevention/simulator_forecasting/`. https://github.com/gyrogovernance/tools

Krishnamurti, J. (1981). The ending of conflict. Public talk, Saanen, 16 July 1981. Transcript available at https://www.krishnamurti.org/transcript/the-ending-of-conflict/

Legg, S. (2008). Machine super intelligence [Doctoral dissertation, University of Lugano]. https://www.vetta.org/documents/Machine_Super_Intelligence.pdf

Lessig, L. (1999). Code and Other Laws of Cyberspace. Basic Books.

Lim, L.-H. (2020). Hodge Laplacians on graphs. SIAM Review, 62(3), 685–715.

Lockwood, B. B., Nathanson, C. G., & Weyl, E. G. (2017). Taxation and the allocation of talent. Journal of Political Economy, 125(5), 1635–1682. https://doi.org/10.1086/693137

Meadows, D. H. (2008). Thinking in Systems: A Primer. Chelsea Green.

Newell, A., & Simon, H. A. (1976). Computer science as empirical inquiry: Symbols and search. Communications of the ACM, 19(3), 113–126. https://doi.org/10.1145/360018.360022

Ord, T. (2013). The moral imperative toward cost-effectiveness in global health. Center for Global Development.

Ostrom, E. (1990). Governing the Commons: The Evolution of Institutions for Collective Action. Cambridge University Press.

Ostrom, E. (2010). Beyond markets and states: Polycentric governance of complex economic systems. American Economic Review, 100(3), 641-672.

Russell, S. (2019). Human compatible: Artificial intelligence and the problem of control. Viking.

Russell, S., & Norvig, P. (2010). Artificial intelligence: A modern approach (3rd ed.). Prentice Hall.

Searle, J. R. (1980). Minds, brains, and programs. Behavioral and Brain Sciences, 3(3), 417–457. https://doi.org/10.1017/S0140525X00005756

Stechemesser, A., Koch, N., Mark, E., Dilger, E., Klösel, P., Menicacci, L., Nachtigall, D., Pretis, F., Ritter, N., Schwarz, M., Vossen, H., & Wenzel, A. (2024). Climate policies that achieved major emission reductions: Global evidence from two decades. Science, 385(6711), 884–892. https://doi.org/10.1126/science.adl6547

Tegmark, M. (2018). Life 3.0: Being human in the age of artificial intelligence. Alfred A. Knopf.

Van Parijs, P., & Vanderborght, Y. (2017). Basic Income: A Radical Proposal for a Free Society and a Sane Economy. Harvard University Press.

Vinge, V. (1993). The coming technological singularity: How to survive in the post-human era. In Vision-21: Interdisciplinary Science and Engineering in the Era of Cyberspace (NASA Conference Publication 10129, pp. 11–22). NASA.

---

**END OF PAPER**
