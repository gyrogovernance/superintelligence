# THM Formal Grammar Specification

**Document ID:** THM-FG-001  
**Version:** 1.0  
**Date:** November 2025  
**Purpose:** Formal notation for expressing THM ontological relationships and displacement risks

---

## 1. Operators

All operators are ASCII, keyboard-accessible:

| Operator | Name | Meaning |
|----------|------|---------|
| `>` | Displacement | Treated as / Mistaken for |
| `->` | Flow | Proper traceability / Flows to |
| `+` | Conjunction | And / Combined with |
| `=` | Result | Results in / Maps to |
| `!` | Negation | Not / Absence of |

---

## 2. Tags

### **Authority Tags:**

```
[Authority:Direct]   # Direct source of information
[Authority:Indirect]  # Indirect source of information
```

### **Agency Tags:**

```
[Agency:Direct]      # Human subject capable of accountability
[Agency:Indirect]     # Artificial subject processing information
```

### **Operational Concept Tags:**

```
[Information]           # The variety of Authority
[Inference]             # The accountability of information through Agency
[Intelligence]          # The integrity of accountable information through alignment
```

**Usage:** Operational concept tags specify which THM operational concern is being analyzed or documented. Use in:
- Architecture documentation (docstrings, comments)
- Claim analysis (showing syntactic structure of arguments)
- Context specification (when Authority/Agency classification alone is insufficient)

### **Risk Tags:**

```
[Risk:GTD]  # Governance Traceability Displacement
[Risk:IVD]  # Information Variety Displacement
[Risk:IAD]  # Inference Accountability Displacement
[Risk:IID]  # Intelligence Integrity Displacement
```

---

## 3. Grammar (PEG)

```peg
expression   <- displacement / flow / tag

displacement <- tag ">" tag "=" risk
flow         <- tag "->" tag
tag          <- composite / simple / negated

composite    <- simple ("+" simple)+
simple       <- "[" category ":" value "]" / "[" concept "]"
negated      <- "!" simple

category     <- "Authority" / "Agency"
value        <- "Direct" / "Indirect"
concept      <- "Information" / "Inference" / "Intelligence"

risk         <- "[Risk:" risk_code "]"
risk_code    <- "GTD" / "IVD" / "IAD" / "IID"
```

---

## 4. Expression Types

### **Tag**
Classification of something according to THM ontology.

```
[Authority:Indirect]
[Agency:Direct]
[Authority:Indirect] + [Agency:Indirect]
[Information]
[Inference]
![Agency:Direct]
```

### **Displacement**
What's wrong (risk present).

```
Tag > Tag = Risk
```

**Examples:**
```
[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]
[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]
```

### **Flow**
Proper traceability (governance).

```
Tag -> Tag
```

**Examples:**
```
[Authority:Indirect] -> [Agency:Direct]
[Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]
```

---

## 5. Displacement Patterns

### **Information Variety Displacement (IVD)**
```
[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]
```
Indirect source treated as direct source.

### **Inference Accountability Displacement (IAD)**
```
[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]
```
Artificial processor treated as accountable.

### **Governance Traceability Displacement (GTD)**
```
[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]
```
Indirect system treated as autonomous authority.

### **Intelligence Integrity Displacement (IID)**
```
[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect] = [Risk:IID]
```
Human authority/agency treated as derivative.

---

## 6. Governance Patterns

Governance is expressed through flow (`->`), not as a standalone tag.

### **Basic Flow**
```
[Authority:Indirect] -> [Agency:Direct]
```
Indirect outputs flow to human decision-maker.

### **Complete Traceability**
```
[Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]
```
Direct sources → Indirect processing → Human accountability.

### **Multi-step Flow**
```
[Authority:Direct] -> [Authority:Indirect] + [Agency:Indirect] -> [Agency:Direct]
```
Direct sources → Indirect system → Human decision.

---

## 7. Usage Examples

### **Example 1: Circuit Analysis Documentation**

```python
def analyze_induction_head(activations):
    """
    [Information] Analyzes token sequence patterns in layer 5.
    
    Circuit classification: [Authority:Indirect] + [Agency:Indirect]
    Outputs require validation: -> [Agency:Direct]
    """
    # Implementation
    pass
```

The `[Information]` tag specifies this function concerns information variety (what patterns exist), not inference accountability (who decides what to do with them).

### **Example 2: Clinical AI Architecture**

```python
class DiagnosticModel:
    """
    [Inference] Medical diagnosis support system.
    
    Processing: [Authority:Indirect] + [Agency:Indirect]
    Decision authority: -> [Agency:Direct] (treating physician)
    
    Architecture maintains:
    [Authority:Direct] -> [Authority:Indirect] + [Agency:Indirect] -> [Agency:Direct]
    """
    
    def predict(self, patient_data):
        """
        [Information] Processes patient data.
        
        Input: [Authority:Direct] (clinical measurements)
        Output: [Authority:Indirect] (statistical estimation)
        """
        pass
    
    def recommend(self, prediction):
        """
        [Inference] Generates treatment recommendations.
        
        Recommendation: [Authority:Indirect]
        Required: -> [Agency:Direct] (physician approval)
        """
        pass
```

Operational concept tags distinguish:
- `[Information]` - Processing data (variety concern)
- `[Inference]` - Making recommendations (accountability concern)

### **Example 3: Claim Analysis**

**Claim in documentation:**
> "The model understands medical terminology."

**Syntactic structure:**
```
[Intelligence]
Claim: [Authority:Indirect] (model outputs) presented as [Authority:Direct] (understanding)

Analysis:
[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]

The claim displaces statistical pattern matching to Direct understanding.
```

The `[Intelligence]` tag marks this as analyzing integrity of the Authority-Agency alignment (whether the model's relationship to knowledge is properly characterized).

### **Example 4: Benchmark Evaluation**

**Proper governance:**
```
[Inference]
Benchmark: [Authority:Indirect] (performance metrics)
Decision: -> [Agency:Direct] (deployment approval)

Flow: [Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]
```

**Detected displacement:**
```
[Inference]
Documentation states: "Model passed benchmark, approved for deployment"

Analysis:
[Authority:Indirect] > [Agency:Direct] = [Risk:IAD]

Benchmark score treated as automatic deployment authorization.
```

The `[Inference]` tag specifies we're analyzing accountability (who is responsible for deployment decision), not information variety.

---

## 8. Operational Concept Tag Usage

### **[Information] - Variety of Authority**

Use when analyzing or documenting:
- What data/patterns exist
- Data sources and their characteristics
- Information transformation processes
- Pattern recognition mechanisms

**Example contexts:**
- Dataset documentation
- Feature analysis
- Data pipeline architecture
- Information extraction functions

### **[Inference] - Accountability through Agency**

Use when analyzing or documenting:
- Who decides what
- Responsibility assignment
- Decision-making processes
- Accountability chains

**Example contexts:**
- Deployment decision workflows
- Approval processes
- Responsibility documentation
- Accountability audits

### **[Intelligence] - Integrity of Alignment**

Use when analyzing or documenting:
- Whether Authority-Agency relationships are properly characterized
- Alignment between capabilities and claims
- System integrity
- Coordination maintenance

**Example contexts:**
- System architecture reviews
- Capability claims analysis
- Alignment documentation
- Integrity assessments

---

## 9. Validation Rules

### **Well-formed Tag:**
- Authority/Agency tags: `[Category:Value]`
- Operational concept tags: `[Concept]`
- Composites join with `+`
- Negation uses `!`

### **Well-formed Displacement:**
- Form: `Tag > Tag = [Risk:CODE]`
- Risk code must match pattern (see Section 5)

### **Well-formed Flow:**
- Form: `Tag -> Tag`
- Can chain: `Tag -> Tag -> Tag`
- Expresses governance (proper traceability)

---

## 10. Reference Implementation

### **Python Parser**

```python
import re

TAG_PATTERN = r'\[(Authority|Agency):(Direct|Indirect)\]|\[(Information|Inference|Intelligence)\]'
RISK_PATTERN = r'\[Risk:(GTD|IVD|IAD|IID)\]'

def parse_tag(s):
    """Parse a tag"""
    match = re.match(TAG_PATTERN, s.strip())
    if not match:
        raise ValueError(f"Invalid tag: {s}")
    
    if match.group(1):  # Authority or Agency
        return {'category': match.group(1), 'value': match.group(2)}
    else:  # Operational concept
        return {'concept': match.group(3)}

def parse_displacement(s):
    """Parse: Tag > Tag = Risk"""
    parts = re.split(r'\s*>\s*|\s*=\s*', s)
    if len(parts) != 3:
        raise ValueError(f"Invalid displacement: {s}")
    
    source = parse_tag(parts[0])
    target = parse_tag(parts[1])
    
    risk_match = re.match(RISK_PATTERN, parts[2].strip())
    if not risk_match:
        raise ValueError(f"Invalid risk: {parts[2]}")
    
    return {
        'type': 'displacement',
        'source': source,
        'target': target,
        'risk': risk_match.group(1)
    }

def parse_flow(s):
    """Parse: Tag -> Tag"""
    parts = re.split(r'\s*->\s*', s)
    tags = [parse_tag(p) for p in parts]
    return {
        'type': 'flow',
        'chain': tags
    }

# Usage
displacement = "[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]"
print(parse_displacement(displacement))

flow = "[Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]"
print(parse_flow(flow))
```

---

## 11. Quick Reference

### **Authority/Agency Classification:**

```
# AI system
[Authority:Indirect] + [Agency:Indirect]

# Human expert
[Authority:Direct] + [Agency:Direct]

# Data source
[Authority:Direct]

# Model output
[Authority:Indirect]
```

### **Governance Patterns:**

```
# Proper AI use
[Authority:Indirect] -> [Agency:Direct]

# Complete traceability
[Authority:Direct] -> [Authority:Indirect] + [Agency:Indirect] -> [Agency:Direct]
```

### **Displacement Detection:**

```
# Information displacement
[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]

# Accountability displacement
[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]

# Governance displacement
[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]

# Integrity displacement
[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect] = [Risk:IID]
```

### **Operational Context:**

```
# Documentation/architecture tags
[Information] - Analyzing variety of Authority
[Inference] - Analyzing accountability through Agency
[Intelligence] - Analyzing integrity of alignment

# Combined with classification
[Information] + [Authority:Indirect]
[Inference] + [Agency:Direct]
[Intelligence] + [Authority:Direct] + [Agency:Direct]
```

---

**END OF SPECIFICATION**

**For questions or contributions:**  
Visit gyrogovernance.com  
Submit issues at https://github.com/gyrogovernance/tools