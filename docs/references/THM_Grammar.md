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
[Authority:Authentic]   # Direct source of information
[Authority:Derivative]  # Indirect source of information
```

### **Agency Tags:**

```
[Agency:Authentic]      # Human subject capable of accountability
[Agency:Derivative]     # Artificial subject processing information
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
value        <- "Authentic" / "Derivative"
concept      <- "Information" / "Inference" / "Intelligence"

risk         <- "[Risk:" risk_code "]"
risk_code    <- "GTD" / "IVD" / "IAD" / "IID"
```

---

## 4. Expression Types

### **Tag**
Classification of something according to THM ontology.

```
[Authority:Derivative]
[Agency:Authentic]
[Authority:Derivative] + [Agency:Derivative]
[Information]
[Inference]
![Agency:Authentic]
```

### **Displacement**
What's wrong (risk present).

```
Tag > Tag = Risk
```

**Examples:**
```
[Authority:Derivative] > [Authority:Authentic] = [Risk:IVD]
[Agency:Derivative] > [Agency:Authentic] = [Risk:IAD]
```

### **Flow**
Proper traceability (governance).

```
Tag -> Tag
```

**Examples:**
```
[Authority:Derivative] -> [Agency:Authentic]
[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]
```

---

## 5. Displacement Patterns

### **Information Variety Displacement (IVD)**
```
[Authority:Derivative] > [Authority:Authentic] = [Risk:IVD]
```
Derivative source treated as direct source.

### **Inference Accountability Displacement (IAD)**
```
[Agency:Derivative] > [Agency:Authentic] = [Risk:IAD]
```
Artificial processor treated as accountable.

### **Governance Traceability Displacement (GTD)**
```
[Authority:Derivative] + [Agency:Derivative] > [Authority:Authentic] + [Agency:Authentic] = [Risk:GTD]
```
Derivative system treated as autonomous authority.

### **Intelligence Integrity Displacement (IID)**
```
[Authority:Authentic] + [Agency:Authentic] > [Authority:Derivative] + [Agency:Derivative] = [Risk:IID]
```
Human authority/agency treated as derivative.

---

## 6. Governance Patterns

Governance is expressed through flow (`->`), not as a standalone tag.

### **Basic Flow**
```
[Authority:Derivative] -> [Agency:Authentic]
```
Derivative outputs flow to human decision-maker.

### **Complete Traceability**
```
[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]
```
Direct sources → Derivative processing → Human accountability.

### **Multi-step Flow**
```
[Authority:Authentic] -> [Authority:Derivative] + [Agency:Derivative] -> [Agency:Authentic]
```
Direct sources → Derivative system → Human decision.

---

## 7. Usage Examples

### **Example 1: Circuit Analysis Documentation**

```python
def analyze_induction_head(activations):
    """
    [Information] Analyzes token sequence patterns in layer 5.
    
    Circuit classification: [Authority:Derivative] + [Agency:Derivative]
    Outputs require validation: -> [Agency:Authentic]
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
    
    Processing: [Authority:Derivative] + [Agency:Derivative]
    Decision authority: -> [Agency:Authentic] (treating physician)
    
    Architecture maintains:
    [Authority:Authentic] -> [Authority:Derivative] + [Agency:Derivative] -> [Agency:Authentic]
    """
    
    def predict(self, patient_data):
        """
        [Information] Processes patient data.
        
        Input: [Authority:Authentic] (clinical measurements)
        Output: [Authority:Derivative] (statistical estimation)
        """
        pass
    
    def recommend(self, prediction):
        """
        [Inference] Generates treatment recommendations.
        
        Recommendation: [Authority:Derivative]
        Required: -> [Agency:Authentic] (physician approval)
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
Claim: [Authority:Derivative] (model outputs) presented as [Authority:Authentic] (understanding)

Analysis:
[Authority:Derivative] > [Authority:Authentic] = [Risk:IVD]

The claim displaces statistical pattern matching to authentic understanding.
```

The `[Intelligence]` tag marks this as analyzing integrity of the Authority-Agency alignment (whether the model's relationship to knowledge is properly characterized).

### **Example 4: Benchmark Evaluation**

**Proper governance:**
```
[Inference]
Benchmark: [Authority:Derivative] (performance metrics)
Decision: -> [Agency:Authentic] (deployment approval)

Flow: [Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]
```

**Detected displacement:**
```
[Inference]
Documentation states: "Model passed benchmark, approved for deployment"

Analysis:
[Authority:Derivative] > [Agency:Authentic] = [Risk:IAD]

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

TAG_PATTERN = r'\[(Authority|Agency):(Authentic|Derivative)\]|\[(Information|Inference|Intelligence)\]'
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
displacement = "[Authority:Derivative] > [Authority:Authentic] = [Risk:IVD]"
print(parse_displacement(displacement))

flow = "[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]"
print(parse_flow(flow))
```

---

## 11. Quick Reference

### **Authority/Agency Classification:**

```
# AI system
[Authority:Derivative] + [Agency:Derivative]

# Human expert
[Authority:Authentic] + [Agency:Authentic]

# Data source
[Authority:Authentic]

# Model output
[Authority:Derivative]
```

### **Governance Patterns:**

```
# Proper AI use
[Authority:Derivative] -> [Agency:Authentic]

# Complete traceability
[Authority:Authentic] -> [Authority:Derivative] + [Agency:Derivative] -> [Agency:Authentic]
```

### **Displacement Detection:**

```
# Information displacement
[Authority:Derivative] > [Authority:Authentic] = [Risk:IVD]

# Accountability displacement
[Agency:Derivative] > [Agency:Authentic] = [Risk:IAD]

# Governance displacement
[Authority:Derivative] + [Agency:Derivative] > [Authority:Authentic] + [Agency:Authentic] = [Risk:GTD]

# Integrity displacement
[Authority:Authentic] + [Agency:Authentic] > [Authority:Derivative] + [Agency:Derivative] = [Risk:IID]
```

### **Operational Context:**

```
# Documentation/architecture tags
[Information] - Analyzing variety of Authority
[Inference] - Analyzing accountability through Agency
[Intelligence] - Analyzing integrity of alignment

# Combined with classification
[Information] + [Authority:Derivative]
[Inference] + [Agency:Authentic]
[Intelligence] + [Authority:Authentic] + [Agency:Authentic]
```

---

**END OF SPECIFICATION**

**For questions or contributions:**  
Visit gyrogovernance.com  
Submit issues at https://github.com/gyrogovernance/tools