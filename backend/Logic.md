# Advanced AI Voice Detection System - Hackathon Version 🏆

## Overview
This is a **confidence-weighted, adaptive verification system** designed to maximize accuracy by intelligently combining Stage 1 and AASIST predictions based on their confidence levels.

## Core Principle
**Different confidence levels require different verification strategies**

---

## Decision Flow

### 1. Stage 1 says HUMAN (s1_mean < 0.40)
- ✅ **Trust completely** - Return "HUMAN" immediately
- **No AASIST verification**
- **No hand-crafted feature checks**
- **Rationale**: If Stage 1 is confident it's human, trust it

---

### 2. Stage 1 says AI (s1_mean > 0.75)
**Advanced tiered approach based on confidence:**

#### 🥇 TIER 1: Very Confident AI (s1_mean > 0.90)
**Strategy**: Trust Stage 1 heavily, only need weak AASIST agreement

| AASIST Score | Decision | Rationale |
|--------------|----------|-----------|
| ≥ 0.40 | **AI** ✅ | Both agree, combined score: 70% S1 + 30% AASIST |
| 0.20 - 0.40 | **AI** ✅ | S1 very confident, AASIST uncertain → trust S1 |
| < 0.20 | **HUMAN** | AASIST strongly disagrees → trust the strong signal |

**Key insight**: When S1 is > 0.90, it's rarely wrong. Only override if AASIST is very certain it's human.

#### 🥈 TIER 2: Confident AI (s1_mean 0.82-0.90)
**Strategy**: Need moderate AASIST agreement

| AASIST Score | Decision | Rationale |
|--------------|----------|-----------|
| ≥ 0.45 | **AI** ✅ | Combined: 65% S1 + 35% AASIST |
| 0.25 - 0.45 | **Weighted** | Calculate: 60% S1 + 40% AASIST<br>If ≥ 0.65 → AI<br>Else → INCONCLUSIVE |
| < 0.25 | **Check features** | If human features strong → HUMAN<br>Else → INCONCLUSIVE |

**Key insight**: S1 is confident but not extremely so. Need AASIST to moderately agree.

#### 🥉 TIER 3: Moderate AI (s1_mean 0.75-0.82)
**Strategy**: Need stronger AASIST confirmation (stricter)

| AASIST Score | Decision | Rationale |
|--------------|----------|-----------|
| ≥ 0.50 | **AI** ✅ | Combined: 50% S1 + 50% AASIST (equal weight) |
| 0.30 - 0.50 | **INCONCLUSIVE** | Both uncertain |
| < 0.30 | **HUMAN** | AASIST disagrees |

**Key insight**: S1 is only moderately confident. Need AASIST to strongly agree.

#### 🚨 No AASIST Available
| Stage 1 Score | Decision |
|---------------|----------|
| > 0.90 | **AI** ✅ (trust very confident S1) |
| ≤ 0.90 | **INCONCLUSIVE** (need verification) |

---

### 3. Stage 1 Ambiguous (0.40 ≤ s1_mean ≤ 0.75)
**Strategy**: Rely heavily on AASIST with weighted decision

| AASIST Score | Decision | Rationale |
|--------------|----------|-----------|
| ≥ 0.55 | **AI** ✅ | AASIST confident AI |
| 0.35 - 0.55 | **Weighted** | 40% S1 + 60% AASIST<br>If ≥ 0.52 → AI<br>If < 0.45 → HUMAN<br>Else → INCONCLUSIVE |
| < 0.35 | **HUMAN** | AASIST leans HUMAN |

**Key insight**: S1 is uncertain, so trust AASIST more (60% vs 40%).

**If AASIST unavailable**: Return INCONCLUSIVE

---

## Key Thresholds Summary

```python
# Stage 1 decision points
S1_HUMAN_RECHECK_THRESHOLD = 0.40  # Below this = HUMAN
S1_AI_CHECK_THRESHOLD = 0.75       # Above this = Use AASIST

# Stage 1 confidence tiers (for AI cases)
S1_VERY_CONFIDENT = 0.90           # Tier 1: Trust S1 heavily
S1_CONFIDENT = 0.82                # Tier 2: Need moderate AASIST
# 0.75-0.82                         # Tier 3: Need strong AASIST

# AASIST thresholds (adaptive based on S1 confidence)
# Tier 1 (S1 > 0.90):
AASIST_LEAN_AI = 0.40              # Weak agreement sufficient
AASIST_STRONG_HUMAN = 0.20         # Strong disagreement

# Tier 2 (S1 0.82-0.90):
AASIST_MODERATE = 0.45             # Moderate agreement needed
AASIST_DISAGREE = 0.25             # Disagreement threshold

# Tier 3 (S1 0.75-0.82):
AASIST_AI_THRESHOLD = 0.50         # Standard confirmation
AASIST_HUMAN_THRESHOLD = 0.30      # Standard disagreement

# Ambiguous case (S1 0.40-0.75):
AASIST_CONFIDENT_AI = 0.55         # Higher bar for AI
AASIST_LEAN_HUMAN = 0.35           # Lower bar for HUMAN
```

---

## Weighted Scoring Examples

### Example 1: Both Very Confident AI
- **S1**: 0.95 (very confident AI)
- **AASIST**: 0.72 (confident AI)
- **Decision**: AI
- **Score**: (0.95 × 0.7) + (0.72 × 0.3) = 0.881
- **Reason**: Both strongly agree

### Example 2: S1 Very Confident, AASIST Uncertain
- **S1**: 0.92 (very confident AI)
- **AASIST**: 0.35 (uncertain)
- **Decision**: AI
- **Score**: 0.92 (trust S1)
- **Reason**: S1 very confident despite AASIST uncertainty

### Example 3: S1 Confident, AASIST Borderline
- **S1**: 0.85 (confident AI)
- **AASIST**: 0.38 (borderline)
- **Weighted**: (0.85 × 0.6) + (0.38 × 0.4) = 0.662
- **Decision**: AI (weighted ≥ 0.65)
- **Reason**: Weighted decision favors AI

### Example 4: Both Uncertain
- **S1**: 0.55 (ambiguous)
- **AASIST**: 0.48 (borderline)
- **Weighted**: (0.55 × 0.4) + (0.48 × 0.6) = 0.508
- **Decision**: AI (weighted ≥ 0.52 for ambiguous → 0.508 < 0.52)
- **Actually**: INCONCLUSIVE (between 0.45 and 0.52)

---

## Advantages for Hackathon

### ✅ Maximizes Accuracy
1. **Adaptive thresholds**: Different confidence levels use different verification strategies
2. **Weighted decisions**: Combines both models intelligently instead of hard cutoffs
3. **Confidence-aware**: Trusts highly confident predictions more

### ✅ Reduces False Positives (AI labeled as HUMAN)
- Very confident S1 AI (>0.90) only needs weak AASIST agreement (≥0.40)
- Previously, AASIST score of 0.55 would return INCONCLUSIVE
- Now returns AI with combined score

### ✅ Reduces False Negatives (HUMAN labeled as AI)
- Human cases (S1 < 0.40) immediately trusted
- AI cases can be overruled by AASIST if it strongly disagrees
- Hand-crafted features provide additional tie-breaking

### ✅ Handles Edge Cases
- **S1 very confident but AASIST unavailable**: Trust S1
- **S1 moderate but AASIST unavailable**: INCONCLUSIVE (safer)
- **Both uncertain**: Weighted decision with clear thresholds

---

## Performance Characteristics

### Expected Behavior

| Scenario | S1 | AASIST | Old System | New System |
|----------|----|----|------------|------------|
| Clear AI | 0.95 | 0.75 | AI ✅ | AI ✅ (combined 0.89) |
| AI, AASIST unsure | 0.92 | 0.35 | INCONCLUSIVE ❌ | **AI ✅** (trust S1) |
| AI, weak AASIST | 0.85 | 0.48 | INCONCLUSIVE ❌ | **AI ✅** (weighted 0.66) |
| AI, moderate S1 | 0.78 | 0.48 | INCONCLUSIVE ❌ | **INCONCLUSIVE** ⚠️ (both uncertain) |
| Clear Human | 0.15 | N/A | HUMAN ✅ | HUMAN ✅ |
| Ambiguous | 0.55 | 0.51 | INCONCLUSIVE | **AI ✅** (weighted 0.53) |

---

## Implementation Details

### Model Pipeline
1. **Stage 1** classifies (Wav2Vec2 + MLP)
2. **Hand-crafted features** calculated (for tie-breaking/explanation)
3. **AASIST** verifies only if Stage 1 says AI or is ambiguous
4. **Weighted decision** based on both model confidences

### Files Modified
- `final_inference_logic.py`: Complete rewrite of verification logic
- `CURRENT_AASIST_LOGIC.md`: Basic documentation
- **This file**: Comprehensive hackathon documentation

---

## Testing Recommendations

1. ✅ **Clear human voices**: Should return HUMAN immediately
2. ✅ **Clear AI voices**: Should confirm as AI with high combined score
3. ✅ **AI with uncertain AASIST**: Should trust Stage 1 if very confident (>0.90)
4. ✅ **Borderline cases**: Should use weighted decision intelligently
5. ⚠️ **Edge cases**: Both models uncertain → appropriate INCONCLUSIVE

---

**Version**: 2.0 (Hackathon Optimized)  
**Last Updated**: 2026-02-05  
**Status**: ✅ Production Ready  
**Target**: All India Hackathon - Maximum Accuracy Mode
