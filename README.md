# LLM-Powered Customer Reengagement Intelligence: Case Study

> Two-pipeline system combining K-means customer segmentation with Gemini-powered campaign content extraction. The insight lives in the analytical cross by matching which message characteristics (tone, theme, CTA strength) produce the highest reengagement rates for each customer segment. Built for a retail reengagement program on GCP.

**Sector:** retail / direct marketing · **Role:** contributor (~35% ownership, team of 2 DS + engineering) · **Stack:** Python · Gemini (Vertex AI SDK) · K-means · BigQuery · Cloud Composer · GCP · **Status:** delivered to production, 2025

The client owns the business metrics and internal identifiers. Features are described by concept; exact reengagement rates are expressed as ranges.

---

## Contents

1. [The problem](#1-the-problem)
2. [Why two separate pipelines](#2-why-two-separate-pipelines)
3. [Pipeline A: customer segmentation](#3-pipeline-a-customer-segmentation)
4. [Pipeline B: campaign feature extraction with Gemini](#4-pipeline-b-campaign-feature-extraction-with-gemini)
5. [Validation: Cohen's Kappa as prompt engineering loss function](#5-validation-cohens-kappa-as-prompt-engineering-loss-function)
6. [The analytical cross: where the insight lives](#6-the-analytical-cross-where-the-insight-lives)
7. [Results and segmentation](#7-results-and-segmentation)
8. [Model card: scope, assumptions and limitations](#8-model-card-scope-assumptions-and-limitations)
9. [Scoping decisions and repository contents](#9-scoping-decisions-and-repository-contents)
10. [What this case study demonstrates](#10-what-this-case-study-demonstrates)

---

## 1. The problem

A retailer running a direct marketing reengagement program faced a chronic retention problem: only 8–12% of lapsed customers return after an inactive period. Campaigns were generic, and the main lever available (message personalisation) was not being used systematically because the retailer had no structured way to understand what message characteristics had worked for which customer types in the past. They could only see which campaigns had the best performance.

Two specific gaps were identified. First, historical campaign content had never been analysed at scale beyond basic open/click metrics. Second, customer segmentation was limited to recency and purchase amount. Neither gap alone explains the low reengagement rate. Together they meant the right messages were not reaching the right segments.

The objective was to use LLMs to extract campaign semantics at scale, segment customers with structured data, and cross both to identify which message characteristics produce the highest reengagement per customer profile. The target was a relative improvement of 25–35% over the historical base rate.

## 2. Why two separate pipelines

The architectural decision that everything else depends on: customer segmentation and campaign content extraction run as independent pipelines. They share no data until the final analytical cross.

This separation is deliberate. Customer clusters must reflect actual customer behaviour, not the campaigns that happened to be sent to those customers. If campaign characteristics entered the clustering step, the segments would absorb whatever biases were already in the send history (e.g., a segment defined partly by "received emotional-tone emails" would tell us nothing about that segment's intrinsic preference, only about what the marketing team had tried). Similarly, campaign feature extraction must produce an objective characterisation of each message independent of which customers it was sent to and how they responded.

The insight emerges from the join between the two outputs.

## 3. Pipeline A: customer segmentation

**Unit of analysis:** the individual customer.  **Input:** structured data in BigQuery. No LLM processing.

### Feature groups

| Group | What it captures |
|---|---|
| RFM | Recency (days since last purchase), frequency (number of purchases), monetary (total and average amount) |
| Behavioural | Historical open rate, click rate, preferred channel |
| Category affinity | Preferred product category, number of distinct categories purchased from |
| Demographics | Age, region, gender where available |
| Lifecycle | Time as customer, number of previous lapse episodes, average lapse duration |

### StandardScaler: a non-negotiable technical requirement

K-means uses Euclidean distance to assign clusters. Features with different magnitudes directly distort that distance: a monetary variable with a range of 5 to 10,000 would dominate a recency variable with a range of 0 to 730, and clustering would largely reflect purchase volume rather than behavioural patterns. StandardScaler brings each feature to mean=0 and std=1, equalising their initial contribution to the Euclidean space. Skipping it would not produce wrong-looking results; it would produce results that look plausible but measure the wrong thing.

### Cluster selection and validation

K=4 was selected. Quantitative validation used the Silhouette Score and the Elbow Method. Qualitative validation consisted of a review with the retailer's domain experts to confirm that the four segments were interpretable and actionable by the marketing team. Mathematical coherence is necessary but not sufficient for a segmentation that feeds an operational campaign calendar. Feature importance post-clustering used a Random Forest proxy to identify which customer features best discriminate between segments.

### Cold start handling

New customers have no interaction history and cannot be assigned to a segment based on historical response patterns. Initial assignment uses category affinity and first purchase amount as proxies. Segment assignment is refined in the first monthly cycle after the customer receives their first campaign.

## 4. Pipeline B: campaign feature extraction with Gemini

**Unit of analysis:** the campaign (a complete email). **Input:** around 50 historical emails from the retailer's direct marketing archive stored in Cloud Storage.

### Extracted attributes

For each campaign, Gemini extracts four categorical attributes:

| Attribute | Categories |
|---|---|
| Tone | urgent / emotional / rational / gratitude |
| Theme | specific product / benefit / loyalty / price urgency |
| CTA strength | weak / moderate / strong |
| Personalisation | generic / semi-personalised / personalised |

### Why Gemini for this task

Extracting structured categorical attributes from free-text marketing emails at scale requires a model that can follow complex instructions and produce consistent structured outputs. The alternative was manual annotation, which would not scale across the archive and would introduce annotator inconsistency across the different stylistic variants in the historical campaigns. Also, setting up this pipeline and validating it would result in an integrated pipeline for future campaigns without the need of human intervention. 

### Authentication

The project uses the Vertex AI SDK (not a direct API key), so authentication runs through the client's GCP Service Account. Customer and campaign data does not leave the client's infrastructure.

### Temperature=0: three concrete reasons

Temperature controls how deterministic the model's token selection is. High temperature makes the model creative and varied; temperature=0 makes it always select the highest-probability token; the same input always produces the same output. Temperature=0 is mandatory for this pipeline for three independent reasons.

**1. Reproducibility.** If the pipeline re-runs (new data, corrections, quarterly refresh), already-processed emails must produce identical classifications. With non-zero temperature, the campaign catalog would shift silently between runs with no data change.

**2. Validity of Cohen's Kappa measurement.** The validation step (see §5) measures agreement between the LLM's classifications and a human expert's. If the LLM produces different classifications on each run, the measured Kappa describes the variance of a particular sampling, not a stable property of the prompt. Temperature=0 guarantees that what Kappa measures is prompt quality, not sampling noise.

**3. JSON schema integrity.** The output must always respect a defined schema (fields, types, enum of allowed values). With non-zero temperature, the model can produce an unexpected field name or a value outside the allowed enum. Temperature=0 maximises schema adherence, which the prompt further enforces with explicit JSON schema formatting and few-shot examples.

Temperature=0 is a necessary condition, not a sufficient one. It works in combination with few-shot examples in the system prompt (including edge cases, multi-tonal emails, implicit CTAs) and explicit JSON schema forcing.

## 5. Validation: Cohen's Kappa as prompt engineering loss function

The entire downstream analysis depends on Gemini extracting campaign attributes consistently with human judgment. If extraction is wrong, the campaign catalog is contaminated and the cross produces incorrect insights. A validation step was not optional. It was the mechanism that defined when the pipeline was ready for production data.

**Setup:** a sample of emails was independently classified by a domain expert using the same criteria as the system prompt. Cohen's Kappa was measured per attribute dimension (tone, theme, CTA strength, personalisation). Threshold: κ > 0.75, the conventional threshold for "substantial agreement" used to validate human annotators before production use. A complementary precision target of >85% controlled false positives in classification.

**How Kappa served as a loss function for prompt engineering:** when Kappa fell below threshold on a dimension, emails where the LLM and the human expert diverged were identified and analysed. These were typically ambiguous cases: emails with mixed tone, CTAs that were implicit rather than explicit, themes that combined two categories. These were added to the system prompt as few-shot examples, targeting the specific failure mode. Each iteration aimed to increase Kappa on the problematic cases without degrading it on others.

The loop continued until all four dimensions crossed the κ > 0.75 threshold. The final system prompt is the accumulated result of those iterations.

## 6. The analytical cross: where the insight lives

The effectiveness matrix is the output that the retailer's marketing team could act on. It joins three data sources:

1. **Historical performance at customer-campaign level.** One row per (customer, campaign) pair: opened, clicked, purchased, amount purchased.
2. **Customer segment.** Output of Pipeline A, which segment each customer belongs to.
3. **Campaign characteristics.** Output of Pipeline B, tone, theme, CTA strength, personalisation for each campaign.

Aggregating by (customer segment × campaign characteristics) and calculating reengagement rate per cell produces a matrix of what works for whom. The retailer's marketing team reads this matrix directly: for a given customer segment, which combination of tone, theme, and CTA produced the highest reengagement rate historically.

## 7. Results and segmentation

*Exact rates are client property and are not published. Ranges and relative ordering are indicative.*

Four customer segments emerged with distinct optimal message profiles. Reengagement rates under the matched-message strategy ranged from roughly 7% for the most disengaged segment (lapsed occasional customers) to roughly 18% for the highest-value segment (frequent high-value customers), the latter at approximately 2x the historical baseline. The gap between matched and unmatched messaging was widest in the two middle segments, where the wrong tone actively suppressed response.

The system does not prescribe a single message per segment. It delivers a ranked preference table per segment that the marketing team uses to guide campaign briefing, not to automate it.

## 8. Model card: scope, assumptions and limitations

### Intended use

Feed retail marketing teams with a structured view of which historical message characteristics correlate with higher reengagement rates for each customer segment, to inform future campaign design.

### Out of scope

- **Causal claims.** The effectiveness matrix identifies correlations in historical data. It does not establish that sending a "rational-tone, benefit-theme" email *caused* Champions to re-engage, correlation could reflect confounders (campaigns sent to Champions may have been more personalised for other reasons). Controlled experiment required for causal validation.
- **Automation of campaign creation.** The output is a ranked preference table, not a content generator. Human campaign teams use it as a briefing input.
- **Retailers without historical campaign data.** The pipeline requires a minimum history of emails with linked customer response records. A client with no prior campaign history cannot generate a meaningful effectiveness matrix by itself; the system becomes useful only after an initial collection period.

### Limitations

1. **50 campaigns is a descriptive sample, not a statistically sufficient sample for a supervised model.** The analysis identifies patterns; it does not have the statistical power to claim significance at a granular cell level. The documented next step is accumulating observations (target: >10K customer-campaign pairs) before training a supervised LightGBM model that replaces the cross-analysis with a direct prediction.
2. **Gemini attribute extraction is validated for the prompt, not for all possible future campaigns.** If the retailer launches campaigns with stylistic characteristics significantly outside the historical training distribution, Kappa on those new campaigns should be re-measured.
3. **K=4 was chosen with stakeholder validation at a specific point in time.** If the customer base composition changes significantly, re-validating the number of clusters and segment profiles is appropriate.

## 9. Scoping decisions and repository contents

### 9.1 Scoping decisions

Out of scope for this public write-up:

- **The exact prompt.** The structure is described in §4 (system role, output schema with four categorical attributes, few-shot slots for ambiguous cases). The final content is the accumulated result of the Kappa-driven iteration described in §5 and is specific to the campaign styles of this client.
- **Internal metrics as absolute numbers.** Reengagement rates, segment composition sizes, and historical baselines are presented as ranges or relative lift. Point estimates are client property.
- **Feature list for the customer clustering.** §3 describes the feature groups (RFM, behavioural, category affinity, demographics, lifecycle). The exact feature names and transformations are not published.
- **Operational segment labels.** The labels used here (Champions, lapsed occasional) are generic descriptors chosen for a reader outside the client's organisation. They are not the operational labels used in the client's campaign tooling.

### 9.2 What is in this repo

- `README.md`, this document.
- `snippets/kappa_validation_loop.py`, an illustrative snippet of the Kappa-driven prompt refinement loop described in §5. The snippet is not runnable against client data; it uses synthetic input to show the control-flow structure of the loop (compute Kappa per dimension, identify divergence cases, surface them as candidate few-shot examples).

## 10. What this case study demonstrates

- **Architectural separation of analysis units.** Keeping customer segmentation and campaign feature extraction as independent pipelines that connect only in the analytical cross is the decision that makes the insight trustworthy. Mixing them would have produced a system that reflects historical send biases, not genuine customer preferences.
- **Temperature=0 with three independent justifications** (reproducibility, Kappa validity, schema integrity), each derived from a concrete failure mode rather than a general preference for determinism.
- **Cohen's Kappa as the operational definition of "ready for production."** Using Kappa as a prompt engineering loss function, iterating on few-shot examples until the threshold is met, is a concrete methodology for validating LLM-based classification before it feeds a downstream analysis.
- **StandardScaler as a non-negotiable preprocessing step**, justified from the Euclidean geometry of K-means rather than from convention.
- **Lift-to-baseline framing.** Every reengagement rate is stated relative to the historical baseline, not as a standalone number. The question the model answers is "how much better than what we already had?" not "what percentage responded?"

---

Martín Terzano · [helliumlab.com](https://helliumlab.com) · [LinkedIn](https://www.linkedin.com/in/martinterzano) · martin@helliumlab.com
