### Intro


Model : AdaWOrld  https://arxiv.org/pdf/2503.18938

- when a neural network learns to predict how videos change over time (without ever being told what action caused the change ) does it accidentally discover a structured, reusable "language of motion"?
- In NLP, self-supervised models learn that words like "run", "jump", "kick" form a coherent semantic cluster without being told what a verb is. The hypothesis here is that latent action models might do the same thing for “visual actions”
- 

### Background

1. **World Models**

World models are neural networks that learn to *simulate* environment dynamics. Given the current state of the world and an action, a world model predicts what happens next. This is valuable for planning (imagine futures before committing to them), for data-efficient RL, and for building general-purpose agents. The core challenge: world models need an *action interface* — a way to tell the model what was done so it can predict what comes next. Historically, this required labelled action data, which is expensive and environment-specific.

1. **Latent Action Models (LAMs)**

The key innovation that this project builds upon is learning the action interface *without* action labels. The intuition: if you look at two consecutive video frames, the difference between them is mostly caused by whatever action was taken. A LAM tries to compress that "difference" into a compact latent code — the latent action — using only the frames themselves as supervision. This makes training scalable to raw internet video.

1. AdaWorld
    - **latent action autoencoder**: a Transformer-based β-VAE that takes two consecutive frames and encodes the transition into a 32-dimensional continuous latent vector. The β hyperparameter controls the trade-off between expressiveness and context-disentanglement.
    - **autoregressive world model** built on Stable Video Diffusion that takes the latent action as a conditioning signal to predict the next frame. The key claimed property: the latent actions are *context-invariant* . AdaWorld even demonstrates simple action composition: averaging the "right" and "jump" latent vectors produces a "jump-right" action

### Review of related lit

**AdaWorld [13]**  :  provides the pretrained latent action model. Key finding =  latent actions are 32-dimensional continuous vectors whose context-invariance was demonstrated qualitatively and via action transfer benchmarks. The UMAP visualization of the paper show some clustering by action type across environments — that's the embryo of what you're extending.

**Genie [1] (Bruce et al., 2024)** is the predecessor that AdaWorld explicitly contrasts with. Genie used a discrete VQ-VAE with only 8 fixed action codes, which limited expressiveness. AdaWorld moved to continuous space — your project asks whether that continuous space has meaningful structure.

**Garrido et al. [16] (2026)** is especially relevant because it directly states the key limitation you're tackling: in the absence of a shared embodiment, latent actions tend to be "localized and camera-relative, reflecting spatial patterns of visual change rather than an embodiment independent notion of action." 

**Open Problems :** 

1.  The structure of latent action spaces is unknown
2. Embodiment-dependence / camera-relativity : 
    - checking whether same labelled actions from different Stable Retro games cluster together in the latent space
    - measuring latent action variance for the same game action across different game levels (same action, varied backgrounds)
3.  **The action-context entanglement problem** 
Even with the β-VAE bottleneck, there's a risk that the latent action encodes not just "what happened" but "what happened in this particular visual context." This is exactly why AdaWorld tunes β carefully. You can probe this by measuring latent action variance for the same game action across different game levels (same action, varied backgrounds).
4. **Interpretability and alignment with human action concepts** 
Even if latent actions cluster, do the clusters mean anything to humans? There's no guarantee that the model's "action primitives" align with human labels like "jump" or "attack." Your probing experiments (training classifiers from action labels → latent codes) directly test this.

### Research Questions

**Primary RQ:** *Does AdaWorld's latent action space exhibit semantic structure consistent with an interpretable "verb space" of atomic action primitives?*

**Secondary RQs:**

- RQ2: Are the geometric clusters of latent actions stable across visually diverse environments, or are they environment specific?
- RQ3: What are the failure modes — where does the latent action space break down or conflate semantically distinct actions?

### Methods

#### Stage 1 — Geometric Analysis of the Latent Action Space

*Data collection:*

Use Stable Retro [17] to collect (frame_t, frame_{t+1}, action_label) triples across multiple platformer environments. Having ground-truth action labels (LEFT, RIGHT, JUMP, ATTACK, etc.)  allows us to evaluate whether the latent clusters align with human interpretable categories without using labels during encoding.

*Latent action extraction:*

Run AdaWorld's pretrained latent action encoder on all frame pairs. This gives you a dataset of (z, action_label, environment_id) tuples

*Visualization:*

Apply UMAP and t-SNE to the z vectors, coloring by (a) action_label and (b) environment_id separately

*Clustering evaluation:*

Apply K-means clustering to the z space and evaluate against action labels using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). If the latent space has the structure you hypothesize, clustering should recover ground-truth action categories better than chance. Compare against a random baseline and against optical flow features (AdaWorld's own ablation baseline).

- NMI measures how much information the cluster assignment V
V provides about the ground truth class label U, normalized to account for the entropy of each. Unlike ARI (which is pair-based), NMI is information-theoretic and operates at the level of the full partition
- 

*Linear probing:*

Train a linear classifier from z → action label (held-out environments for test). High linear probe accuracy suggests that action semantics are linearly accessible in the latent space — strong evidence for the "verb space" hypothesis. Non-linear probing gives an upper bound.

*Cross-environment stability:*

For each action label present in multiple environments, compute the within-cluster variance (same label, different environments) vs. between-cluster variance (different labels). If latent actions are truly context-invariant, within-cluster variance should be low.

Failure Modes

Analogy Testing : Vector Arithmetic in Action Space

### Results