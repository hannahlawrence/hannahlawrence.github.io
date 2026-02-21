---
layout: distill
title: "Square Peg, Round Hole: Plugging Non-Sequential Data into Sequential Language Models"
description: "Autoregressive (AR) models are central to modern generative AI systems, yet their sequential inductive bias clashes with modalities that lack an obvious ordering, such as images, graphs, and point clouds. Despite this mismatch, AR models are widely used beyond language, owing to their scalability and controllability. This post highlights the growing set of techniques that make non-sequential data amenable to autoregressive modeling. There are two broad directions: approaches that choose or optimize a generation order for a fixed tokenization, and approaches that redesign the tokenization itself to simplify each next-token prediction step. We emphasize the tradeoffs these methods face, particularly between compression and autoregressive &quot;modelability&quot;. By drawing these connections, we aim to motivate future work on tokenizations tailored to the needs of autoregressive models for arbitrary datatypes."
date: 2026-04-27
future: true
htmlwidgets: true

authors:
  - name: Hannah Lawrence
    url: "https://hannahlawrence.github.io"
    affiliations:
      name: MIT

bibliography: references.bib

toc:
  - name: Introduction
  - name: "Primer: Autoregressive Modeling"
    subsections:
      - name: What exactly are tokens?
      - name: Tokenization-free methods
      - name: Advantages
  - name: Non-sequential data
    subsections:
      - name: Examples
  - name: Can we just use non-sequential models?
  - name: "Example: Images"
  - name: Aligning sequential models and non-sequential data
    subsections:
      - name: "Model-level alignment"
        subsections:
          - name: Marginalizing over orderings
          - name: Heuristically choosing the order
          - name: Learning the order
      - name: "Tokenization-level alignment"
        subsections:
          - name: Heuristically encouraging AR modelability
          - name: Autoregressive priors
  - name: "Conclusion"
  - name: Appendix
  - name: Footnotes

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---



<div style="text-align: center; margin: 2rem 0;">
  <div style="display: inline-block; max-width: 300px; width: 100%;">
    {% include figure.liquid
        path="blog/autoregressive-tokenization/images/square_peg.png"
        class="img-fluid rounded z-depth-1"
    %}
  </div>
</div>

# Introduction
Autoregressive sequence models sit at the center of modern generative AI, excelling in settings like natural language where data arrive in a well-defined sequence. However, many important modalities do not immediately offer such a linear structure. Images, graphs, point clouds, and sets lack an intrinsic notion of "the next token."


{% include figure.liquid path="blog/autoregressive-tokenization/images/intro.png" class="img-fluid" %}
<div class="caption" style="text-align: center;">
    How can one apply sequential models to non-sequential (e.g. non-language) data? (Note that we will use "autoregressive model" and "sequential model" interchangeably.)
</div>

Despite this apparent mismatch between modeling assumption and data structure, autoregressive (AR) models have been repeatedly applied in such non-lingual settings <d-cite key="Antunes2024,10.1609/aaai.v39i24.34804,sun2024autoregressivemodelbeatsdiffusion"></d-cite>. There are good reasons: AR models offer variable-length generation, precise likelihoods, flexible conditioning, and step-by-step controllability <d-cite key="wang2024diverse,chen2024diffusion"></d-cite>. Moreover, from a practical perspective, autoregressive models have been engineered and scaled to perfection, with well-established scaling laws, training recipes, and ready-to-use open source libraries.

This blog post explores the emerging landscape of techniques for turning non-sequential data into discrete 1D sequences, which autoregressive models can effectively process. It is intended for a diverse audience, including anyone who wishes to design machine learning systems for non-sequential data (images, molecules, point clouds, etc).

We start with a primer on autoregressive modeling, including tokenization and positional encodings. Readers familiar with these concepts already should skip ahead to the following [section](#what-exactly-are-tokens), which defines "non-sequential data". We then categorize recent research into two distinct kinds of approaches: **model-level methods**, which optimize the generation order for a fixed set of tokens, and **tokenization-level methods**, which redesign the discrete input representation itself to align with a sequential prior. In the case of tokenization-level methods, we highlight the inherent tradeoff between compressibility and modelability. Although these methods often originate in different communities and target different modalities, they are instances of the same underlying challenge. Our aim is to draw out these connections, map the shared structure across approaches, and sketch a broader landscape of possibilities for modeling non-sequential data with sequential architectures.

# Primer: Autoregressive Modeling

From the early days of recursive neural networks to the current transformer revolution, **Autoregressive Models (ARMs)** have emerged as a central paradigm for sequence-based generative modeling. By treating data as a series of discrete tokens (drawn from some finite vocabulary) and modeling their joint distribution through next-token prediction, machine learning systems achieve strong performance on tasks ranging from fluent text generation to complex program synthesis <d-cite key="chen2021evaluatinglargelanguagemodels,openai2024gpt4technicalreport"></d-cite>.

A central assumption behind ARMs is that data can be meaningfully factorized into a sequence of tokens,
$$ x = (x_1,\dots,x_n)$$
where each token depends on those who came before it. Although **any** sequence of data can be factorized as

$$ p(x_1,\dots,x_n)=p(x_1)p(x_2\mid x_1)p(x_3 \mid x_1,x_2)\dots $$

via the chain rule, by "meaningfully" we refer to how easy it is to model each of the individual factors $$p(x_i \mid x_1,\dots,x_{i-1})$$  — more on this in the next section.

Under this factorization, the model is trained to predict the next token $x_i$ given the context of preceding tokens $x_1,\dots,x_{i-1}$. In transformers, this is implemented via causal masking, where the self-attention mechanism prevents any position from attending to "future" tokens.

<div style="text-align: center; margin: 2rem 0;">
  <div style="display: inline-block; max-width: 450px; width: 100%;">
    {% include figure.liquid
        path="blog/autoregressive-tokenization/images/cropped_CLM.gif"
        class="img-fluid rounded z-depth-1"
    %}
    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
      Autoregressive models predict next-token distributions over tokens based only on the preceding tokens.
    </div>
  </div>
</div>

## What exactly are tokens?
So, an autoregressive model operates on a sequence of discrete tokens representing a piece of data. But what exactly are tokens, and how are they computed? At first, tokens might seem unnecessary. For example, one could simply input the raw byte sequence (e.g. UTF-8 values for text) into an autoregressive model. However, byte sequences can become very long, making long-range dependencies harder to learn and obscuring meaningful linguistic structure that the model could otherwise exploit. As a result, byte-level models often require more computation and struggle to match the efficiency and performance of systems that use more semantically informed units <d-cite key="10.1609/aaai.v33i01.33013159"></d-cite>.

This motivated the use of **tokenization**: the process of mapping raw data into a sequence of discrete symbols drawn from a finite vocabulary[^soft]. For text data, the most commonly used tokenization schemes such as Byte-Pair Encoding (BPE) <d-cite key="sennrich-etal-2016-neural"></d-cite>, WordPiece <d-cite key="devlin-etal-2019-bert"></d-cite>, or unigram tokenization segment strings into subword units:

* "tokenization" → ['token', 'ization']

* "codebook" → ['code', 'book']

Subword methods strike a balance between vocabulary size and sequence length, whereas byte-level tokenization uses a near-minimal vocabulary that results in no information loss but produces considerably longer sequences. These choices reflect a fundamental tension in tokenizer design: **a tokenizer optimized purely for compression (i.e. the best reconstruction per bit budget) is not necessarily the one that is easiest for a generative model to predict** (as evidenced by e.g. <d-cite key="lester2024training"></d-cite>, who demonstrate that a naive compression-based tokenizer works poorly for language modeling). Later sections will revisit this reconstruction-generation tradeoff from multiple angles.

Each token is moreover associated with a positional encoding, which encodes that token's position in the sequence and breaks the native permutational invariance of the attention mechanism. However, even without positional encodings, the causal attention mask of autoregressive models still forces token generation to occur in a specific order. Thus, simply removing positional encodings does not fundamentally change the sequential nature of causal attention, as works like NoPos <d-cite key="haviv-etal-2022-transformer"></d-cite> have demonstrated.

## Tokenization-free methods

A few recent "tokenization-free" approaches have moved away from the tokenization paradigm, which suffers from various idiosyncrasies and challenges for multilingual data <d-cite key="neitemeier2025hierarchical"></d-cite>. Instead of committing to a predefined vocabulary or a fixed sequence structure, approaches such as the Byte Latent Transformer <d-cite key="pagnoni-etal-2025-byte"></d-cite> and H-Net <d-cite key="hwang2025dynamicchunkingendtoendhierarchical"></d-cite> let the representational units evolve during generation. If the model can decide how to construct these building blocks as it trains, then the tokenization becomes an emergent property of the model's inference dynamics, rather than something constructed ahead of time.
While promising as methods for transcending hand-designed, modality-specific tokenizations, both of these models **remain autoregressive**. In other words, they both work with fixed sequences of bytes. Thus, although we will focus on the more widespread tokenization paradigm in the rest of this blog post, the mismatch between sequential models and non-sequential data prevails for tokenization-free methods, too.

## Advantages

Autoregressive models have several useful properties that make them appealing across a wide range of generative settings <d-cite key="chen2024diffusion"></d-cite>:
* **Variable-length generation**: The model can decide dynamically when to stop generating, for example by emitting an end-of-sequence symbol. This is especially important in settings where the desired output length is unknown or input-dependent.
* **Flexible conditioning**: ARMs allow conditioning on a prefix of any length, which enables a broad class of conditional generation and editing tasks.
* **Efficient sampling**: In large transformers, once a token has been generated, its key-value representations can be cached and reused for a dramatic speed-up at inference time.
* **Compatibility with search and planning algorithms**: Because AR models assign a decomposable likelihood to each partial sequence, they can be combined with search procedures that explore multiple continuations in parallel (e.g., beam search).
* **Online feedback control**: Since generation proceeds step-by-step, external signals can intervene during generation. Intermediate states can be adjusted to guide the next token, enabling closed-loop interaction and real-time control <d-cite key="Hafner2020Dream"></d-cite>.

However, these advantages come at the cost of a rigid dependency structure: the model must commit to a specific, one-token-at-a-time generation order.

# Non-sequential data

We've alluded to the idea that autoregressive models rely on a *meaningful* factorization of the data into a sequence, but that certain data is "non-sequential". What does this mean exactly? Let's start with some examples. Spoken and written language are clearly "sequential" in a meaningful way: they are both generated in, and meant to be consumed in, a certain temporal sequence. But if someone asked you to order the pixels from an image into a sequence, what would you choose? Raster order? Top-to-bottom, or bottom-to-top? How about the atoms in a molecule?

Perhaps you would answer that it depends on what you want to *use* the sequence for. Otherwise, how can you choose between many seemingly equivalent orderings? Images, molecular graphs, and 3D point clouds are defined by spatial relationships and symmetries, but there is no single, canonical ordering of elements on which all readers of this post would agree. To distinguish between orderings, we require the notion of **modelability for autoregressive models**.

At a high-level, modelability is a general and ubiquitous idea in representation learning: simply put, some representations are easier for models to learn from than others. This perspective echoes e.g. Xu et al.'s <d-cite key="Xu2020A"></d-cite> notion of usable information, which highlights that two representations can encode exactly the same information yet differ dramatically in how easy they are for a model to approximate <d-cite key="dieleman2025latents"></d-cite>. A similar view appears in the rate-distortion-usefulness tradeoff[^tradeoff] of Tschannen et al. <d-cite key="tschannen2018recentadvancesautoencoderbasedrepresentation"></d-cite>, where the "usefulness" of a representation depends not only on information content, but also on how that information is organized.

We focus here on the specific notion of modelability for autoregressive models. The data representation is now not a single vector per datapoint, but a sequence of discrete tokens $x_1,\dots,x_n$ per datapoint. Since autoregressive models factor the data distribution as
$\prod_{i=1}^n p(x_i \mid x_{<i})$,
 modelability asks: are the induced conditional distributions $p(x_i \mid x_{<i})$ **easily learnable by your model class**? Formally, we can write this as the expected binary cross-entropy (BCE) loss (denoted by $\ell(\text{distribution}, \text{true label})$) of the best next-token prediction model $p_{\theta^*}$ from the model class:

 $$
 \mathbb{E}_{x_1,\dots,x_n} \ell \left(p_\theta(\cdot \mid x_{< i}), x_i \right)
 $$

Here, by the "best" model we mean the model produced by a training procedure (usually, optimizing for the same BCE loss) over a finite training set.

In words, the autoregressive modelability of a tokenization is simply the test perplexity of the best next-token prediction model. Language (under any standard tokenizer) is highly modelable because next-token models do a good job at, well, predicting the next tokens. Note that modelability is a property of a specific tokenization of a data distribution, not the modality itself. For any data distribution and model class, we can ask what tokenization yields the optimal modelability score (the equation above). Thus, this notion of modelability implicitly depends on the inductive biases and computational limitations (e.g., finite context length and recency bias) of the model class.[^tokenization]

In this sense, the dichotomy of "sequential" and "non-sequential" is overly simplistic, since one can arbitrarily pick a sequence for any input data. What we really mean is, based on domain knowledge or just common sense, is there an **obviously modelable** sequence? If not, we call the modality "non-sequential", and assert that more complex methods are needed to identify a modelable tokenization (more on this later).

## Examples

The difference in modelability between different tokens orders is especially clear in domains where different prediction orders induce subproblems of highly varying difficulty. For example, consider training a model to solve Sudoku puzzles. At a given current state, some cells might be nearly forced, while others are highly ambiguous -- so, the difficulty of the prediction subproblem depends strongly on which cell is predicted first. As explored by Kim and Shah et al. <d-cite key="kim2025train"></d-cite>, changing the prediction order of the unfilled Sudoku tiles can shift the model from easy, highly-constrained cases to much harder, underdetermined ones.

<div style="text-align: center; margin: 2rem 0;">
  <div style="display: inline-block; max-width: 450px; width: 100%;">
    {% include figure.liquid
        path="blog/autoregressive-tokenization/images/sudoku.png"
        class="img-fluid rounded z-depth-1"
    %}
    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
      Example of a 4x4 Sudoku board where filling in a highly constrained square first forces the remaining moves, while starting elsewhere would be more ambiguous. Figure taken from <d-cite key="woojrSudokuKids"></d-cite>.
    </div>
  </div>
</div>

A similar example appears in arithmetic tasks, where models have been observed to perform better when generating blocks of digits right-to-left than left-to-right, perhaps reflecting how carries propagate in the computation <d-cite key="singh2024tokenizationcountsimpacttokenization,lee2024digitstodecisions"></d-cite>. Across these settings, a consistent pattern emerges: prediction orders that better align with a task's underlying structure tend to be more effective.

## Caution: semantic meaning of data ordering
In this blogpost, we talk a lot about permuting data tokens. However, naively permuting data tokens clearly doesn't make sense for domains like language or vision: the sentence "Work is more important than family" has the opposite meaning from its permutation, "Family is more important than work"[^poem]. Similarly, permuting the pixels of an image can create an entirely new image. In contrast, permuting the points in a point cloud or nodes in a graph preserves the underlying object, and therefore doesn't lose any information. (It's possible to formalize this using the concept of equivalence classes, but we focus on modelability, since it is the most relevant concept for autoregressive modeling.) Thus, when we talk about permuting data tokens, what we really mean is changing the generation order -- while retaining the information describing the input object itself.

Applying standard language modeling objectives to non-sequential data requires us to force a square peg into a round hole: we must linearize the intrinsic geometry of the data into a flat sequence. This raises a challenge: **how should sequential models operate when there is no intrinsic order to exploit?**

# Can we just use non-sequential models?
Given that many modalities lack a natural left-to-right structure, one might reasonably ask: why use autoregressive models at all? Indeed, some of the most popular alternative frameworks -- like generative masked language models and diffusion models -- avoid one-token-at-a-time prediction, and operate on entire sequences at once. A brief overview of these approaches is provided in the [Appendix](#appendix).

However, compared to autoregressive models, these non-sequential architectures sacrifice several desirable properties discussed [earlier](#primer-autoregressive-modeling), including variable-length outputs, efficient sampling with the KV cache, and step-wise guidance. Thus, there remain many settings in which applying autoregressive methods is still desirable. We should note that there is a growing line of work aiming to combine the strengths of autoregressive and diffusion models <d-cite key="hoogeboom2022autoregressive,chen2024diffusion,arriola2025block"></d-cite>. We set these aside in this post, as our focus is specifically on autoregressive models.

A well-studied domain where the lack of an inherent ordering becomes especially salient is vision. Images do not come with a built-in sequence structure, yet significant effort has been devoted to making them autoregressively modelable. We start with this domain as a case study, before diving into more recent trends of tokenization-model alignment.

# Example: Images
Images have no inherent traversal order, yet to apply autoregressive models, we must linearize them into a 1D sequence. The earliest attempts, such as PixelRNN <d-cite key="10.5555/3045390.3045575"></d-cite>, flattened the image into a sequence of raw pixels and predicted them one by one in raster scan order (top-to-bottom, left-to-right). While these models achieved strong likelihood scores, they struggled to generate high-quality samples compared to diffusion models <d-cite key="theis2016noteevaluationgenerativemodels"></d-cite>. One limiting factor of pixel-level flattening was that it would result in very long sequences that are difficult to model (e.g., a 256 x 256 image results in a sequence of length 65,536). In fact, this is the same reason that tokenization arose for language <d-cite key="sennrich-etal-2016-neural"></d-cite>!

To solve the sequence length problem, the fundamental unit of computation shifted from pixels to patches. This strategy was standardized by Vision Transformers (ViTs) <d-cite key="dosovitskiy2021an"></d-cite>: divide the image into fixed-size squares (e.g., 16 x 16), embed each patch as a token with positional encodings, and arrange them in a sequence. Crucially, ViTs retained the raster scan order, as shown at the bottom of the following Figure from Dosovitsky et al. <d-cite key="dosovitskiy2021an"></d-cite>.

<div style="text-align: center; margin: 2rem 0;">
  <div style="display: inline-block; width: 100%;">
    {% include figure.liquid
        path="blog/autoregressive-tokenization/images/vit.png"
        class="img-fluid rounded z-depth-1"
    %}
    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
      The Vision Transformer architecture, where the input image is converted into a sequence by flattering the patch grid according to a raster scan order.
    </div>
  </div>
</div>

The modern paradigm of autoregressive models for images adopts this patch-based approach via the following two-stage architecture:

1. **Stage 1 - Tokenization**: Models first compress the image into a codebook drawn from a learned visual dictionary, e.g. using a VQGAN <d-cite key="esser2020taming"></d-cite> or VQ-VAE <d-cite key="vandenOord2017"></d-cite>. The tokenizer is trained once, typically with reconstruction or perceptual losses, and then frozen.

2. **Stage 2 - Modeling**: A separate autoregressive (AR) Transformer is trained on the learned tokens in raster order.  By offloading low-level reconstruction to the tokenizer, the AR model can devote its computational capacity to modeling global structure and long-range interactions.

<div style="text-align: center; margin: 2rem 0;">
  <div style="display: inline-block; width: 100%;">
    {% include figure.liquid
        path="blog/autoregressive-tokenization/images/two_stage.png"
        class="img-fluid rounded z-depth-1"
    %}
    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
      Autoregressive image modeling tends to follow a two-step approach. In the first stage, discrete image tokens are trained using a reconstruction loss. In the second stage, the now-fixed tokens are fed into a transformer for generation.
    </div>
  </div>
</div>

Although VQ-VAE and VQGAN tokenizers learn a visual "vocabulary", they do not remove the need for a fixed ordering, as the tokens are still ordered for input to the transformer. Thus, the **inherent mismatch between the prediction order and the causal structure of natural images remains**: predicting a patch from only previously raster-ordered predecessors might force the model to commit to global structural decisions (e.g., "this is a dog") based on ambiguous local evidence (e.g., a patch of fur in the top-left corner).  This is just one example of how a representation can induce conditional prediction tasks as subproblems that are poorly aligned with the modality's underlying structure. With this as motivation, we now turn to the general problem of aligning sequential models with non-sequential data.

# Aligning sequential models and non-sequential data
Earlier, we noted a basic tension: representations that preserve all the information in the data tend to make generation more difficult, while representations that simplify prediction inevitably compress or bias the underlying signal, to the detriment of generation <d-cite key="lester2024training"></d-cite>. The goal, then, is to make sequential modeling easier without giving up too much reconstruction quality.
We organize the emerging methods for navigating this tradeoff into two distinct categories. One approach keeps the tokenization, and therefore the reconstruction quality, fixed, but modifies the model so that the existing sequence becomes easier to generate (still autoregressively). The other approach keeps the model class fixed but changes the tokenization itself, aiming to find representations that are jointly good for reconstruction and sequential prediction.
In other words, the two overarching categories of approaches we identify are:

* **Model-level**: Given a fixed tokenization, adjust or learn an optimal ordering.
* **Tokenization-level**: Given a desired ordering, adjust or learn the tokenization itself so that the resulting tokens are better aligned with it.

We further subdivide these categories according to the following flowchart, which serves as a roadmap for the remainder of the post.

{% include figure.liquid path="blog/autoregressive-tokenization/images/roadmap.png" class="img-fluid" %}
<div class="caption" style="text-align: center;">
    What we term model-level vs token-level approaches for applying autoregressive models to data without clear sequential structure.
</div>


## Model-level alignment
When a modality lacks an intrinsic traversal order, the challenge is to decide (or discover) the prediction order that leads to the easiest, most structurally coherent subproblems for a model to learn. Several lines of work explore increasingly sophisticated ways of doing this.

### Marginalizing over orderings
What if we simply train the model to be robust to any sequence using data augmentation? This is precisely the method behind **Any-Order Autoregressive Models (AO-ARMs)**, where the model is trained under random orderings drawn uniformly from all permutations of the input sequence <d-cite key="wang2025learningorder"></d-cite>. Given a permutation $\sigma$ of indices $\{1,\dots,n\}$, the learned distribution factorizes as
$$p(\mathbf{x} \mid \sigma) = \Pi_{i=1}^n p(x_{\sigma_i} \mid \mathbf{x}_{\sigma_{<i}})$$
where $\sigma_{<i}$ corresponds to indices $\{1, \ldots, i-1\}$ under the permutation σ. In this formulation, the permutation can be interpreted as a latent variable that specifies which conditional subproblem the model solves at each step.
<div style="text-align: center; margin: 2rem 0;">
  <div style="display: inline-block; max-width: 450px; width: 100%;">
    {% include figure.liquid
        path="blog/autoregressive-tokenization/images/aoarm4_cropped.gif"
        class="img-fluid rounded z-depth-1"
    %}
    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
      Any-order autoregressive models predict next-token distributions over randomly selected token orderings.
    </div>
  </div>
</div>

Moreover, each permutation effectively defines a masking pattern: at step $$i$$, the model observes the variables in $\sigma_{<i}$ and treats all variables in $\sigma_{>i}$ as unobserved. Training across many permutations therefore exposes the model to a wide variety of partially observed inputs and forces it to learn conditional distributions of the form
 $$p(x_i \mid \mathbf{x}_{-i}),$$
which is the same family of conditionals targeted by masked language models. As noted by <d-cite key="kim2025train"></d-cite>, **AO-ARMs can be viewed as an autoregressive reformulation of the masked language modeling objective**, differing only in whether the observed subset is chosen by a permutation or by an explicit mask.


A related approach is **σ-GPT** <d-cite key="10.1007/978-3-031-70368-3_9"></d-cite>, which also trains on random permutations but does so with explicit positional encodings for both the true index and the permutation order. This enables generation and conditioning in any order.

### Heuristically choosing the order

AO-ARMs already introduce the view that, once tokenization is fixed, the remaining structural choice is the permutation that determines the order of prediction. The default any-order training treats this latent variable as uniformly distributed over all permutations, which forces the model to learn a wide family of conditionals. The methods4 in this section build directly on this perspective, exploring how the choice of latent permutation influences the difficulty of the induced prediction tasks and downstream performance.

Kim, Shah et al. <d-cite key="kim2025train"></d-cite> examine this sensitivity in Masked Diffusion Models (MDMs) trained under the same uniform distribution over permutations used in AO-ARMs. Instead of sampling tokens in a random or left-to-right order, they propose **adaptive MDM inference**, greedily selecting the position with the lowest predictive entropy. This "most confident first" rule is shown to dramatically improve performance on tasks such as Sudoku by allowing the model to avoid the hard, ambiguous subproblems. Notably, when this learned ordering from the MDM model is included in the training of an ARM, it also improves the performance compared to an ARM trained on the default (left-to-right) ordering.

Pramanik et al. <d-cite key="pramanik2025distillingsemanticallyawareorders"></d-cite> adopt a related idea for images in Ordered AR. After training on random patch permutations, their model evaluates all unfilled positions in parallel and selects the next patch using a top-k scoring heuristic. Fine-tuning on these adaptively chosen paths yields a canonical semantic order that leads to improved Frèchet Inception Distance (FID) metrics compared to the raster order. Taken together, both Ordered AR and adaptive MDM inference treat order selection as a heuristic search problem, showing that even simple rules for choosing the next token can substantially improve autoregressive generation.

### Learning the order
Rather than using a heuristic, Wang et al. <d-cite key="wang2025learningorder"></d-cite> directly parametrize the ordering via a neural network. In **Learning Order Autoregressive Models (LO-ARMs)**, the ordering is treated explicitly as a latent variable: the model learns an order-policy
$$p_x(\sigma) = \Pi_i p(\sigma_i \mid \sigma_{<i}, x_{\sigma_{<i}})$$
that, given a partially masked input, chooses a position to unmask next. At the same time, a shared model (UNet for images, Graph Transformer for molecules) produces value logits for every position. Once the order-policy selects a position $$\sigma_i$$, the model applies a softmax to choose **what value to place there**. The resulting orderings were found to reflect coherent structural patterns, such as placing border pixels in images or bond structures in molecular graphs first in the ordering!

{% include figure.liquid path="blog/autoregressive-tokenization/images/lo-arm.png" class="img-fluid" %}
<div class="caption" style="text-align: center;">
    The LO-ARM sampling process, in which a shared model jointly predicts which position to unmask next and what value to assign.
</div>

A closely related learned-ordering approach is **REOrder** <d-cite key="kutscher2025reorderingpatchesimprovesvision"></d-cite>, which also trains a policy network to select the next position, and then trains an autoregressive model to follow the discovered task-optimal ordering.

The model-level methods show that reordering can meaningfully reduce the difficulty of the generative task, but only within the limits imposed by the underlying tokenization. Since the representation itself is fixed, these approaches cannot introduce coarse-to-fine structure or reshape the information content of the tokens; they can only choose a more favorable sequence in which to model them. As a result, **model-level alignment can improve modelability, but it cannot fully explore the reconstruction-generation tradeoff**.

## Tokenization-level alignment
Tokenization-level methods are motivated by the following question: how can we provide sequential models with the "most sequential," i.e. most modelable, representation of the input data? If the model generates autoregressively, then the tokenizer can be designed with autoregressive generation in mind from the start!. Thus, they address the reconstruction-generation tradeoff directly at the representation level, rather than merely reordering the prediction path.

### Heuristically encouraging AR modelability
A natural starting point is to impose an ordering that we have good reason to believe (e.g. based on domain intuition) will be easier for an autoregressive model to learn. Instead of relying on the model to discover a useful sequence structure on its own, we can choose an ordering that reflects how information in the modality is organized. A prominent example is **Visual Autoregressive Modeling** <d-cite key="tian2024visual"></d-cite>, which aims to align the prediction order with the hierarchical structure of images. VAR  follows a **next-scale** prediction strategy, predicting coarse global structure first and then refining with higher-resolution tokens[^diffusion]. By replacing a rigid raster order with a more semantically coherent prediction schedule, VAR closed much of the historical gap between AR and diffusion models in metrics such as image quality, inference speed, data efficiency, and scalability.


{% include figure.liquid path="blog/autoregressive-tokenization/images/var.png" class="img-fluid" %}
<div class="caption" style="text-align: center;">
    Figure from <d-cite key="tian2024visual"></d-cite>. AR and VAR both perform sequential generation, but AR generates one patch at a time, whereas VAR generates a (globally) better-resolved image with each timestep.
</div>

Note that if prefixes of the token sequence already encode global structure, then shorter sequences essentially give a coarse depiction of the image, while longer sequences progressively increase fidelity. Several works have leveraged this property of coarse-to-fine tokenization to allow for a **variable number of tokens per image**. In doing so, they also tackle a core limitation of the fixed-size patch grid in standard tokenizers, where every image is forced into the same number of tokens regardless of its complexity. A plain sky and a dense texture both consume the same token budget, creating inefficiency for simple images and information loss for complex ones. Instead, sequence length can track the information density of the image.
Some examples of variable-length tokenization methods include:

* **FlexTok** <d-cite key="bachmann2025flextok"></d-cite> uses nested dropout, repeatedly chopping off the tail during training so that high-level content is forced into the early positions.

* **Matryoshka Multimodal Models** <d-cite key="cai2025matryoshka"></d-cite> learn nested token subsets where each prefix is already a valid representation and additional tokens just add finer details.

* **One-D-Piece** <d-cite key="miwa2025onedpieceimagetokenizermeets"></d-cite>introduces a "Tail Token Drop" regularization that removes later tokens on the fly and pushes essential global semantics into the head of the sequence.

This shift from "next-patch'' to "next-scale'' prediction reframes what it means for images to be "non-sequential", and illustrates the flexibility that comes with modifying the data representation directly (rather than just the ordering of predefined tokens). While there is no obvious choice of order in the spatial dimension, VAR suggests that the ordering along the resolution dimension is highly modelable in an autoregressive manner.

This principle likely extends far beyond computer vision. Whether generating molecular graphs (e.g., defining the scaffold before functional groups) or 3D geometry (e.g., blocking out shapes before refining surface details), the more modelable sequence for complex data might be a trajectory from low to high complexity or resolution, rather than a linear path through spatial components. Moreover, the optimal generation path need not be semantically interpretable. Beyond explicit hierarchies like resolution, we expect that many high-dimensional datasets possess hidden latent structures that define a natural generation order, even if it is abstract and invisible to human observers[^language].
Identifying the most modelable order may require deep domain knowledge in most settings. This, in turn, motivates methods that dispense with predefined heuristics and instead build autoregressive-friendly structure directly into the learned tokenization.

### Autoregressive priors
In the standard two-stage tokenization-generation paradigm, Stage 1 and Stage 2 are often treated as independent problems. The tokenizer minimizes reconstruction loss, while the generator minimizes prediction loss. However, a tokenizer that is optimal for reconstruction with a global, bidirectional decoder is often suboptimal for autoregressive generation.
To bridge this gap, recent works have introduced an autoregressive prior directly into the Stage 1 training process. By enforcing causal constraints during tokenization, these methods ensure the resulting codebook aligns with the sequential nature of the downstream generator. We visualize three representative approaches to this alignment in the figure below:

{% include figure.liquid path="blog/autoregressive-tokenization/images/token_level.png" class="img-fluid" %}
<div class="caption" style="text-align: center;">
    Three approaches to incorporating autoregressive priors during tokenization. CRT <d-cite key="ramanujan2025when"></d-cite> adds next-token prediction on continuous latents, AliTok <d-cite key="wu2025sequencemodelingalignmenttokenizer"></d-cite> imposes causal decoding during Stage 1 but relaxes it during Stage 2, and LARP <d-cite key="wang2025larp"></d-cite> applies an AR prior only to global query tokens produced by a stochastic quantizer.
</div>

**Causally Regularized Tokenization** (CRT). Ramanujan et al.  <d-cite key="ramanujan2025when"></d-cite> keep the standard encoder-quantizer-decoder architecture, but modify the objective function. CRT adds a next-token prediction loss on the pre-quantized continuous latents, encouraging tokens to be predictable from their predecessors. This explicitly trades off reconstruction quality for AR predictability and yields better downstream generative performance as well as computational efficiency.

**Aligned Tokenizer** (AliTok). Wu et al. <d-cite key="wu2025sequencemodelingalignmenttokenizer"></d-cite> propose AliTok, which directly constrains the decoder to be autoregressive. While the causal decoder provides a mechanism for enforcing sequential structure in the tokens, it also limits reconstruction quality. During the second stage, this limitation is mitigated by jointly training a high-fidelity bidirectional decoder while retaining the causal structure from the first stage. This approach offers a practical means of reconciling the two objectives, albeit through a multi-step process.

**Learned AutoRegressive generative Prior** (LARP): Instead of forcing autoregressive constraints onto all patch tokens, LARP <d-cite key="wang2025larp"></d-cite> adds a set of learned "holistic" query tokens that summarize high-level video semantics. An AR prior is trained only on these (de-quantized) query vectors, giving them a coherent causal structure without imposing constraints on the low-level patch tokens. Since they opt to use a stochastic vector quantization scheme (sampling from the codebook similarity distribution), the AR prior is trained to predict the next-token distribution.
Despite the surface-level methodological differences between these methods, they all build sequentiality directly into the learned tokenizer, using general learning methods rather than heuristics.

# Conclusion

In sum, autoregressive models offer a flexible, efficient method for generative modeling tasks, but they require tokens to be input in some ordering. However, many modalities of interest outside language lack a clear, natural ordering for tokenization. Although one can simply choose an arbitrary ordering convention, it may not optimize the resultant model's generation quality. This is the notion of "modelability," which we apply specifically to autoregressive models. To improve the autoregressive modelability of arbitrary modalities, model-based approaches broadly attempt to find the most modelable ordering of some fixed tokenization. In contrast, tokenization-based approaches take sequential generation into account when constructing the tokenization itself. This encourages ordered tokenizations that are easy to incrementally predict. In a scattered landscape of diverse tokenization and ordering strategies, we hope to have provided not just a methodological survey, but a unifying perspective.

For researchers who work with boutique architectures or non-language data modalities, we want to highlight tokenization and "sequential-ization" as promising directions for future research -- in particular, modality-specific tokenization methods that anticipate the sequential nature of their downstream model and align with it. Notably, most existing alignment strategies have been explored primarily in the image domain, leaving substantial room for discovering analogous structures in other forms of data.

There are many possible routes for the future of non-sequential data. Perhaps specialized architectures for each modality will win out in the end, and the mismatch we expound upon in this blogpost won't be relevant! But at this point, it seems highly unlikely that the application of large, generalist sequential models for non-sequential data will disappear entirely. After all, even agents calling specialized models as tools must be able to describe the objects of interest with a sequence of tokens. Thus, the square peg for the round hole remains.

[^poem]: As is famously capitalized upon in Jonathan Reed's reversible poem "The Lost Generation" -- a clever work that can be read forwards and backwards, with diametrically opposed meanings

[^soft]: Recent work has explored the use of continuous or "soft" tokens with an infinite vocabulary to allow for a more semantically rich latent space <d-cite key="tschannen2025jetformer"></d-cite>.

[^language]: Even when there is a canonical ordering for the input, it might not be the most faithful to the underlying generative process. For example, one could imagine that the "best" ordering for certain language modeling tasks might be structured according to some hierarchy of abstract concepts or reasoning steps rather than the default left-to-right sequence. There is thus a growing body of work on Transformers that operate in a latent token space <d-cite key="sun2025enhancinglatentcomputationtransformers"></d-cite>.

[^canon]: There are also several works which perform input canonicalization, i.e. choosing an optimal permutation of input data for in-context learning <d-cite key="Lu2021FantasticallyOP"></d-cite> or graph generation <d-cite key="kim2024learningprobabilisticsymmetrizationarchitecture"></d-cite>. Although these approaches also learn an optimal ordering of the input data, they differ from most of the approaches we discuss in the blogpost by committing to an ordering all at once, rather than incrementally as generation progresses.

[^diffusion]: This loosely parallels the way diffusion models reconstruct signals by first resolving low-frequency components and then adding higher-frequency detail. Dieleman <d-cite key="dieleman2025latents"></d-cite> gives an intuitive explanation of the viewpoint that "diffusion is spectral autoregression", and Falck <d-cite key="falck2025fourier"></d-cite> offers a complementary analysis that clarifies the conditions under which the connection holds.

[^tradeoff]: The tradeoff between modelability and reconstruction is not strict ("if A increases, B necessarily decreases"): indeed, model-level approaches to computing the optimal ordering of tokens preserve reconstruction while improving modelability. However, the two are generally competing objectives. For example, as noted in CRT <d-cite key="ramanujan2025when"></d-cite>, the optimally modelable tokenization is a single constant token, which fails at reconstruction.

[^tokenization]: This naturally dovetails with the theory of tokenization discussed by Rajaraman et al. <d-cite key="rajaraman2025theorytokenizationllms"></d-cite>: there, the advantages of tokenization were related to the limitations of the transformer, which tended to learn unigram models in their setting.


# Appendix
## Examples of non-sequential generative models
Masked Language Models (MLMs) depart from the autoregressive likelihood factorization by adopting a denoising objective that permits bidirectional context. The model learns to reconstruct missing tokens from a global view of the input, rather than relying only on past context. This approach was popularized at scale by BERT, which showed that masked language modeling can produce highly transferable representations for a wide range of downstream tasks <d-cite key="devlin-etal-2019-bert"></d-cite>.

Given an input sequence $\mathbf{x}$, a random subset of tokens at positions $M \subset \{1,\dots,n\}$ are replaced with a special [MASK] token. The model receives the masked sequence $\tilde{\mathbf{x}}$, where $\tilde{\mathbf{x}}_i = \text[MASK]$ for $i \in M$, and is optimized to estimate the conditional distributions,
$$p(x_i \mid \tilde{\mathbf{x}}), \quad \text{for all } i \in M.$$

In expectation, this objective is repeated over many different random masks $$M$$, exposing the model to a rich family of partially observed subproblems. In this sense, **autoregressive models can be seen as a specific subproblem of MLMs**, where the masking is always applied to the final position in the input sequence and the prediction is conditioned only on past tokens. Note that unlike ARMs, MLMs do not yield a tractable likelihood over complete sequences. The loss is a sum of cross-entropies over the randomly masked positions, which is not equal to the log-likelihood of the whole sequence (as was the case with ARMs):
$$\mathcal{L_{\text{MLM}}}(\theta) = -\sum_{i \in M} \log p_\theta(x_i \mid \tilde{\mathbf{x}}).$$

Despite relaxing the left-to-right prediction order, MLMs still rely on the underlying positional structure of the sequence (as captured by the positional encodings) to determine which tokens constitute the context for each prediction. Thus, both ARMs and MLMs fundamentally assume an ordered sequence of tokens.

If we view masking as a type of corruption process, then iterating the reconstruction step naturally gives rise to **diffusion models** <d-cite key="Ho2020"></d-cite>. Diffusion models define a forward process that starts from the data $x_0$ and progressively adds noise until the sample becomes nearly Gaussian. The reverse process refines the entire sample at once rather than predicting one symbol at a time, meaning that generation is defined without any notion of token order. While diffusion models have historically achieved better generative performance than autoregressive approaches in continuous domains, recent work suggests that this gap may be narrowing as better latent parameterizations become available <d-cite key="tian2024visual"></d-cite>.

While diffusion was originally formulated using continuous Gaussian noise, several lines of work show that the same iterative denoising idea extends naturally to discrete domains. Discrete denoising diffusion probabilistic models (D3PM) replace Gaussian noise with a categorical corruption process such as random token replacement <d-cite key="austin2021"></d-cite>. Alternatively, **Masked Diffusion Models (MDMs)** use masking rather than categorical replacement as the corruption operator <d-cite key="lou2024discrete"></d-cite>, where each diffusion step applies a random masking pattern and the model is trained to reconstruct the missing content. As highlighted by Zheng et al. <d-cite key="zheng2025masked"></d-cite>, this makes the **learning problem of MDMs equivalent to MLMs**.

# Footnotes
