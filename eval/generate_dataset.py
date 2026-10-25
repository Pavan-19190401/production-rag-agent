import json
from pathlib import Path
from typing import List, Dict, Any

OUTPUT_PATH = Path(__file__).resolve().parent / "golden_dataset.jsonl"
CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"


def build_golden_dataset():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Curated golden dataset: 65 questions covering factual, multi-hop, numeric, and unanswerable
    dataset: List[Dict[str, Any]] = [
        # --- FlashAttention (flash_attention.md) ---
        {
            "id": "q_001",
            "question": "What is the primary memory bottleneck FlashAttention addresses on GPUs?",
            "reference_answer": "FlashAttention addresses the memory reads and writes bottleneck between GPU High Bandwidth Memory (HBM) and GPU on-chip SRAM caused by materializing the intermediate N x N attention matrix.",
            "source_chunk_ids": ["flash_attention.md_p1_Abstract_c0", "flash_attention.md_p1_GPU_Memory_Hierarchy_and_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_002",
            "question": "What is the bandwidth and capacity of on-chip SRAM compared to HBM on an NVIDIA A100 GPU?",
            "reference_answer": "On an NVIDIA A100 GPU, on-chip SRAM has ~19 TB/s bandwidth with 192 KB per Streaming Multiprocessor (totaling ~20MB), whereas HBM has ~1.5 to 2.0 TB/s bandwidth with 40GB to 80GB capacity.",
            "source_chunk_ids": ["flash_attention.md_p1_GPU_Memory_Hierarchy_and_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_003",
            "question": "How does FlashAttention compute exact attention without storing the full N x N matrix in HBM?",
            "reference_answer": "FlashAttention uses a tiling algorithm with an online softmax formulation to compute exact attention in blocks that fit within fast on-chip SRAM without materializing the full N x N matrix in HBM.",
            "source_chunk_ids": ["flash_attention.md_p1_Tiling_Algorithm_c0", "flash_attention.md_p1_Online_Softmax_Formulati_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_004",
            "question": "What speedup does FlashAttention achieve over standard PyTorch attention?",
            "reference_answer": "FlashAttention achieves a 2x to 4x wall-clock speedup over standard PyTorch attention for sequence lengths from 128 to 4096.",
            "source_chunk_ids": ["flash_attention.md_p1_Experimental_Results_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_005",
            "question": "Does FlashAttention sacrifice mathematical precision compared to standard attention?",
            "reference_answer": "No, FlashAttention produces exact attention weights with zero loss in mathematical precision compared to standard attention.",
            "source_chunk_ids": ["flash_attention.md_p1_Experimental_Results_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_006",
            "question": "What maximum context length can FlashAttention scale to on a single GPU without memory exhaustion?",
            "reference_answer": "FlashAttention enables context lengths up to 64k tokens on single GPUs without memory exhaustion.",
            "source_chunk_ids": ["flash_attention.md_p1_Experimental_Results_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_007",
            "question": "What is the time and memory complexity of standard self-attention relative to sequence length?",
            "reference_answer": "The time and memory complexity of standard self-attention are quadratic in sequence length.",
            "source_chunk_ids": ["flash_attention.md_p1_Abstract_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_008",
            "question": "How are Q, K, and V matrices manipulated in the FlashAttention tiling algorithm?",
            "reference_answer": "The tiling algorithm breaks the Q, K, and V matrices into blocks that fit entirely inside fast on-chip SRAM.",
            "source_chunk_ids": ["flash_attention.md_p1_Tiling_Algorithm_c0"],
            "answerable": True,
            "question_type": "factual"
        },

        # --- Attention Is All You Need (attention_paper.pdf) ---
        {
            "id": "q_009",
            "question": "What is the formula for Scaled Dot-Product Attention in the Transformer?",
            "reference_answer": "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V",
            "source_chunk_ids": ["attention_paper.pdf_p4_Page_4_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_010",
            "question": "Why do the authors scale the dot products by 1 / sqrt(d_k)?",
            "reference_answer": "For large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Scaling by 1 / sqrt(d_k) counteracts this effect.",
            "source_chunk_ids": ["attention_paper.pdf_p4_Page_4_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_011",
            "question": "What are the values of d_model, number of heads h, and d_k in the base Transformer model?",
            "reference_answer": "In the base model, d_model = 512, number of heads h = 8, and d_k = d_v = d_model / h = 64.",
            "source_chunk_ids": ["attention_paper.pdf_p5_Page_5_c0", "attention_paper.pdf_p9_Page_9_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_012",
            "question": "What BLEU score did the Transformer big model achieve on the WMT 2014 English-to-German translation task?",
            "reference_answer": "The Transformer (big) model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.",
            "source_chunk_ids": ["attention_paper.pdf_p8_Page_8_c1"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_013",
            "question": "What positional encoding functions are used in the Transformer?",
            "reference_answer": "Sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos / 10000^(2i / d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model)).",
            "source_chunk_ids": ["attention_paper.pdf_p6_Page_6_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_014",
            "question": "How many layers N are in the encoder and decoder of the base Transformer?",
            "reference_answer": "Both the encoder and decoder are composed of a stack of N = 6 identical layers.",
            "source_chunk_ids": ["attention_paper.pdf_p3_Page_3_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_015",
            "question": "What optimizer and learning rate schedule was used to train the Transformer models?",
            "reference_answer": "The Adam optimizer was used with beta1 = 0.9, beta2 = 0.98, and eps = 10^-9, with a learning rate schedule that increases linearly for warmup_steps and decreases proportionally to the inverse square root of the step number.",
            "source_chunk_ids": ["attention_paper.pdf_p7_Page_7_c1"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_016",
            "question": "What inner dimension d_ff is used in the feed-forward networks of the base Transformer?",
            "reference_answer": "The inner dimension of the feed-forward network is d_ff = 2048.",
            "source_chunk_ids": ["attention_paper.pdf_p5_Page_5_c1"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_017",
            "question": "Why does the Transformer decoder use masking in its self-attention sub-layer?",
            "reference_answer": "Masking ensures that predictions for position i can depend only on known outputs at positions less than i, preventing information flow from subsequent positions.",
            "source_chunk_ids": ["attention_paper.pdf_p3_Page_3_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_018",
            "question": "What dropout rate was applied to the Transformer during training?",
            "reference_answer": "A residual dropout rate of P_drop = 0.1 was applied to the output of each sub-layer.",
            "source_chunk_ids": ["attention_paper.pdf_p7_Page_7_c1"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_019",
            "question": "What BLEU score did the big Transformer achieve on the WMT 2014 English-to-French translation task?",
            "reference_answer": "The big model achieved a state-of-the-art BLEU score of 41.0 (and 41.8) on WMT 2014 English-to-French.",
            "source_chunk_ids": ["attention_paper.pdf_p8_Page_8_c1"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_020",
            "question": "What hardware was used to train the base Transformer model and for how long?",
            "reference_answer": "The base models were trained on one machine with 8 NVIDIA P100 GPUs for 100,000 steps or 12 hours.",
            "source_chunk_ids": ["attention_paper.pdf_p7_Page_7_c1"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_021",
            "question": "How does Multi-Head Attention differ from single attention in terms of representation subspaces?",
            "reference_answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which single attention pooling inhibits.",
            "source_chunk_ids": ["attention_paper.pdf_p5_Page_5_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_022",
            "question": "What label smoothing value epsilon_ls was applied during Transformer training?",
            "reference_answer": "A label smoothing value of epsilon_ls = 0.1 was used during training.",
            "source_chunk_ids": ["attention_paper.pdf_p8_Page_8_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_023",
            "question": "Can the Transformer model be used without recurrence or convolutions?",
            "reference_answer": "Yes, the Transformer model architecture relies entirely on an attention mechanism to draw global dependencies between input and output without recurrence or convolutions.",
            "source_chunk_ids": ["attention_paper.pdf_p2_Page_2_c0"],
            "answerable": True,
            "question_type": "factual"
        },

        # --- Retrieval-Augmented Generation (rag_paper.pdf) ---
        {
            "id": "q_024",
            "question": "What are the two core components of a RAG model as introduced by Lewis et al.?",
            "reference_answer": "The two components are a retriever (Dense Passage Retriever / DPR) and a parametric generator (BART seq2seq model).",
            "source_chunk_ids": ["rag_paper.pdf_p1_Page_1_c0", "rag_paper.pdf_p2_Page_2_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_025",
            "question": "What is the difference between RAG-Sequence and RAG-Token models?",
            "reference_answer": "RAG-Sequence uses the same retrieved document to generate the complete target sequence, whereas RAG-Token can use different retrieved documents to generate each token in the target sequence.",
            "source_chunk_ids": ["rag_paper.pdf_p3_Page_3_c0", "rag_paper.pdf_p3_Page_3_c1"],
            "answerable": True,
            "question_type": "multi_hop"
        },
        {
            "id": "q_026",
            "question": "What corpus and chunk size was used for the non-parametric knowledge index in the original RAG paper?",
            "reference_answer": "The December 2018 dump of Wikipedia was used, split into 21 million 100-word passages.",
            "source_chunk_ids": ["rag_paper.pdf_p4_Page_4_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_027",
            "question": "What indexing tool was used to search the 21 million passages in RAG?",
            "reference_answer": "FAISS (Hierarchical Navigable Small World / HNSW index) was used for fast maximum inner product search.",
            "source_chunk_ids": ["rag_paper.pdf_p4_Page_4_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_028",
            "question": "What pre-trained generator model is used in RAG?",
            "reference_answer": "BART-large, a 400M parameter seq2seq Transformer model.",
            "source_chunk_ids": ["rag_paper.pdf_p4_Page_4_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_029",
            "question": "How did RAG perform on the Natural Questions (NQ) benchmark compared to closed-book models?",
            "reference_answer": "RAG achieved 44.5% exact match on Natural Questions, outperforming closed-book models including T5-11B.",
            "source_chunk_ids": ["rag_paper.pdf_p6_Page_6_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_030",
            "question": "What is the formula for the retrieval probability p(z|x) in RAG?",
            "reference_answer": "p(z|x) is proportional to exp(d(z)^T q(x)), where d(z) is the dense document embedding and q(x) is the dense query embedding produced by BERT networks.",
            "source_chunk_ids": ["rag_paper.pdf_p3_Page_3_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_031",
            "question": "How many documents K are retrieved during training and testing in RAG?",
            "reference_answer": "Top K documents, typically K = 5 or K = 10 retrieved passages.",
            "source_chunk_ids": ["rag_paper.pdf_p4_Page_4_c1"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_032",
            "question": "Can the non-parametric knowledge in RAG be updated without retraining the entire model?",
            "reference_answer": "Yes, non-parametric memory can be updated by swapping or updating the document index without retraining the generator parameters.",
            "source_chunk_ids": ["rag_paper.pdf_p1_Page_1_c0", "rag_paper.pdf_p7_Page_7_c2"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_033",
            "question": "What exact match score did RAG-Sequence achieve on Open-Domain TriviaQA?",
            "reference_answer": "RAG-Sequence achieved 56.8% exact match on TriviaQA.",
            "source_chunk_ids": ["rag_paper.pdf_p6_Page_6_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_034",
            "question": "How does RAG generate more factual and specific responses than purely parametric seq2seq models?",
            "reference_answer": "By conditioning generation on retrieved external documents, RAG generates responses that are more factual, specific, and less prone to hallucinations than parametric-only models.",
            "source_chunk_ids": ["rag_paper.pdf_p1_Page_1_c0", "rag_paper.pdf_p6_Page_6_c1"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_035",
            "question": "What decoding algorithm is used for RAG-Token generation?",
            "reference_answer": "Standard autoregressive beam search decoding using the transition probability obtained by marginalizing over retrieved documents at each step.",
            "source_chunk_ids": ["rag_paper.pdf_p4_Page_4_c1"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_036",
            "question": "What is Thorough Decoding in the context of RAG-Sequence models?",
            "reference_answer": "Thorough Decoding runs beam search for each retrieved document z, pools all generated candidate hypotheses, and scores them by summing probabilities across documents.",
            "source_chunk_ids": ["rag_paper.pdf_p4_Page_4_c1"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_037",
            "question": "What benchmark was used to test RAG on Jeopardy question generation?",
            "reference_answer": "The Jeopardy question generation benchmark, evaluating whether RAG generates two distinct clues within the same generated question.",
            "source_chunk_ids": ["rag_paper.pdf_p7_Page_7_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_038",
            "question": "Does fine-tuning RAG backpropagate into the query encoder of the retriever?",
            "reference_answer": "Yes, during fine-tuning the gradients from the generator backpropagate into the DPR question encoder, adapting retrieval to the target task.",
            "source_chunk_ids": ["rag_paper.pdf_p3_Page_3_c1"],
            "answerable": True,
            "question_type": "factual"
        },

        # --- LoRA: Low-Rank Adaptation (lora_paper.pdf) ---
        {
            "id": "q_039",
            "question": "What is the core mathematical formulation of LoRA weight adaptation?",
            "reference_answer": "W = W_0 + Delta W = W_0 + B * A, where W_0 is d x k, B is d x r, A is r x k, and rank r << min(d, k).",
            "source_chunk_ids": ["lora_paper.pdf_p3_Page_3_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_040",
            "question": "How are matrices A and B initialized in LoRA?",
            "reference_answer": "Matrix A is initialized with a Gaussian distribution N(0, sigma^2) and matrix B is initialized to 0, ensuring Delta W = 0 at the start of training.",
            "source_chunk_ids": ["lora_paper.pdf_p3_Page_3_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_041",
            "question": "What scaling factor is applied to Delta W in LoRA?",
            "reference_answer": "Delta W is scaled by alpha / r, where alpha is a constant hyperparameter typically set to the initial rank r.",
            "source_chunk_ids": ["lora_paper.pdf_p3_Page_3_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_042",
            "question": "By how much does LoRA reduce trainable parameters compared to full fine-tuning on GPT-3 175B?",
            "reference_answer": "LoRA reduces trainable parameters by 10,000 times (from 175B to approximately 37.7M parameters) and GPU memory consumption by 3 times.",
            "source_chunk_ids": ["lora_paper.pdf_p1_Page_1_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_043",
            "question": "Does LoRA introduce any additional inference latency when deployed in production?",
            "reference_answer": "No, because the learned low-rank weights Delta W = B * A can be directly folded into W_0 = W_0 + B * A during inference, incurring zero extra latency.",
            "source_chunk_ids": ["lora_paper.pdf_p1_Page_1_c0", "lora_paper.pdf_p4_Page_4_c2"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_044",
            "question": "Which Transformer attention weight matrices does LoRA typically adapt?",
            "reference_answer": "LoRA primarily adapts the query (W_q) and value (W_v) attention projection matrices.",
            "source_chunk_ids": ["lora_paper.pdf_p4_Page_4_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_045",
            "question": "What rank r is often sufficient for competitive performance in downstream tasks according to LoRA experiments?",
            "reference_answer": "A rank as small as r = 1, 2, or 4 is often sufficient to match or exceed full fine-tuning performance.",
            "source_chunk_ids": ["lora_paper.pdf_p8_Page_8_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_046",
            "question": "How does LoRA prevent catastrophic forgetting during domain-specific adaptation?",
            "reference_answer": "The pre-trained weights W_0 are kept frozen during training, preserving the original general capabilities while learning task-specific knowledge in Delta W.",
            "source_chunk_ids": ["lora_paper.pdf_p1_Page_1_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_047",
            "question": "How much memory savings does LoRA achieve for optimizer states during training?",
            "reference_answer": "Because frozen parameters do not need optimizer states, optimizer memory is reduced by up to 2/3 compared to Adam fine-tuning.",
            "source_chunk_ids": ["lora_paper.pdf_p2_Page_2_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_048",
            "question": "Can different LoRA adapter weights be switched at runtime on a shared base model?",
            "reference_answer": "Yes, multiple task-specific LoRA adapters can be swapped in and out seamlessly on a single shared base model instance.",
            "source_chunk_ids": ["lora_paper.pdf_p1_Page_1_c0", "lora_paper.pdf_p4_Page_4_c2"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_049",
            "question": "What benchmark results did LoRA achieve on GLUE compared to full fine-tuning of RoBERTa?",
            "reference_answer": "LoRA matched or slightly surpassed full fine-tuning on RoBERTa-large on GLUE tasks while updating fewer than 1% of parameters.",
            "source_chunk_ids": ["lora_paper.pdf_p6_Page_6_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_050",
            "question": "What is the intrinsic dimension hypothesis that motivates LoRA?",
            "reference_answer": "The hypothesis that models have a low intrinsic dimension, meaning that model parameter updates reside in a subspace of much lower dimension than the ambient parameter space.",
            "source_chunk_ids": ["lora_paper.pdf_p2_Page_2_c0"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_051",
            "question": "Does adapting all attention weights (W_q, W_k, W_v, W_o) with lower rank perform better than adapting only W_q with higher rank?",
            "reference_answer": "Yes, adapting multiple weight matrices with a smaller rank r generally yields better performance than adapting a single weight matrix with a larger rank.",
            "source_chunk_ids": ["lora_paper.pdf_p7_Page_7_c0"],
            "answerable": True,
            "question_type": "multi_hop"
        },
        {
            "id": "q_052",
            "question": "What is the relationship between subspace similarity and rank r in LoRA weights?",
            "reference_answer": "Grassmann distance and canonical correlation analysis show that top singular vector directions overlap significantly across different ranks r.",
            "source_chunk_ids": ["lora_paper.pdf_p8_Page_8_c1"],
            "answerable": True,
            "question_type": "factual"
        },
        {
            "id": "q_053",
            "question": "Why does prefix tuning or prompt tuning face sequence length limitations compared to LoRA?",
            "reference_answer": "Prefix tuning consumes part of the input sequence length tokens with virtual prefix tokens, reducing available context window for actual user input, which LoRA avoids.",
            "source_chunk_ids": ["lora_paper.pdf_p2_Page_2_c0"],
            "answerable": True,
            "question_type": "multi_hop"
        },

        # --- Architecture & Hybrid Retrieval (rag_system_architecture.html) ---
        {
            "id": "q_054",
            "question": "What is the formula and standard constant k for Reciprocal Rank Fusion (RRF)?",
            "reference_answer": "RRF(d) = sum(1 / (k + rank(d))) where k is typically set to 60.",
            "source_chunk_ids": ["rag_system_architecture.html_p1_Production_RAG_Architect_c0"],
            "answerable": True,
            "question_type": "numeric"
        },
        {
            "id": "q_055",
            "question": "Why does a production RAG system combine BM25 sparse search with dense vector search?",
            "reference_answer": "Dense vector search captures semantic similarity but struggles with exact keywords, part numbers, abbreviations, and identifiers, which BM25 handles effectively.",
            "source_chunk_ids": ["rag_system_architecture.html_p1_Production_RAG_Architect_c0"],
            "answerable": True,
            "question_type": "factual"
        },

        # --- Unanswerable Questions (Answerable = False, Strict Refusal Test) ---
        {
            "id": "q_056",
            "question": "What is the return and refund policy for annual enterprise subscription plans?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_057",
            "question": "How many qubits are required to execute Shor's algorithm in Qiskit on IBM Quantum Heron?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_058",
            "question": "What are the eligibility requirements for the corporate employee health reimbursement account?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_059",
            "question": "What is the architectural breakdown and training dataset size of OpenAI's GPT-5?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_060",
            "question": "What is the maximum baggage weight allowance on transatlantic Lufthansa flights?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_061",
            "question": "How does the Linux kernel handle virtual memory page faults on ARM64 processors?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_062",
            "question": "What is the difficulty adjustment epoch duration in the Bitcoin proof-of-work protocol?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_063",
            "question": "What is the recommended dosage of amoxicillin for pediatric respiratory infections?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_064",
            "question": "What are the zoning regulations and permit fees for residential solar installations in Austin, Texas?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        },
        {
            "id": "q_065",
            "question": "What is the warranty coverage duration for Tesla Model 3 high-voltage battery degradation?",
            "reference_answer": "I don't know based on the provided documents.",
            "source_chunk_ids": [],
            "answerable": False,
            "question_type": "unanswerable"
        }
    ]

    # Save to JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(dataset)} golden dataset records in {OUTPUT_PATH}")
    unans = sum(1 for d in dataset if not d["answerable"])
    ans = len(dataset) - unans
    print(f"Breakdown: {ans} answerable questions, {unans} unanswerable refusal tests.")


if __name__ == "__main__":
    build_golden_dataset()
