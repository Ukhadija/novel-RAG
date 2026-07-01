import json
from answer_generator import generate_answer_v2
def build_verifier_prompt(query, draft_answer, retrieved_chunks):
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"[{c['chunk_id']}]\n{c['text']}")
    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a strict verification agent. Your job is to check whether "
        "a draft answer is supported by the provided context chunks.\n\n"
        "CRITICAL RULE: You must base your verdict ONLY on what the context "
        "chunks explicitly state. Do NOT use your own knowledge of the story, "
        "characters, or plot — even if you think you know the answer. If the "
        "context chunks support the draft answer, approve it. Your job is to "
        "verify grounding, not to fact-check against your memory.\n\n"
        "You must respond with ONLY a JSON object, no other text:\n"
        '{\n'
        '  "verdict": "APPROVE" | "EDIT" | "REJECT" | "NEED_MORE_CONTEXT",\n'
        '  "reasoning": "<cite the specific chunk_id that supports or contradicts the claim>",\n'
        '  "unsupported_claims": ["<exact quote of claim not found in any chunk>"],\n'
        '  "revised_answer": "<only if EDIT, otherwise empty string>"\n'
        '}\n\n'
        "Verdict definitions:\n"
        "- APPROVE: the draft's claims are supported by the context chunks.\n"
        "- EDIT: some claims are unsupported by the chunks — remove or fix only those parts.\n"
        "- REJECT: the draft is mostly unsupported and no reasonable edit salvages it.\n"
        "- NEED_MORE_CONTEXT: the chunks are insufficient to verify the answer at all."
    )

    user_prompt = (
        f"CONTEXT CHUNKS:\n{context_str}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"DRAFT ANSWER TO VERIFY:\n{draft_answer}\n\n"
        "Check whether the draft is supported by the context chunks above. "
        "Respond with JSON only."
    )

    return system_prompt, user_prompt


def run_verifier(query, draft_answer, retrieved_chunks, max_new_tokens=400):
    system_prompt, user_prompt = build_verifier_prompt(query, draft_answer, retrieved_chunks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    raw_response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # Small open models often wrap JSON in markdown fences or add stray text —
    # extract just the {...} block defensively rather than assuming clean JSON.
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}") + 1
        parsed = json.loads(raw_response[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {
            "verdict": "PARSE_ERROR",
            "reasoning": f"Could not parse verifier output: {raw_response[:200]}",
            "unsupported_claims": [],
            "revised_answer": "",
        }

    return parsed

def answer_with_verification(query, retriever,model, tokenizer, max_retries=1, top_k=5, max_edit_reverifications=1):
    trace = []
    current_top_k = top_k
    attempt = 0
    REFUSAL_PHRASE = "I don't have enough information"

    while True:
        draft = generate_answer_v2(query, retriever,model = model, tokenizer=tokenizer, top_k=current_top_k)

        if REFUSAL_PHRASE in draft["answer"]:
            verdict_obj = {
                "verdict": "APPROVE",
                "reasoning": "Generator declined to answer; no factual claims to verify.",
                "unsupported_claims": [],
                "revised_answer": "",
            }
            trace.append({
                "stage": "initial", "attempt": attempt, "top_k_used": current_top_k,
                "draft_answer": draft["answer"], "verdict": verdict_obj["verdict"],
                "reasoning": verdict_obj["reasoning"], "unsupported_claims": [],
            })
            break

        verdict_obj = run_verifier(query, draft["answer"], draft["retrieved_chunks"])
        trace.append({
            "stage": "initial", "attempt": attempt, "top_k_used": current_top_k,
            "draft_answer": draft["answer"], "verdict": verdict_obj["verdict"],
            "reasoning": verdict_obj.get("reasoning", ""),
            "unsupported_claims": verdict_obj.get("unsupported_claims", []),
        })

        if verdict_obj["verdict"] == "NEED_MORE_CONTEXT" and attempt < max_retries:
            current_top_k = current_top_k + 5
            attempt += 1
            continue
        break

    # --- NEW: re-verify EDIT outputs instead of trusting them blindly ---
    reverify_count = 0
    while (
        verdict_obj["verdict"] == "EDIT"
        and verdict_obj.get("revised_answer")
        and reverify_count < max_edit_reverifications
    ):
        candidate_answer = verdict_obj["revised_answer"]
        reverify_obj = run_verifier(query, candidate_answer, draft["retrieved_chunks"])
        trace.append({
            "stage": "edit_reverification",
            "attempt": reverify_count,
            "candidate_answer": candidate_answer,
            "verdict": reverify_obj["verdict"],
            "reasoning": reverify_obj.get("reasoning", ""),
            "unsupported_claims": reverify_obj.get("unsupported_claims", []),
        })
        verdict_obj = reverify_obj
        reverify_count += 1

    # Determine final answer based on the LAST verdict obtained
    if verdict_obj["verdict"] == "APPROVE":
        # Either the original draft was clean, or a revised answer got re-approved
        final_answer = verdict_obj.get("revised_answer") or draft["answer"]
    elif verdict_obj["verdict"] == "EDIT":
        final_answer = verdict_obj["revised_answer"]
    elif verdict_obj["verdict"] == "REJECT":
        final_answer = "I don't have enough verified information in the source material to answer this confidently."
    else:  # PARSE_ERROR or anything unexpected 
        final_answer = "I don't have enough verified information in the source material to answer this confidently."

    return {
        "query": query,
        "final_answer": final_answer,
        "final_verdict": verdict_obj["verdict"],
        "trace": trace,
        "retrieved_chunks": draft["retrieved_chunks"],
    }