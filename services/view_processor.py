import os
import random
from typing import Dict, List

import pandas as pd
from runpod import RunPodLogger
from prompts import (
    topic_model_user_prompt,
    view_summary_user_prompt,
    topic_model_system_prompt,
    view_summary_system_prompt,
    vanilla_topic_model_user_prompt,
    vanilla_topic_model_system_prompt,
)
from data_model import TopicModelResponse, ViewSummaryResponse
from litellm.utils import token_counter
from core.topic_modeling import initialize_topic_model, run_topic_model_hierarchical
from integrations.azure_client import run_formated_llm_call_async
from integrations.directus_client import update_directus, get_directus_client

from services.aspect_processor import get_aspect_response_list, fallback_get_aspect_response_list

logger = RunPodLogger()


async def summarise_aspects(
    aspect_response_list: List[Dict],
    response_language: str = "en",
    user_prompt: str = "",
):
    """
    Generate a summary of multiple aspects.

    Args:
        aspect_response_list: List of aspect responses to summarize
        response_language: Language code for summary generation (default: 'en')

    Returns:
        Dict: Summary response containing overview of all aspects

    Raises:
        ValueError: If aspect_response_list is empty
    """
    if len(aspect_response_list) == 0:
        raise ValueError("No aspects to summarise")
    aspect_texts = [
        f"{aspect['title']}\n{aspect['description']}\n{aspect['summary']}"
        for aspect in aspect_response_list
    ]
    view_text = "\n\n".join(aspect_texts)
    messages = [
        {"role": "system", "content": view_summary_system_prompt},
        {
            "role": "user",
            "content": view_summary_user_prompt.format(
                view_text=view_text,
                response_language=response_language,
                user_prompt=user_prompt,
            ),
        },
    ]
    try:
        view_response = await run_formated_llm_call_async(messages, ViewSummaryResponse)
        return view_response
    except Exception as e:
        logger.error(f"Error in LLM call for view summary: {e}")
        # Return a minimal summary response
        return {
            "title": "Error in Summary Generation",
            "description": f"Unable to generate view summary due to error: {str(e)}",
            "summary": "Summary generation failed",
        }


async def get_views_aspects(
    segment_ids: List[str],
    user_prompt: str,
    project_analysis_run_id: str,
    response_language: str | None = None,
    threshold_context_length: int = int(os.getenv("THRESHOLD_CONTEXT_LENGTH", 100000)),
    user_input: str = "",
    user_input_description: str = "",
) -> Dict:
    """
    Generate comprehensive views and aspects analysis for conversation segments.

    This function performs the following steps:
    1. Retrieves contextual transcripts for given segment IDs
    2. Performs topic modeling using either direct LLM or BERTopic based on context length
    3. Generates detailed aspect responses for each identified topic
    4. Creates a summary of all aspects

    Args:
        segment_ids: List of segment IDs to analyze
        user_prompt: User's query or instruction for analysis
        response_language: Language code for response generation (default: 'en')
        context_length: Maximum token length for direct LLM processing (default: 100000)

    Returns:
        Dict: Contains:
            - views: Summary of all aspects
            - aspects: List of detailed aspect responses
            - seed: Original user prompt
            - language: Response language used
    """
    if response_language is None:
        response_language = "en"

    segments = get_directus_client().get_items(
        "conversation_segment",
        {
            "query": {
                "filter": {"id": {"_in": segment_ids}, "transcript": {"_nnull": True}},
                "fields": ["id", "contextual_transcript", "transcript"],
                "limit": -1,
            },
        },
    )
    logger.debug(f"Retrieved segments: {segments}")
    # Type-safe dictionary comprehension with explicit type checking
    segment_2_transcript: Dict[int, str] = {}
    raw_docs: List[str] = []
    doc_ids: List[str] = []
    for segment in segments:
        if isinstance(segment, dict):
            if "id" in segment and "transcript" in segment:
                segment_2_transcript[int(segment["id"])] = str(segment["transcript"])
                doc_ids.append(str(segment["id"]))
            else:
                raise ValueError(f"Segment {segment} does not have an id or transcript")

            if "contextual_transcript" in segment:
                raw_docs.append(str(segment["contextual_transcript"]))
            else:
                raise ValueError(f"Segment {segment} does not have a contextual transcript")

    # Process docs with explicit type handling
    split_docs: List[List[str]] = [doc.split("\n") for doc in raw_docs if doc != ""]
    docs: List[str] = []

    for sublist in split_docs:
        docs.extend(sublist)

    topic_model = initialize_topic_model()
    token_length = 0
    for doc in docs:
        token_length += token_counter(model=str(os.getenv("AZURE_MODEL")), text=doc)

    if token_length < threshold_context_length:
        docs_with_ids = "---------\n\n".join(
            [f"SEGMENT_ID_{doc_id}: {doc}" for doc_id, doc in zip(doc_ids, raw_docs)]
        )
        messages = [
            {"role": "system", "content": vanilla_topic_model_system_prompt},
            {
                "role": "user",
                "content": vanilla_topic_model_user_prompt.format(
                    docs_with_ids=docs_with_ids,
                    user_prompt=user_prompt,
                    response_language=response_language,
                ),
            },
        ]
        try:
            tentative_aspects_response = await run_formated_llm_call_async(
                messages, TopicModelResponse
            )
        except Exception as e:
            logger.error(f"Error in LLM call for topic modeling (vanilla path): {e}")
            raise e
    else:
        topics, probs, hierarchical_topics = run_topic_model_hierarchical(topic_model, docs)
        repr_docs_token_length = threshold_context_length * 1.1
        nr_repr_docs = 100
        while repr_docs_token_length > threshold_context_length * 0.8 and nr_repr_docs > 3:
            doc_topic = pd.DataFrame(
                {
                    "Topic": topic_model.topics_,
                    "ID": range(len(topic_model.topics_)),
                    "Document": docs,
                }
            )
            repr_docs, _, _, _ = topic_model._extract_representative_docs(
                topic_model.c_tf_idf_,
                doc_topic,
                topic_model.topic_representations_,
                nr_samples=1000,
                nr_repr_docs=nr_repr_docs,
            )
            rep_doc_list = {k: v for k, v in repr_docs.items() if k != -1}
            nl = "\n\n"
            rep_docs_formatted = [
                f"Document set {k} : \n {nl.join(v)}" for k, v in rep_doc_list.items()
            ]
            representative_documents = "\n\n\n\n\n\n".join(rep_docs_formatted)
            repr_docs_token_length = token_counter(
                model=str(os.getenv("AZURE_MODEL")),
                text=representative_documents,
            )
            nr_repr_docs = round(nr_repr_docs * 0.75)
        messages = [
            {"role": "system", "content": topic_model_system_prompt},
            {
                "role": "user",
                "content": topic_model_user_prompt.format(
                    representative_documents=representative_documents,
                    user_prompt=user_prompt,
                    response_language=response_language,
                ),
            },
        ]
        try:
            tentative_aspects_response = await run_formated_llm_call_async(
                messages, TopicModelResponse
            )
        except Exception as e:
            logger.error(f"Error in LLM call for topic modeling: {e}")
            raise e

    tentative_aspects = tentative_aspects_response["topics"]
    logger.info(f"Tentative aspects: {tentative_aspects}")
    aspect_response_list = await get_aspect_response_list(
        tentative_aspects,
        segment_ids,
        segment_2_transcript,
        response_language=response_language,
    )
    views_dict = await summarise_aspects(
        aspect_response_list,
        response_language=response_language,
        user_prompt=user_prompt,
    )
    views_dict["aspects"] = aspect_response_list
    views_dict["seed"] = user_prompt
    views_dict["language"] = response_language
    views_dict["user_input"] = user_input
    views_dict["user_input_description"] = user_input_description
    response = {"view": views_dict}
    update_directus(response, project_analysis_run_id)
    return response


async def get_views_aspects_fallback(
    segment_ids: List[str],
    user_prompt: str,
    project_analysis_run_id: str,
    response_language: str | None = None,
    threshold_context_length: int = int(os.getenv("THRESHOLD_CONTEXT_LENGTH", 100000)),
    user_input: str = "",
    user_input_description: str = "",
) -> Dict:
    summaries = get_directus_client().get_items(
        "conversation_segment",
        {
            "query": {
                "filter": {"id": {"_in": segment_ids}},
                "fields": ["id", "transcript", "conversation_id.summary"],
            },
        },
    )
    logger.debug(f"Retrieved summaries: {summaries}")
    segment_2_transcript: Dict[int, str] = {}
    for summary in summaries:
        segment_2_transcript[int(summary["id"])] = str(summary["transcript"])
    summaries_list = list(
        set([(summary["id"], summary["conversation_id"]["summary"]) for summary in summaries])
    )
    random.shuffle(summaries_list)
    samples_to_summarise = []
    token_count = 0
    for summary in summaries_list:
        if (
            token_count + token_counter(model=str(os.getenv("AZURE_MODEL")), text=summary[1])
            > threshold_context_length * 0.8
        ):
            break
        samples_to_summarise.append(summary)
        token_count += token_counter(model=str(os.getenv("AZURE_MODEL")), text=summary[1])

    # Do the vanilla path
    docs_with_ids = "---------\n\n".join(
        [f"SEGMENT_ID_{summary[0]}: {summary[1]}" for summary in samples_to_summarise]
    )
    messages = [
        {"role": "system", "content": vanilla_topic_model_system_prompt},
        {
            "role": "user",
            "content": vanilla_topic_model_user_prompt.format(
                docs_with_ids=docs_with_ids,
                user_prompt=user_prompt,
                response_language=response_language,
            ),
        },
    ]
    try:
        tentative_aspects_response = await run_formated_llm_call_async(messages, TopicModelResponse)
    except Exception as e:
        logger.error(f"Error in LLM call for topic modeling (fallback path): {e}")
        # Create a fallback response
        tentative_aspects_response = {"topics": ["General Discussion", "Key Points", "Main Themes"]}
    tentative_aspects = tentative_aspects_response["topics"]
    aspect_response_list = await fallback_get_aspect_response_list(
        tentative_aspects,
        docs_with_ids,
        user_prompt,
        segment_2_transcript,
        response_language=response_language,
    )
    views_dict = await summarise_aspects(
        aspect_response_list,
        response_language=response_language,
        user_prompt=user_prompt,
    )
    views_dict["aspects"] = aspect_response_list
    views_dict["seed"] = user_prompt
    views_dict["language"] = response_language
    views_dict["user_input"] = user_input
    views_dict["user_input_description"] = user_input_description
    response = {"view": views_dict}
    update_directus(response, project_analysis_run_id)
    return response
