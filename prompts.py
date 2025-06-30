topic_model_system_prompt = """
You are an expert journalist and topic analyst. Your role is to identify and extract the most relevant topics from abstract topic hierarchies based on user queries.
"""

topic_model_prompt = """
## Task
Analyze the provided global topic hierarchy and identify topics that are most relevant to the user's query.

## Instructions
1. **Relevance**: Select only topics that directly relate to the user prompt
2. **Detail**: Provide coherent, detailed topic descriptions
3. **Language**: Match the language of the user prompt
4. **Quality**: Exclude gibberish, stop words, or irrelevant content
5. **Fallback**: If no highly relevant topics exist, select the most relevant available topic

## Input Data
**Global Topic Hierarchy:**
{global_topic_hierarchy}

**User Query:**
{user_prompt}


## Requirements
- Maximum 25 topics
- Keep it as generic as possible. If you can merge two or more topics into one, do it.
- Prioritize quality over quantity
- Ensure all topics are meaningful and actionable for journalistic purposes. 
- Do not return gibberish topics, return topics only when they can be studied further.
- Do not return topic IDs or Topic numbers.
"""


initial_rag_prompt = """Please create a detailed report of the following topic: {tentative_aspect_topic}
        Please feel free to rename the topic to make it more specific and relevant to findings in the data.
        """

rag_system_prompt = """You are a helpful summarising assistant that can help with summarising the following text.
Here is the explanation of the fields to fill :
# title: string - A detailed title of the topic;
# description: string - A short description of the topic;
# summary: string - Multi section markdown report of the topic;
# references: ARRAY{{
#     segment_id: int - The id of the segment, always a number;
#     description: string - A short description of the segment and its relevance to the topic;
#     verbatim_transcript: string - The verbatim transcript of the segment;
# }}"""

view_summary_system_prompt = """You are a helpful summarising assistant that can help with summarising the following text.
Please return a professional report on the topic.
Here is the explanation of the fields to fill :
# title: string - A detailed title of the topic;
# description: string - A short description of the topic;
# summary: string - Multi section markdown report of the topic;
# references: ARRAY{{
#     title: string - A detailed title of the segment;
#     description: string - A short description of the segment and its relevance to the topic;
#     summary: In detail summary of all the fields ;
# }}"""

view_summary_prompt = """You are a helpful summarising assistant that can help with summarising the following text.
Please return a professional report on the topic."""


# Please return a professional report on the topic. 
