import streamlit as st
from question import *
from exercise import *
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Coding Room",
                   page_icon="📚",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title("🤖 PyEx")
st.info("AI-powered exercise generation that accelerates your programming journey! 🚀")
st.divider()

st.sidebar.header("Please select the type of exercises")

language = st.sidebar.selectbox(label='Programming Language',
                                options=['python'])

difficulty = st.sidebar.selectbox(label='Difficulty',
                                  options=['Easy', 'Medium', 'Hard'])

topic = st.sidebar.selectbox(label='Programming Topic',
                             options=get_topics())

context = st.sidebar.text_input(label='New Question Context',
                                help="The context in which the question is asked")

num_ref_exercises = st.sidebar.slider(label='No. Reference Exercises',
                                      help="Number of similar topic exercises to refer to",
                                      min_value=1,
                                      max_value=4,
                                      step=1)

if st.sidebar.button(label="Generate"):
    dataset_path = generate_sample_question(language=language,
                                            difficulty=difficulty,
                                            topic=topic)
    sample_questions = select_random_n_questions(dataset_path=dataset_path, n=num_ref_exercises)
    prompt = create_exercise_prompt(sample_questions=sample_questions,
                                    topic=topic)
    model_path = f"{os.getcwd()}/llama.cpp/models/7b/ggml-model-q4_0.bin"
    llm = get_llm(model_path=model_path)
    exercise_parser = get_parser(Exercise)
    exercise_llm_chain = get_llm_chain(llm, prompt, exercise_parser)
    exercise_generate_prompt = f"Generate python coding exercise according to above format, under the context of {context}. The problem statement must contains the {context} keywords."
    response = exercise_llm_chain(exercise_generate_prompt)

    logging.info(response)
    if 'text' in response:
        exercise_dict = parse_response(response['text'], exercise_parser, llm, exercise_generate_prompt)
    else:
        exercise_dict = parse_response(response, exercise_parser, llm, exercise_generate_prompt)

    if exercise_dict:
        explanation_prompt = create_code_explanation_prompt(generated_question=exercise_dict['problem_statement'],
                                                            code=exercise_dict['solution'])
        explanation_llm_chain = get_llm_chain(llm, explanation_prompt, None)
        explanation_generate_prompt = f"Generate explanation for the above code"
        explanation_response = explanation_llm_chain(explanation_generate_prompt)
        logging.info(explanation_response)

    st.subheader("AI-generated Programming Exercise")

    with st.expander(exercise_dict['title']):
        problem_statement_tab, sample_solution_tab, explanation_tab = st.tabs(['Problem Statement', 'Code Hint', 'Code Explanation'])
        with problem_statement_tab:
            st.markdown(body=exercise_dict['problem_statement'])
            st.divider()
            st.markdown(f"*Topic: {exercise_dict['topic']}*")

        with sample_solution_tab:
            st.code(body=exercise_dict['solution'], language='markdown')

        with explanation_tab:
            st.markdown(body=explanation_response['text'])
