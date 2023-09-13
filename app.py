import streamlit as st
from question import *
from exercise import *
from feedback import *
import logging
from langchain import callbacks

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Coding Room",
                   page_icon="📚",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)


@st.cache_data(show_spinner="Generating exercise ...")
def get_exercise(_exercise_parser, _exercise_llm_chain):

    exercise_generate_prompt = f"Generate python coding exercise according to above format, under the context of {context}. The problem statement must contain the {context} keywords."
    exercise_generate_metadata = {
                                    "metadata": {
                                        "type": "exercise_generator"
                                    }
                                 }

    with callbacks.collect_runs() as cb:
        exercise_generate_response = _exercise_llm_chain.invoke({"question": exercise_generate_prompt}, exercise_generate_metadata)
        exercise_chain_run_id = cb.traced_runs[-1].id
        logging.info(exercise_chain_run_id)

    logging.info(exercise_generate_response)
    if 'text' in exercise_generate_response:
        exercise_dict = parse_response(exercise_generate_response['text'], _exercise_parser, llm, exercise_generate_prompt)
    else:
        exercise_dict = parse_response(exercise_generate_response, _exercise_parser, llm, exercise_generate_prompt)

    return exercise_dict, exercise_chain_run_id

@st.cache_data(show_spinner="Please hang tight ...")
def get_explanation(exercise_dict):
    explanation_prompt = create_code_explanation_prompt(generated_question=exercise_dict['problem_statement'],
                                                            code=exercise_dict['solution'])
    explanation_llm_chain = get_llm_chain(llm, explanation_prompt, None, tag=os.getenv('ENV_TAG', 'test-run'))
    explanation_generate_prompt = f"Generate explanation for the above code"
    explanation_generate_metadata = {
                                        "metadata": {
                                            "type": "explanation_generator"
                                        }
                                    }

    with callbacks.collect_runs() as cb:
        explanation_response = explanation_llm_chain.invoke({"question": explanation_generate_prompt}, explanation_generate_metadata)
        explanation_chain_run_id = cb.traced_runs[-1].id
        logging.info(explanation_chain_run_id)
    logging.info(explanation_response)
    return explanation_response, explanation_chain_run_id


st.title("🤖 PyEx")
st.info("AI-powered exercise generation that accelerates your programming journey! 🚀")

st.sidebar.header("Programming Exercise Generator")

language = st.sidebar.selectbox(label='Programming Language',
                                options=['python'])

difficulty = st.sidebar.selectbox(label='Difficulty',
                                  options=['Easy', 'Medium', 'Hard'])

topic = st.sidebar.selectbox(label='Programming Topic',
                             options=get_topics())

context = st.sidebar.text_input(label='New Question Keyword Context',
                                help="The context under which the question will be formed.",
                                placeholder="cars/balloons/trains")

num_ref_exercises = st.sidebar.slider(label='No. Reference Exercises',
                                      help="Number of similar topic exercises to refer to",
                                      min_value=1,
                                      max_value=4,
                                      step=1)


generate_btn = st.sidebar.button(label="Generate")

if generate_btn or 'feedback_state' in st.session_state:

    if generate_btn:
        st.cache_data.clear()

    dataset_path = generate_sample_question(language=language,
                                            difficulty=difficulty,
                                            topic=topic)
    sample_questions = select_random_n_questions(dataset_path=dataset_path, n=num_ref_exercises)
    prompt = create_exercise_prompt(sample_questions=sample_questions,
                                    topic=topic)
    logging.info(prompt)
    model_path = f"{os.getcwd()}/llama.cpp/models/7b/ggml-model-q4_0.bin"
    llm = get_llm(model_path=model_path, tag="test-run")
    # exercise_parser = get_parser(Exercise)
    # exercise_llm_chain = get_llm_chain(llm, prompt, exercise_parser, tag=os.getenv('ENV_TAG', 'test-run'))
    # exercise_generate_prompt = f"Generate python coding exercise according to above format, under the context of {context}. The problem statement must contains the {context} keywords."
    # exercise_generate_metadata = {
    #                                 "metadata": {
    #                                     "type": "exercise_generator"
    #                                 }
    #                              }
    # exercise_generate_response = exercise_llm_chain(exercise_generate_prompt)

    # with callbacks.collect_runs() as cb:
    #     exercise_generate_response = exercise_llm_chain.invoke({"question": exercise_generate_prompt}, exercise_generate_metadata)
    #     exercise_chain_run_id = cb.traced_runs[-1].id
    #     logging.info(exercise_chain_run_id)

    # logging.info(exercise_generate_response)
    # if 'text' in exercise_generate_response:
    #     exercise_dict = parse_response(exercise_generate_response['text'], exercise_parser, llm, exercise_generate_prompt)
    # else:
    #     exercise_dict = parse_response(exercise_generate_response, exercise_parser, llm, exercise_generate_prompt)

    exercise_parser = get_parser(Exercise)
    exercise_llm_chain = get_llm_chain(llm, prompt, exercise_parser, tag=os.getenv('ENV_TAG', 'test-run'))
    exercise_dict, exercise_chain_run_id = get_exercise(exercise_parser, exercise_llm_chain)

    if exercise_dict:
        # explanation_prompt = create_code_explanation_prompt(generated_question=exercise_dict['problem_statement'],
        #                                                     code=exercise_dict['solution'])
        # explanation_llm_chain = get_llm_chain(llm, explanation_prompt, None, tag=os.getenv('ENV_TAG', 'test-run'))
        # explanation_generate_prompt = f"Generate explanation for the above code"
        # explanation_generate_metadata = {
        #                                     "metadata": {
        #                                         "type": "explanation_generator"
        #                                     }
        #                                 }
        # explanation_response = explanation_llm_chain(explanation_generate_prompt)
        # with callbacks.collect_runs() as cb:
        #     explanation_response = explanation_llm_chain.invoke({"question": explanation_generate_prompt}, explanation_generate_metadata)
        #     explanation_chain_run_id = cb.traced_runs[-1].id
        #     logging.info(explanation_chain_run_id)

        # logging.info(explanation_response)

        explanation_response, explanation_chain_run_id = get_explanation(exercise_dict)

    st.subheader(body="AI-generated Programming Exercise",
                 divider="rainbow")

    with st.expander(exercise_dict['title']):
        problem_statement_tab, explanation_tab, feedback_tab = st.tabs(['❓ Problem Statement', '💡 Code Hint Explanation', '❤️ Feedback'])
        with problem_statement_tab:
            st.markdown(body=exercise_dict['problem_statement'])
            st.divider()
            st.markdown(f"*Topic: {exercise_dict['topic']}*")

        with explanation_tab:
            st.code(body=exercise_dict['solution'], language='markdown')
            st.markdown(body=explanation_response['text'])

        with feedback_tab:
            st.session_state['feedback_state'] = True
            exercise_comment = st.text_input(label='Comments on the exercise generation')
            exercise_accuracy_score = st.slider(label='Accuracy score on the exercise',
                              min_value=0.0,
                              max_value=1.0,
                              step=0.1,
                              value=0.5,
                              help="0.0 is the least accurate and 1.0 is the most accurate")
            exercise_correction = st.text_input(label='Additional correction on the exercise generation')


            explanation_comment = st.text_input(label='Comments on the code explanation')
            explanation_accuracy_score = st.slider(label='Accuracy score on the explanation',
                              min_value=0.0,
                              max_value=1.0,
                              step=0.1,
                              value=0.5,
                              help="0.0 is the least accurate and 1.0 is the most accurate")
            explanation_correction = st.text_input(label='Additional correction on the explanation response')

            if st.button(label="Submit"):
                send_comment(run_id=exercise_chain_run_id,
                            comment=exercise_comment,
                            score=exercise_accuracy_score,
                            correction={"additional_correction": exercise_correction})

                send_comment(run_id=explanation_chain_run_id,
                            comment=explanation_comment,
                            score=explanation_accuracy_score,
                            correction={"additional_correction": explanation_correction})

                st.toast(body='Thank you for your input!', icon='✅')
