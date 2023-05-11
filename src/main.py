import os
from functools import lru_cache  # removesleastRecentlyUsedCache
from pathlib import Path

import gradio as gr
import openai
from data import Pipeline, information


def image_classifier(image: str):
    # Actual prediction pipeline
    # refer: ./data.py
    prediction = Pipeline(
        image=image
    ).predict()  # opens to data.py which consists the pipeline function which performs preprocess and send image tp predict
    leaf_details = chatGPT(
        leaf_name=prediction
    )  # gets the leafs details from chatGPT api
    return prediction, leaf_details


def load_env(root_path: str = "./.env"):
    # Decent alternative for load env (excess deps)
    root_path = (
        os.path.join(Path(__file__).parent.parent.resolve(), ".env")
        if not root_path
        else root_path
    )
    with open(root_path, "r") as f:
        env_data = f.read().strip().replace(" ", "").split()
        for data in env_data:
            key, value = data.split("=")
            os.environ[key] = value


@lru_cache()
def chatGPT(leaf_name: str):
    # Use chatGPT to fetch information about the leaf.

    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You need to give me medicinal information about leaves,\
                    be short and crisp and only give medicinal information.",
                },
                {
                    "role": "user",
                    "content": f"Give me information about this leaf: {leaf_name}",
                },
            ],
        )
    except Exception:
        return information[leaf_name]

    return response["choices"][0]["message"]["content"]


def load_css(css: str = "./static/style.css"):
    # Edit css file for any changes in css.

    with open(css, "r") as f:
        css = f.read().strip()

    return css


with gr.Blocks(title="Major Project", css=load_css()) as main:
    # Main app handler uses FastAPI under the hood.
    gr.Markdown(
        """
    # Medicinal Plant Classfier
    #### Traditional Indian medicinal plants are a rich source of many vital nutrients in accessible forms, which are utilized to boost the body's immune system. However, the majority of individuals are unable to recognise these plants. The main goal of this is to educate people about the identification and medical applications of various herbs, some of which may be readily available. Normally, the healing properties of these plants would go unutilized and unnoticed. Taking that note into consideration, the aim and purpose of Medicinal Plant Classifier is to provide the users with an interface which can help them identify which plant is it and what are it medicinal benefits.
    #### Input the Plant Leaf Image below:
     """
    )

    with gr.Tabs() as tabs:
        with gr.TabItem("Classifier"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="filepath", label="Input Image")
                    predict_button = gr.Button("Predict")
                with gr.Column():
                    classification = gr.TextArea(
                        label="Classification",
                        placeholder="Classification appears hear on submit.",
                    )

        with gr.TabItem("Medicinal Properties"):
            details = gr.TextArea(label="Details")

    predict_button.click(
        fn=image_classifier,
        inputs=inp,
        outputs=[classification, details],
    )


load_env()
gr.close_all()

main.launch(
    show_api=False,
    debug=False,
    show_error=False,
    server_name="0.0.0.0",
    share=False,
    server_port=8000,
)
