import streamlit as st
from huggingface_hub import InferenceApi
from annotated_text import annotated_text
import nltk
import spacy
from transformers import pipeline

def extract_software(model,message):
    if model=="softcite":
        results = softcite_pipeline(message)
    elif model=="somesci":
        results = somesci_pipeline(message)
    elif model=="benchmark":
        results = benchmark_pipeline(message)
    elif model=="benchmark_multidomain":
        results = benchmark_pipeline(message)

    print("***************************************************************************")
    print("Model:"+model+" Prediction: "+str(results))
    
    return results

def getEntityToken(token, predictions):

    for prediction in predictions:
        if token.idx == prediction["start"] and len(token.text) == len(prediction["word"]):
            return prediction["entity_group"]
    
    return ""

def annotate_text(text, predictions):

    
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(text)

    tokens = [token.text+" " for token in docx]

    res = []

    for token in docx:
        entity = getEntityToken(token, predictions)
        if entity != "":
            res.append((token.text+" ",entity))
        else:
            res.append(token.text+" ")

    #print(res)

    annotated_text(res)

def a_text(text, predictions):
    print("PRED:"+str(predictions))
    annotated_results = []
    #res = [token["software-name"]["rawForm"] for token in predictions["mentions"]]
    
    last_end = 0
    for token in predictions:
        ent_text = token["word"]
        ent_label = token["entity_group"]
        start = token["start"]
        end = token["end"]
        if start > last_end:
            annotated_results.append(text[last_end:start])
        annotated_results.append((text[start:end], ent_label))
        last_end = end
    annotated_results.append(text[last_end:])
    annotated_text(*annotated_results)

def main():
    st.title("Software Mention Benchmark")

    text_container = st.container()

    text_container.markdown("This demo extract software mentions from a text. It uses fine-tuned models from SCIBERT using different corpora.") 
    text_container.markdown("* SoMESCi [1]: We have used the corpus uploaded to [Github](https://github.com/dave-s477/SoMeSci/tree/9f17a43f342be026f97f03749457d4abb1b01dbf/PLoS_sentences), more specifically, the corpus created with sentences.")
    text_container.markdown("* Softcite [2]: This project has published another corpus for software mentions, which is also available on [Github](https://github.com/howisonlab/softcite-dataset/tree/master/data/corpus). We have to note that we only use the annotations from bio domain.")
    text_container.markdown("* [Benchmark Bio](https://huggingface.co/oeg/software_benchmark_bio): Corpus created by reconciling the somesci and softcite corpora. Only BIO domain")
    text_container.markdown("* [Benchmark Multidomain](https://huggingface.co/oeg/software_benchmark_multidomain): Corpus created by reconciling the somesci and softcite corpora (including economics domain) and papers with code [corpus](https://doi.org/10.5281/zenodo.10033751).")

    text_container.markdown("To try the demo, please enter your own text in the box below  and then click on \"Analyze\". Software mentions in each model will be highlighted.")
    
    text = st.text_area("Enter your text","Type here (longer sentences are recommended so the model can pick up the right context)", key="sentence_text")

    if st.button("Analyze"):
               
        st.subheader("SOMESCI")
        nlp_result = extract_software("somesci",text)
        a_text(text, nlp_result)
        
        st.subheader("SOFTCITE")
        nlp_result = extract_software("softcite",text)
        a_text(text, nlp_result)
        
        st.subheader("BENCHMARK BIO")
        nlp_result = extract_software("benchmark",text)
        a_text(text, nlp_result)

        st.subheader("BENCHMARK MULTIDOMAIN")
        nlp_result = extract_software("benchmark_multidomain",text)
        a_text(text, nlp_result)

    st.subheader("References")

    st.markdown("1. Schindler, D., Bensmann, F., Dietze, S., & Krüger, F. (2021, October). Somesci-A 5 star open data gold standard knowledge graph of software mentions in scientific articles. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 4574-4583).")
    st.markdown("2. Du, C., Cohoon, J., Lopez, P., & Howison, J. (2021). Softcite dataset: A dataset of software mentions in biomedical and economic research publications. Journal of the Association for Information Science and Technology, 72(7), 870-884.")

    st.markdown("---")
    col_about, col_figures = st.columns([2,1])
    col_about.markdown("Daniel Garijo and Esteban Gonzalez")
    col_about.markdown("Version: 0.0.1")
    col_about.markdown("Last revision: October, 2023")
    col_about.markdown("Github: <https://github.com/oeg-upm/software_mentions_benchmark>")
    col_about.markdown("Built with [streamlit](https://streamlit.io/)")

    logo_oeg, logo_upm = col_figures.columns(2)
        
    logo_oeg.image("images/logo-oeg.gif", width=100)
    logo_upm.image("images/upmlogo.png", width=100)
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
        


if __name__ == '__main__':
    softcite_pipeline= pipeline('ner', model='models/softcite',aggregation_strategy='average')
    somesci_pipeline = pipeline('ner', model='models/somesci',aggregation_strategy='average')
    benchmark_pipeline = pipeline('ner', model='models/benchmark',aggregation_strategy='average')
    benchmark_multidomain_pipeline = pipeline('ner', model='models/benchmark_multidomain',aggregation_strategy='average')
    main()

