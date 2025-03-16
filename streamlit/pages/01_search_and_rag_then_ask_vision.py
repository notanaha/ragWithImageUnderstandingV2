import os, json
from IPython.display import Image
import openai
from azure.storage.blob import BlobServiceClient
import streamlit as st
from utilities import utils

#--------------------------------------#
# Set variables                        #
#--------------------------------------#
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
aoai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
aoai_embedding_model = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
aoai_chat_model = os.environ["AZURE_OPENAI_CHAT_MODEL"]

client = openai.AzureOpenAI(
    azure_endpoint=aoai_endpoint,
    api_key=aoai_api_key,
    api_version=aoai_api_version
)

connection_string = os.environ["STORAGE_CONN_STR"]
storage_sas_token = os.environ["STORAGE_SAS_TOKEN"] 
container_name = os.environ["CONTAINER_NAME"]
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

separator_word = os.environ["SEPARATOR_WORD"]

#--------------------------------------#
# main                                 #
#--------------------------------------#
def main():

    query = st.text_input("Enter your question:", value="ペットボトルの投棄方法は 1 から 9 のどれですか？")

    if st.button("Enter", key="confirm"):
        if query.lower() == "reset":
            for key in st.session_state.keys():
                del st.session_state[key]
            st.write("Conversation history has been reset.")
            return

        if 'messages' not in st.session_state:

            #--------------------------------------#
            # Retrieving answer from Search Index  #
            #--------------------------------------#
            st.write("Retrieving answer from Search Index.")
            
            answer_context = utils.search_index(query, client, aoai_embedding_model)

            #--------------------------------------#
            # Retrieving answer from gpt-4o        #
            #--------------------------------------#
            with open(os.path.join('utilities','system_message_01.txt'), "r", encoding = 'utf-8') as f:
                system_message = f.read()

            messages = [] 
            messages.append({"role": "system","content": system_message})

            content = {}
            content["question"] = query
            content["context"] = answer_context
            messages.append({"role": "user", "content": str(content)})

            st.write("Retrieving answer from gpt-4o")
            print("Retrieving answer from gpt-4o")

            response = utils.gpt4o_query(messages, client, aoai_chat_model)
            answer = response.choices[0].message.content

            answer = json.loads(answer)
            print("answer: ", answer)

            for num, item in enumerate(answer["answers"]):
                if not item["answer"].startswith("Sorry"):
                    st.write("answer["+ str(num) + "]: ", item["answer"])
                    st.write("  page["+ str(num) + "]: ", item["page"]) 
                    print("answer["+ str(num) + "]: ", item["answer"])
                    print("  page["+ str(num) + "]: ", item["page"])

            answer_string = ', '.join(json.dumps(item, ensure_ascii=False) for item in answer["answers"] \
                                    if not item["answer"].startswith("Sorry"))
             
            #--------------------------------------#
            # Get Image URL on Storage Account     #
            #--------------------------------------#

            image_urls = []
            for item in answer["answers"]:
                if not item["answer"].startswith("Sorry"):
                    blob_name = item["page"]
                    image_titles_and_urls = utils.list_blobs_titles_and_urls(blob_service_client, container_name, blob_name)
                    if image_titles_and_urls:
                        image_urls.append(image_titles_and_urls)

            if len(image_urls) == 0:
                print("No images found.")
                st.write("No images found.")
                return

            #--------------------------------------#
            # Retrieving answer from gpt-4o vision #
            #--------------------------------------#
            with open(os.path.join('utilities','system_message_02.txt'), "r", encoding = 'utf-8') as f:
                system_message = f.read()

            messages = []
            messages.append({"role": "system","content": system_message})

            content = []
            content.append({"type": "text", "text": "query: " + query})
            content.append({"type": "text", "text":"answer from gpt-4o: " + answer_string})

            for url in image_urls:    # pick up the relevant images
                storagepath = url['title']
                storagepath_stem = os.path.splitext(storagepath)[0]
                if not (storagepath_stem[-3] == separator_word and storagepath_stem[-2:].isdigit()):
                    storagepath_stem += separator_word + '01' # storagepath doesn't have a number suffix
                for item in answer["answers"]:
                    if storagepath_stem == item['page']:
                        content.append({"type": "text", "text": storagepath_stem})
                        content.append({"type": "image_url", "image_url": {"url": url['url']+storage_sas_token}})
                        break

            messages.append({"role": "user","content":content})

            st.write("Retrieving answer from gpt-4o vision")
            print("Retrieving answer from gpt-4o vision")

            response = utils.gpt4o_query(messages, client, aoai_chat_model)
            answer = response.choices[0].message.content
            answer = json.loads(answer)
            
            #st.write(json.dumps(answer, indent=4, ensure_ascii=False))
            st.write(answer["answer"])
            st.write(os.path.splitext(answer["citation"])[0])  
            print(answer)

            #--------------------------------------#
            # Download Images                      #
            #--------------------------------------#
            for url in image_urls:    # pick up the relevant images
                storagepath = url['title']                

                utils.list_blobs_download(blob_service_client, container_name, storagepath)
                st.write("\n",os.path.splitext(storagepath)[0])
                st.image(os.path.join("downloads", storagepath))

            #--------------------------------------#
            # Prepare for the next iteration       #
            #--------------------------------------#
            utils.append_conversation_history(messages, response, role="assistant")

            st.session_state['messages'] = messages
            st.session_state['storagepath'] = storagepath
            
        else:
            messages = st.session_state['messages']
            storagepath = st.session_state['storagepath']

            content = []

            content.append({"type": "text", "text": query})
            messages.append({"role": "user","content":content})

            st.write("Retrieving answer from gpt-4o vision")
            print("RRetrieving answer from gpt-4o vision")

            response = utils.gpt4o_query(messages, client, aoai_chat_model)
            answer = response.choices[0].message.content
            st.write(answer)
            print(answer)

            st.image(os.path.join("downloads", storagepath))

            utils.append_conversation_history(messages, response, role="assistant")
            st.session_state['messages'] = messages


if __name__ == '__main__':
    main()

