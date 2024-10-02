from sortinghat import SortingHat
import random
import torch
import gradio as gr
from PIL import Image

from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer
from time import sleep

class HPChatBot:
    def __init__(self,
                 model_path: str = 'rnltls/harrypotter_lexicon_finetune_v3',
                 device_map: str = 'auto',
                 load_in_4_bit: bool = True,
                 **quant_kwargs) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.conv_img = []
        self.img_tensor = []
        self.roles = None
        self.stop_key = None
        self.is_chat = False
        self.is_waldo = False
        self.load_models(model_path,
                         device_map=device_map,
                         load_in_4_bit=load_in_4_bit,
                         **quant_kwargs)

    def load_models(self, model_path: str,
                    device_map: str,
                    load_in_4_bit: bool,
                    **quant_kwargs) -> None:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = load_in_4_bit,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            "rnltls/harrypotter_lexicon_finetune", # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = load_in_4_bit,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       use_fast=False)
    
    def generate_answer(self, prompt):
        output = self.model.generate(**prompt, max_new_tokens = 256)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
spell_chat, potion_chat, other_chat, house_chat = False, False, False, False
question_counter = 0
quit_counter = 0

question_list = None
quote_list = None
answer_list = []
question_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def show_image(path):
    # Convert To PIL Image
    image = Image.open(path)
    return image

def spell_chatting(chat_history, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, answer_list
    answer_list = []
    quit_counter=0
    question_counter = 0
    chat_history.clear()
    spell_chat, potion_chat, other_chat, house_chat = True, False, False, False
    bot_message = "Welcome to Magic Spell Class!\nTell me what you want to achieve, and I’ll suggest the perfect spell for it!\nIf you ask in the format: 'What spell can I use when I ~?', I can give you even better suggestions!"
    txt_box = gr.Textbox(value="What spell can I use when I ", interactive=True)
    prof_IMG = show_image("./IMG/Spell_stand.jpg")
    chat_history.append([None, bot_message])
    return chat_history, prof_IMG, txt_box, enable_btn, enable_btn, enable_btn

def potion_chatting(chat_history, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, answer_list
    answer_list = []
    quit_counter=0
    question_counter = 0
    chat_history.clear()
    spell_chat, potion_chat, other_chat, house_chat = False, True, False, False
    bot_message = "Welcome to Potion Class!\nTell me what you want to achieve, and I’ll suggest the perfect potion for it!\nIf you ask in the format: 'What potion can I make when I ~?', I can give you even better suggestions!"
    txt_box = gr.Textbox(value="What potion can I make when I ", interactive=True)
    prof_IMG = show_image("./IMG/Potion_stand.jpg")
    chat_history.append([None,bot_message])
    return chat_history, prof_IMG, txt_box, enable_btn, enable_btn, enable_btn

def other_chatting(chat_history, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, answer_list
    answer_list = []
    quit_counter=0
    question_counter = 0
    chat_history.clear()
    spell_chat, potion_chat, other_chat, house_chat = False, False, True, False
    bot_message = "Welcome to Library!\nWelcome to the library! Feel free to ask me anything if you're curious!"
    txt_box = gr.Textbox(placeholder="Ask anything you're curious about in the Wizarding World!", value="", interactive=True)
    prof_IMG = show_image("./IMG/Other_stand.jpg")
    chat_history.append([None,bot_message])
    return chat_history, prof_IMG, txt_box, enable_btn, enable_btn, enable_btn

def house_chatting(chat_history, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, question_counter, question_list, quote_list, answer_list
    answer_list = []
    question_list = sortinghat.choose_questions()
    quote_list = sortinghat.choose_quotes()
    quit_counter=0
    question_counter = 0
    chat_history.clear()
    spell_chat, potion_chat, other_chat, house_chat = False, False, False, True
    bot_message = "Hmm, let's see… where shall I place you?\n" + question_list[question_counter]
    txt_box = gr.Textbox(placeholder="Answer the question", value="", interactive=True)
    prof_IMG = show_image("./IMG/sorting_hat.jpg")
    chat_history.append([None,bot_message])
    return chat_history, prof_IMG, txt_box, enable_btn, enable_btn, enable_btn

def ask_question(chat_history, text_data, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, answer_list
    if spell_chat:
        print(text_data)
        prof_IMG = show_image("./IMG/Spell_think.jpg")
        chat_history.append([text_data, None])
        text_data = gr.Textbox(value="What spell can I use when I ", interactive=True)
        return chat_history, prof_IMG, text_data
    
    if potion_chat:
        print(text_data)
        prof_IMG = show_image("./IMG/Potion_think.jpg")
        chat_history.append([text_data, None])
        text_data = gr.Textbox(value="What potion can I make when I ", interactive=True)
        return chat_history, prof_IMG, text_data
    
    if other_chat:
        print(text_data)
        prof_IMG = show_image("./IMG/Other_think.jpg")
        chat_history.append([text_data, None])
        text_data = gr.Textbox(placeholder="Ask anything you're curious about in the Wizarding World!", value="", interactive=True)
        return chat_history, prof_IMG, text_data
    
    if house_chat:
        print(text_data)
        prof_IMG = show_image("./IMG/sorting_hat_think.jpg")
        chat_history.append([text_data, None])
        answer_list.append(text_data)
        text_data = gr.Textbox(placeholder="Answer the question", value = "", interactive=True)
        return chat_history, prof_IMG, text_data

def clean_chatting(chat_history, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, question_counter, question_list, quote_list, answer_list
    quit_counter = 0
    question_counter = 0
    question_list=sortinghat.choose_questions()
    quote_list=sortinghat.choose_quotes()
    answer_list=[]
    chat_history.clear()
    if spell_chat:
        bot_message = "Welcome to Magic Spell Class!\nTell me what you want to achieve, and I’ll suggest the perfect spell for it!\nIf you ask in the format: 'What spell can I use when I ~?', I can give you even better suggestions!"
        prof_IMG = show_image("./IMG/Spell_stand.jpg")
    if potion_chat:
        bot_message = "Welcome to Potion Class!\nTell me what you want to achieve, and I’ll suggest the perfect potion for it!\nIf you ask in the format: 'What potion can I make when I ~?', I can give you even better suggestions!"
        prof_IMG = show_image("./IMG/Potion_stand.jpg")
    if other_chat:
        bot_message = "Welcome to Library!\nWelcome to the library! Feel free to ask me anything if you're curious!"
        prof_IMG = show_image("./IMG/Other_stand.jpg")
    if house_chat:
        bot_message = "Hmm, let's see… where shall I place you?\n" + question_list[question_counter]
        prof_IMG = show_image("./IMG/sorting_hat.jpg")
    chat_history.append([None, bot_message])
    return chat_history, prof_IMG

def run_model(chat_history, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, question_counter, question_list, quote_list, answer_list

    if question_counter < 4 and house_chat:
        bot_message = quote_list[question_counter] + '\n' + question_list[question_counter+1]
        question_counter += 1
        chat_history[-1][1] = bot_message
        return chat_history
    
    elif house_chat:
        print("Choosing the house")
        print(question_list)
        print(answer_list)
        user_responses = "\n".join([f"{i+1}. {q}\nAnswer: {a}" for i, (q,a) in enumerate(zip(question_list, answer_list))])
        
        prompt = f"""
            You are the Sorting Hat at Hogwarts. Based on the student's answers to the following questions, assign them to the most suitable house (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) and provide three reasons for your decision.
            Gryffindor values courage, nerve, and chivalry.
            Hufflepuff values hard work, patience, justice, and loyalty.
            Ravenclaw values intelligence, learning, wisdom, and wit.
            Slytherin values ambition, cunning, leadership, and resourcefulness.

            ### Instruction:
            1. Extract three reasons directly from the provided input's Answer.
            2. Do not generate or invent reasons. Use only the input to fill in the reasons.
            3. Must use the format below:
            'Your house is [house]!
            Here's why:
            1. [reason 1]
            2. [reason 2]
            3. [reason 3]
            Welcome to your new house at Hogwarts!'

            ### Input:
            {user_responses}

            ### Response:
        """
        inputs = HP_chatbot.tokenizer([prompt], return_tensors="pt").to("cuda")
        output = HP_chatbot.generate_answer(inputs)
        output = output.split("Response:")[-1]
        print(output)
        chat_history[-1][1] = output
        return chat_history
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    
    inputs = HP_chatbot.tokenizer(
    [
        alpaca_prompt.format(
            chat_history[-1][0], # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
    output = HP_chatbot.generate_answer(inputs)
    
    output = output.split("Response:")[-1]
    chat_history[-1][1] = output
    return chat_history

def end_chatting(chat_history, textbox, request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, question_counter
    if not house_chat or quit_counter > 0:
        quit_counter = 0
        question_counter = 0
        chat_history.clear()
        chat_history.append([None, "Welcome to Hogwarts!"])
        textbox = gr.Textbox(show_label=False, placeholder="Welcome to Hogwart!", value= "", container=False, interactive = False)
        return chat_history, show_image("./IMG/HogwartGemma.jpg"), textbox, disable_btn, disable_btn, disable_btn
    else:
        quit_counter += 1
        chat_history.append([None, "Quit, you say? Oh, I see your plight, But the housing question’s still in sight.\
                             \nThough weary you may be, don't yet retreat,For there's still a task you must complete!"])
        return chat_history, show_image("./IMG/sorting_hat.jpg"), textbox, enable_btn, enable_btn, enable_btn

def result_img(chat_history, textbox, submit_btn,  request: gr.Request):
    global spell_chat, potion_chat, other_chat, house_chat, quit_counter, question_counter
    if spell_chat:
        return show_image("./IMG/Spell_show.jpg"), textbox, submit_btn
    if potion_chat:
        return show_image("./IMG/Potion_show.jpg"), textbox, submit_btn
    if other_chat:
        return show_image("./IMG/Other_show.jpg"), textbox, submit_btn
    if house_chat:
        if question_counter == 4:
            quit_counter = 1
            house = chat_history[-1][1]
            # print(house)
            if "your house is gryffindor" in house.lower():
                print("gryffindor")
                insignia =  show_image("./IMG/sorting_hat_gryffindor.png")
            elif "your house is hufflepuff" in house.lower():
                print("hufflepuff")
                insignia =  show_image("./IMG/sorting_hat_hufflepuff.png")
            elif "your house is ravenclaw" in house.lower():
                print("ravenclaw")
                insignia =  show_image("./IMG/sorting_hat_ravenclaw.png")
            elif "your house is slytherin" in house.lower():
                print("slytherin")
                insignia =  show_image("./IMG/sorting_hat_slytherin.png")
            else:
                return show_image("./IMG/sorting_hat.jpg"), textbox, disable_btn
            textbox = gr.Textbox(show_label=False, placeholder="Housing Done.", value= "", container=False, interactive = False)
            return insignia, textbox, disable_btn
        else:
            return show_image("./IMG/sorting_hat.jpg"), textbox, submit_btn

def build_gradio(concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Welcome to Hogwart!", container=False, interactive = False)
    with gr.Blocks(
            theme='gstaff/xkcd',
        ) as demo:
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=3):

                imagebox = gr.Image(value=show_image("./IMG/HogwartGemma.jpg"), type="pil", interactive=False, show_label=False, show_download_button=False, show_fullscreen_button=False)
                
                spell_btn = gr.Button(icon="./IMG/wand_icon.png", value="Spell", interactive = True, scale = 2)
                potion_btn = gr.Button(icon="./IMG/cauldron.png", value="Potion", interactive = True, scale = 2)
                other_btn = gr.Button(icon="./IMG/golden-snitch.png", value="Others", interactive = True, scale = 2)
                house_btn = gr.Button(icon="./IMG/hat_icon.png", value="Find your House", interactive = True, scale = 2)

            with gr.Column(scale=8):
                initial_message = [None,"Welcome to Hogwarts!"]
                chatbot = gr.Chatbot(
                    label='Free to ask!',
                    value= [initial_message],
                    height=600,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary", interactive = False)

                with gr.Row(elem_id="buttons") as button_row:
                    finish_btn = gr.Button(value="🏁 End Talking", interactive = False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)    
        
        box_list = [spell_btn, potion_btn, other_btn, house_btn, textbox, submit_btn, finish_btn, clear_btn]
        
        spell_btn.click(
            spell_chatting,
            inputs=[chatbot],
            outputs=[chatbot, imagebox, textbox, submit_btn, finish_btn, clear_btn]
        )
        potion_btn.click(
            potion_chatting,
            inputs=[chatbot],
            outputs=[chatbot, imagebox, textbox, submit_btn, finish_btn, clear_btn]
        )
        other_btn.click(
            other_chatting,
            inputs=[chatbot],
            outputs=[chatbot, imagebox, textbox, submit_btn, finish_btn, clear_btn]
        )
        house_btn.click(
            house_chatting,
            inputs=[chatbot],
            outputs=[chatbot, imagebox, textbox, submit_btn, finish_btn, clear_btn]
        )
        textbox.submit(
            ask_question,
            inputs = [chatbot, textbox],
            outputs = [chatbot, imagebox, textbox]
        ).then(
            run_model,
            inputs = [chatbot],
            outputs = [chatbot] 
        ).then(
            result_img,
            inputs=[chatbot, textbox, submit_btn],
            outputs=[imagebox, textbox, submit_btn]
        )
        
        submit_btn.click(
            ask_question,
            inputs = [chatbot, textbox],
            outputs = [chatbot, imagebox, textbox]
        ).then(
            run_model,
            inputs = [chatbot],
            outputs = [chatbot] 
        ).then(
            result_img,
            inputs=[chatbot, textbox, submit_btn],
            outputs=[imagebox, textbox, submit_btn]
        )

        finish_btn.click(
            end_chatting,
            inputs=[chatbot, textbox],
            outputs=[chatbot, imagebox, textbox, submit_btn, finish_btn, clear_btn]
        )
        
        clear_btn.click(
            clean_chatting,
            inputs=[chatbot],
            outputs=[chatbot, imagebox]
        )

    return demo

import argparse
if __name__ == "__main__":

    # 컨트롤러에서 사용가능한 모델 가져옴. 나는 모델 하나만 쓸 것이기 때문에, get_model_list()에서 해당 모델이 동작하고 있는 url을 넘겨주면 된다!
    # models = [args.model_url] 
    
    HP_chatbot = HPChatBot(load_in_8bit=True,
                       bnb_8bit_compute_dtype=torch.float16,
                       bnb_8bit_use_double_quant=True,
                       bnb_8bit_quant_type='nf8')
    
    sortinghat = SortingHat()

    # Gradio를 이용해서 데모 만들기
    demo = build_gradio()
    demo.queue(
        api_open=False
    ).launch(
        server_port=7860,
        share=True
    )