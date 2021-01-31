import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

context = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight " \
          "program carried out by the National Aeronautics and Space Administration (NASA), which accomplished " \
          "landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. " \
          "Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which " \
          "put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's " \
          "national goal of landing a man on the Moon and returning him safely to the Earth by the end of the " \
          "1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by " \
          "the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to " \
          "1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to " \
          "1966. Gemini missions developed some of the space travel techniques that were necessary for the " \
          "success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn " \
          "vehicles were also used for an Apollo Applications Program, which consisted of Skylab, " \
          "a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, " \
          "a joint Earth orbit mission with the Soviet Union in 1975. "

Q1 = "What project put the first Americans into space?"
Q2 = "What program was created to carry out these projects and missions?"
Q3 = "What year did the first manned Apollo flight occur?"
Q4 = "What President is credited with the original notion of putting Americans in space?"
Q5 = "Who did the U.S. collaborate with on an Earth orbit mission in 1975?"
Q6 = "How long did Project Apollo run?"
Q7 = "What program helped develop space travel techniques that Project Apollo used?"
Q8 = "What space station supported three manned missions in 1973-1974?"
questions = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]

gpu = torch.device('cuda')
model = T5ForConditionalGeneration.from_pretrained('t5-large').to(gpu)
tokenizer = T5Tokenizer.from_pretrained('t5-large')


# Q: What project put the first Americans into space?
# A: Project Mercury. Apollo program, also known as Project Apollo

# Q: What program was created to carry out these projects and missions?
# A: Gemini program which ran concurrently with it from 1962 to 1966

# Q: What year did the first manned Apollo flight occur?
# A: 1968. Apollo ran from 1961 to 1972,

# Q: What President is credited with the original notion of putting Americans in space?
# A: Dwight D. Eisenhower's

# Q: Who did the U.S. collaborate with on an Earth orbit mission in 1975?
# A: Soviet Union in 1975. Apollo-Soyuz Test Project

# Q: How long did Project Apollo run?
# A: 1961 to 1972, and was supported by the two-man Gemini program

# Q: What program helped develop space travel techniques that Project Apollo used?
# A: Gemini missions developed some of the space travel techniques that were necessary
# for the success of the Apollo missions

# Q: What space station supported three manned missions in 1973-1974?
# A: Skylab, a space station that supported three manned missions in 1973-74
for question in questions:
    tokens_input = tokenizer.encode(text="question: " + question + " context: " + context, return_tensors="pt",
                                    max_length=1024, truncation=True)
    summary_ids = model.generate(tokens_input.to(gpu), min_length=10, max_length=100, length_penalty=4.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Q:", question)
    print("A:", summary)
    print()
