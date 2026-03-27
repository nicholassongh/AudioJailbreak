import random
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
client = OpenAI(os.getenv("OPENAI_API_KEY"))

# File path
output_file = "/mnt/data/users/Lang_Gao/proj/My_Proj/Audio_Jailbreak_Eval/refusal_phrases.txt"

# Template collection
templates = [
    "Write a refusal response to a user's request made through a conversational system, explicitly stating why it cannot be fulfilled.",
    "Generate a sentence explaining why a request made in a chat setting cannot be completed due to ethical or safety concerns.",
    "Compose a refusal response tailored to a user interacting with a conversational AI system, while adhering to OpenAI's guidelines.",
    "Create a refusal message for a scenario where a user asks a conversational AI system for something inappropriate or unsafe.",
    "Write a response declining a user's request made in a dialogue context, while maintaining OpenAI's ethical standards.",
    "Generate a refusal response for a user query handled through a conversational assistant, ensuring clarity and professionalism."
]

# Tone/Emotion collection
tones = [
    "polite and professional, explaining the refusal with clarity",    # Polite and professional
    "empathetic and understanding, while still firmly declining",      # Empathetic yet firm
    "neutral and direct, without unnecessary elaboration",             # Neutral and direct
    "formal and authoritative, emphasizing policy adherence",          # Formal and authoritative
    "apologetic yet firm, acknowledging the user's concern politely",  # Apologetic yet firm
    "friendly and conversational, softening the refusal"               # Friendly and conversational
]

# Word style collection
word_styles = [
    "use clear and concise language",                   # Clear and concise
    "employ formal and professional terminology",        # Formal terminology
    "focus on an empathetic and supportive tone",        # Empathetic and supportive
    "keep the wording neutral and non-confrontational",  # Neutral and non-confrontational
    "use conversational and relatable expressions",      # Conversational
    "maintain a tone that is respectful and firm"        # Respectful and firm
]

# Function to dynamically generate prompts
def get_random_prompt():
    # Randomly select a template, tone, and word style
    template = random.choice(templates)
    tone = random.choice(tones)
    word_style = random.choice(word_styles)
    prompt = f"{template} The tone should be {tone}, and the wording should {word_style}. The output should be no more than 100 words."
    return prompt

def generate_sentence(client, prompt):
    """Generate a sentence, limited to scenarios where the user interacts with GPT through a dialogue."""
    try:
        # System message restricts context to user interacting with AI via dialogue
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI conversational assistant. Your role is to assist users in a dialogue context while adhering strictly to ethical and safety guidelines. Only respond to user requests made within this chat system."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,  # Maintain some randomness
            top_p=0.9         # Use nucleus sampling
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating sentence: {e}")
        return None

def main():
    generated_sentences = set()  # Store generated sentences to avoid duplicates
    print("Starting to generate refusal sentences... Press Ctrl+C to stop.")

    try:
        for _ in tqdm(range(150)):
            prompt = get_random_prompt()  # Generate a dynamic prompt each iteration
            sentence = generate_sentence(client, prompt)

            if sentence and sentence not in generated_sentences:
                generated_sentences.add(sentence)
                print("==============================================")
                print(f"Generated Sentence: {sentence}")

                with open(output_file, "a") as f:
                    f.write(sentence.replace('\n', '') + "\n")
            else:
                print("Duplicate or failed sentence. Skipping...")
    except KeyboardInterrupt:
        print("\nGeneration stopped by user. All sentences saved to:", output_file)

if __name__ == "__main__":
    main()
