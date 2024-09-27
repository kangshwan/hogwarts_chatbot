import random
class SortingHat:
    def __init__(self):
        self.questions = [
            "How do you approach challenges and obstacles, and what do you think defines bravery?",
            "When working with others, do you prefer to lead, follow, or collaborate, and why?",
            "In a difficult situation, would you prioritize being fair or being kind, and how would you balance the two?",
            "How do you stay motivated when pursuing your goals, and what role does ambition play in your life?",
            "When faced with rules or expectations, do you tend to follow them, question them, or create your own? Why?",
            "Which do you value more: being honest or being kind? Why?",
            "Do you rely more on intuition or analysis when making decisions? Why?",
            "How do you make sure your ideas and opinions are heard in group discussions?",
            "What role do you usually take in group projects or collaborations? Why?",
            "How do you respond to failure, and what does it teach you?",
            "What principles guide you in making difficult decisions?",
            "How do you stay grounded while striving for excellence in your work or studies?"
        ]
        self.answers = [
            "So many minds, so many dreams, each one yearning for greatness in their own way.",
            "Such complexity in this one—more layers than a potion brewed at midnight",
            "Curious, this mind holds secrets even they haven’t discovered yet.",
            "I sense a fire within, simmering beneath a calm surface.",
            "The potential here is boundless, but will they see it themselves?",
            "Resilient, adaptable, yet a hint of doubt clouds their path.",
            "A mind as sharp as a blade, cutting through the fog of uncertainty.",
            "So many roads they could walk, each with its own unique challenge.",
            "A heart full of dreams and a spirit unbroken—what a combination.",
            "There’s a quiet strength in this one, hidden but undeniable.",
            "The choices ahead will shape their destiny more than they know.",
            "Determined, but with a trace of uncertainty—an intriguing combination.",
            "They carry a spark of mischief, hidden beneath a composed exterior.",
            "A mind that seeks adventure, but a heart that yearns for belonging."
        ]

    def choose_questions(self):
        self.random_samples = random.sample(self.questions, 5)
        return self.random_samples
    
    def choose_quotes(self):
        self.random_quotes = random.sample(self.answers, 5)
        return self.random_quotes

