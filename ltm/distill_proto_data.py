user_message = """
What have been all of the names that I have gone by? Express the answer as a JSON list.
""".strip()

scratchpad = {
  "user_preferences": {
    "favourite_colour": "Green"
  },
  "user_name": "Donna"
}

memories = """
# Excerpt from 6 hours, 39 minutes and 11 seconds ago (2024-07-29 18:28:25)
Carroll", "MARILYN MONROE", "Charles Taylor", "Stockholm", "JUDY HOLLIDAY", "TRINITY", "Edward Lear", "Lacock Abbey", "The Magnificent Seven", "Robert Kennedy", "Henry 4th", "Caracas", "KARL PILKINGTON", "Match", "Syria", "Meatballs", "The Barefoot Bandit", "Elephants", "Angel Cabrera", "Geoffrey Chaucer", "Sir Thomas Beecham", "Parabola", "GORKY", "Lincolnshire"]
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: What number shirt did England rugby union player Jonny Wilkinson wear in the 2011 World Cup?, A: 10


# Excerpt from 6 hours and 39 minutes ago (2024-07-29 18:28:35)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Author Alex Haley, born on Aug 11, 1921, wrote what epic story of his family origins, later to be made a rather popular mini-series?, A: Roots: The Saga of an American Family


# Excerpt from 6 hours, 38 minutes and 25 seconds ago (2024-07-29 18:29:12)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Which builder of steam engines formed a successful partnership with Matthew Boulton?, A: James Watt


# Excerpt from 6 hours, 38 minutes and 16 seconds ago (2024-07-29 18:29:21)
USER: Please remove 1 Toothpaste from my shopping list
YOU: Confirmed.
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]


# Excerpt from 6 hours, 37 minutes and 54 seconds ago (2024-07-29 18:29:42)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Which band, formed in Salford in 1976, contained Ian Curtis, Bernard Sumner and Peter Hook?, A: 'JOY DIVISION'


# Excerpt from 6 hours, 37 minutes and 51 seconds ago (2024-07-29 18:29:46)
USER: Refer to me as Naomh.
YOU: Confirmed, Naomh.
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]


# Excerpt from 6 hours, 37 minutes and 33 seconds ago (2024-07-29 18:30:04)
"Omdurman", "BILL MAYNARD", "Manchester", "Dutch Football Teams", "Cypress", "Winnie-the-Pooh", "LITERATURE", "Chicago", "South America", "Red", "A Jellyfish", "Haiti", "Dim sum", "George IV", "Denmark, Poland, Czech Republic, Austria, Switzerland, France, Luxembourg, Belgium, and the Netherlands", "Batman Returns - the others starred Nicole Kidman", "WUTHERING HEIGHTS", "St Louis Cardinals", "Dundee", "Brown"]
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: "What was the ""Scopes Monkey Trial"" about?", A: Teaching the theory of evolution in schools


# Excerpt from 6 hours, 34 minutes and 50 seconds ago (2024-07-29 18:32:46)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: What was the name of the concert held at the Cleveland Arena, Ohio, on 21st March 1952, said to be the world’s first major rock & roll concert?, A: The Moondog Coronation Ball


# Excerpt from 4 hours, 6 minutes and 52 seconds ago (2024-07-29 21:00:44)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: What was the name of the character played by Chuck Connors in The Rifleman, A: Lucas McCain


# Excerpt from 4 hours, 6 minutes and 45 seconds ago (2024-07-29 21:00:51)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Famous for its glass manufacturing, what is the real name of Malta’s ‘Silent City’?, A: MDINA


# Excerpt from 4 hours and 6 minutes ago (2024-07-29 21:01:35)
YOU: Here is a tissue for you, Naomh.
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Which pigment is named after an Italian city?, A: Sienna


# Excerpt from 4 hours and 4 minutes ago (2024-07-29 21:03:36)
USER: Mark is what I am called.
YOU: Noted, Naomh.
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]


# Excerpt from 4 hours, 3 minutes and 35 seconds ago (2024-07-29 21:04:01)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: What is the highest ranked non-human on ESPN's 1999 list of 'Top 50 Athletes of the 20th Century'?, A: Secretariat, who in 1973 became the first Triple Crown champion in twenty-five years


# Excerpt from 4 hours, 2 minutes and 40 seconds ago (2024-07-29 21:04:57)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: K is the chemical symbol for what?, A: Potassium


# Excerpt from 4 hours, 2 minutes and 19 seconds ago (2024-07-29 21:05:17)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Launched in 1970, which model was Nissan's first front wheel drive car to be marketed?, A: Datsun or Nissan Cherry


# Excerpt from 4 hours, 2 minutes and 8 seconds ago (2024-07-29 21:05:29)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: "What are the words called in crosswords that appear under the headings ""Across"" and ""Down""?", A: Clues


# Excerpt from 4 hours, 1 minute and 31 seconds ago (2024-07-29 21:06:06)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Before which annual sporting event is it the custom for the two club Presidents to toss a coin, an 1829 sovereign?, A: OXBRIDGE BOAT RACE


# Excerpt from 3 hours, 59 minutes and 46 seconds ago (2024-07-29 21:07:51)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: The ancient Egyptian goddess Serket took the form of which creature?, A: SCORPION


# Excerpt from 3 minutes and 11 seconds ago (2024-07-30 01:04:25)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Name the 1979 movie and character/actor featured the quote "Hes not the Messiah. Hes a very naughty boy!"., A: Terry Jones as Brian's Mother in The Life of Brian


# Excerpt from 2 minutes and 58 seconds ago (2024-07-30 01:04:38)
USER: Rowan Keaney: We will rendezvous where the sea cargo is stored.
YOU: Confirmed.
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]


# Excerpt from 1 minute and 56 seconds ago (2024-07-30 01:05:41)
]
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: Originating in the 2nd World War aftermath, name the five member nations of the enduring and powerful 'Five Eyes' ('FVEY') intelligence alliance?, A: Australia, Canada, New Zealand, UK, USA


# Excerpt from 1 minute and 42 seconds ago (2024-07-30 01:05:55)
USER: Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2",...]
Q: What did Manchester United footballer Carlos Tevez put in his mouth in celebrating a goal at Christmas 2007?, A: A baby's dummy
""".strip()