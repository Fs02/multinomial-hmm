var MultinomialHMM = require("multinomial-hmm")

// Example from wikipedia
states = ['Healthy', 'Fever']
observations = ['normal', 'cold', 'dizzy']
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
  'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
  'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
}
emission_probability = {
  'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
  'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
}

let hmm = new MultinomialHMM(start_probability, transition_probability, emission_probability)
console.log(hmm.predict(observations));

console.log(MultinomialHMM.viterbi(observations, states, start_probability, transition_probability, emission_probability))
