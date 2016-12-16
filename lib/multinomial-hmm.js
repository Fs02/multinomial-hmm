module.exports = class MultinomialHMM {
  constructor(initial, transition, emission) {
    this.initial = initial
    this.transition = transition
    this.emission = emission
  }

  predict(observations) {
    return MultinomialHMM.viterbi(
      observations,
      Object.keys(this.transition),
      this.initial,
      this.transition,
      this.emission
    ).states
  }

  static viterbi(observations, states, initial, transition, emission) {
    let V = [{}]
    for (let st of states) {
      if (initial[st] == undefined) {
        initial[st] = 1
      }
      V[0][st] = {
        prob: initial[st] * emission[st][observations[0]],
        prev: undefined
      }
    }

    // Run Viterbi when t > 0
    for (let t = 1; t < observations.length; ++t) {
      V.push({})
      for (let st of states) {
        // find max transition probability
        let max_tr_prob = 0
        let max_prob = 0
        let max_prev_st = undefined
        for (let prev_st of states) {
          if (emission[st][observations[t]] == undefined) {
            V[t][st] = {
              prob: 0.001,
              prev: prev_st
            }
            break
          } else {
            let tr_prob = V[t-1][prev_st]["prob"] * transition[prev_st][st]
            if (tr_prob > max_tr_prob) {
              max_tr_prob = tr_prob
              max_prob = tr_prob * emission[st][observations[t]]
              max_prev_st = prev_st
            }
            V[t][st] = {
              prob: max_prob,
              prev: max_prev_st
            }
          }
        }
      }
    }

    let opt = []
    // The highest probability
    let max_prob = 0
    let previous = undefined
    let last_V = V[V.length-1]
    for (let st of Object.keys(last_V)) {
      if (last_V[st]["prob"] > max_prob) {
        max_prob = last_V[st]["prob"]
        previous = st
      }
    }
    opt.push(previous)
    // Follow the backgrack
    for (let t = V.length - 2; t >= 0; --t) {
      previous = V[t+1][previous]["prev"]
      opt.unshift(previous)
    }

    return {states: opt, prob: max_prob}
  }
}
